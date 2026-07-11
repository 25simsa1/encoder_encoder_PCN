"""Overfit the bidirectional class on a COCO64 subset with its relaxed PC schedule.
The optimizer is the class's own beta-less LARS-on-weights; this script only sets the
learning rate and runs the schedule. Logs a PC-energy proxy + max|state| for divergence."""
import os, argparse, time
import tensorflow as tf, numpy as np
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from conv_pcn_layer import Conv2DPCNLayer
from pcn_config import COCO64_156M, COCO64_GEN
from infonce import infonce_grads
import coco64_data as D

def energy_stats(m):
    """max|state| (divergence) + mean per-layer forward prediction error (PC energy proxy)."""
    max_abs, total_err, n = 0.0, 0.0, 0
    for L in m.trainable_layers:
        s = getattr(L, "state", None)
        if s is None:
            continue
        max_abs = max(max_abs, float(tf.reduce_max(tf.abs(s))))
        prev = getattr(L, "prev_layer", None)
        if prev is not None:
            try:
                err = float(tf.reduce_mean(tf.square(L(prev.predict_next()) - L.predict_next())))
                total_err += err; n += 1
            except Exception:
                pass
    return (total_err / n if n > 0 else float("nan")), max_abs

def generative_step(m, img_np, txt_np, mask_np, k1, k2, gen_lr=None):
    """PC-native generative step. Phase 1: caption clamped, image = zeros unclamped, relax k1
    so the shared latents become text-driven. Phase 2: clamp the TRUE image; leave the latents
    UNCLAMPED but do NOT relax them (fixed text-driven top-down sources), relax the image decode
    intermediates k2 to bridge, then the existing local weight step. Only clamp + update_state +
    update_wts; no backprop, no separate decoder. Ends in the recon clamp configuration.
    gen_lr (optional): a gentler learning rate applied to the decode layers for THIS weight step
    only (restored after), so the generative step can train the decode over many epochs without
    degrading reconstruction. None keeps each layer's own rate."""
    T = tf.convert_to_tensor
    img = T(img_np); txt = T(txt_np); mask = T(mask_np)
    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]

    # Phase 1: text-drive the shared latents (image = zeros, unclamped)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.zeros_like(img), txt, mask)
    m.img_input.is_clamped = False
    for _ in range(k1):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()

    # Phase 2: clamp the true image; latents stay unclamped but fixed (not relaxed);
    # relax the decode intermediates, then the local weight step
    m.img_input.set_state(img); m.img_input.is_clamped = True
    for _ in range(k2):
        for L in decode:
            L.update_state()
    # weight step, optionally with a gentler rate on the decode layers (restored after)
    _orig = None
    if gen_lr is not None:
        _orig = [(L, L.learning_rate, getattr(L, "bias_lr", None)) for L in decode]
        for L in decode:
            L.learning_rate = gen_lr
            if hasattr(L, "bias_lr"):
                L.bias_lr = gen_lr
    for L in decode:
        L.update_wts(); L.update_b()
    if _orig is not None:
        for L, lr0, blr0 in _orig:
            L.learning_rate = lr0
            if blr0 is not None:
                L.bias_lr = blr0

    # restore the recon clamp config (image+text clamped, latents unclamped) for the next recon step
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True

def chl_step(m, img_np, txt_np, mask_np, k0, k1, k2, gen_lr):
    """Contrastive-Hebbian generative step (PC-native, no backprop, no separate decoder).
    Phase 0: caption clamped, image=zeros unclamped, relax k0 so the shared latents become
    text-set; then HOLD the latents fixed (unclamped but excluded from the decode loops) for
    both phases below, so the contrast varies only the image.
    FREE phase: latents fixed, image FREE, relax the decode k1 -> the standalone generation;
    anti-learn it (decode weight step with -gen_lr, weight decay off).
    CLAMPED phase: same fixed latents, img_input clamped to the TRUE image, relax the decode
    k2 -> the target; learn it (decode weight step with +gen_lr, weight decay off).
    Net decode update: wts -= gen_lr*(g_clamped - g_free). Ends in the recon clamp config."""
    T = tf.convert_to_tensor
    img = T(img_np); txt = T(txt_np); mask = T(mask_np)
    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]

    def _weight_step(signed_lr):
        # local weight step on the decode with a signed rate and weight decay OFF, restored after
        orig = [(L, L.learning_rate, getattr(L, "bias_lr", None), getattr(L, "weight_decay", None)) for L in decode]
        for L in decode:
            L.learning_rate = signed_lr
            if hasattr(L, "bias_lr"):
                L.bias_lr = signed_lr
            if hasattr(L, "weight_decay"):
                L.weight_decay = 0.0
        for L in decode:
            L.update_wts(); L.update_b()
        for L, lr0, blr0, wd0 in orig:
            L.learning_rate = lr0
            if blr0 is not None:
                L.bias_lr = blr0
            if wd0 is not None:
                L.weight_decay = wd0

    # Phase 0: text-drive the shared latents (image zeros, unclamped, full relax)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.zeros_like(img), txt, mask)
    m.img_input.is_clamped = False
    for _ in range(k0):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()
    # latents are now text-set; they stay unclamped but are excluded from the decode loops (fixed)

    # FREE phase: latents fixed, image FREE, relax the decode, then ANTI-LEARN
    m.img_input.is_clamped = False
    for _ in range(k1):
        for L in decode:
            L.update_state()
        m.img_input.update_state()
    _weight_step(-gen_lr)

    # CLAMPED phase: latents fixed, img_input clamped to the TRUE image, relax the decode, then LEARN
    m.img_input.set_state(img); m.img_input.is_clamped = True
    for _ in range(k2):
        for L in decode:
            L.update_state()
    _weight_step(+gen_lr)

    # restore the recon clamp config for the next recon step
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True

def infonce_relax_step(m, img, txt, mask, relax, lam, tau):
    """Relax-then-step with an InfoNCE error injected at the deepest branch codes each
    relax substep, then the existing local LARS weight step. Eager (does not use the
    compiled sweep). Returns (infonce_acc, infonce_loss)."""
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    ui, vi = m._infonce_codes
    acc = tf.constant(0.0); loss = tf.constant(0.0)
    for _ in range(relax):
        for L in m.trainable_layers:
            L.update_state()
        du, dv, acc, loss = infonce_grads(ui.state, vi.state, tau)
        ui.state.assign_sub(ui.state_lr * lam * du)
        vi.state.assign_sub(vi.state_lr * lam * dv)
    for L in m.trainable_layers:
        L.update_wts(); L.update_b()
    return float(acc), float(loss)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2000); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-2); ap.add_argument("--relax", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--ckpt", default="ckpt_coco64")
    ap.add_argument("--energy-every", type=int, default=50); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--state-clip", type=float, default=float("inf"))  # cap |state| after relaxation; inf = off
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--trust-cap", type=float, default=float("inf"))
    ap.add_argument("--conv-activation", default="relu")
    ap.add_argument("--config", default="coco64_156m", choices=["coco64_156m", "coco64_gen"])
    ap.add_argument("--infonce-lambda", type=float, default=0.0)
    ap.add_argument("--infonce-tau", type=float, default=0.07)
    ap.add_argument("--train-mode", default="recon", choices=["recon", "gen", "chl"])
    ap.add_argument("--gen-every", type=int, default=1)
    ap.add_argument("--gen-relax-k0", type=int, default=None)   # phase-0 (set the latent) relax steps
    ap.add_argument("--gen-relax-k1", type=int, default=None)
    ap.add_argument("--gen-relax-k2", type=int, default=None)
    ap.add_argument("--gen-lr", type=float, default=None)   # gentler rate for the generative weight step only
    a = ap.parse_args()

    img, txt, mask = D.load_batch(a.pairs, seed=0)
    print(f"data: img{img.shape} txt{txt.shape} mask{mask.shape}", flush=True)
    CONFIGS = {"coco64_156m": COCO64_156M, "coco64_gen": COCO64_GEN}
    m = EncoderEncoderPCN(a.lr, config=CONFIGS[a.config])
    print(f"config={a.config}", flush=True)
    if np.isfinite(a.state_clip):
        nclip = 0
        for L in m.trainable_layers:
            if hasattr(L, "state_clip"):
                L.state_clip = a.state_clip; nclip += 1
        print(f"state_clip = {a.state_clip} set on {nclip} layers", flush=True)
    if a.conv_activation != "relu":
        nc = 0
        for L in m.trainable_layers:
            if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
                L.activation = a.conv_activation; nc += 1
        print(f"conv_activation={a.conv_activation} set on {nc} conv layers", flush=True)
    for L in m.trainable_layers:
        if hasattr(L, "weight_decay"):
            L.weight_decay = a.weight_decay
        if hasattr(L, "trust_cap"):
            L.trust_cap = a.trust_cap
    print(f"weight_decay={a.weight_decay} trust_cap={a.trust_cap}", flush=True)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    # realize weights so they can be checkpointed
    b0 = slice(0, a.batch)
    m.pass_through(tf.convert_to_tensor(img[b0]), tf.convert_to_tensor(txt[b0]), tf.convert_to_tensor(mask[b0]))
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ckpt = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    mgr = tf.train.CheckpointManager(ckpt, a.ckpt, max_to_keep=1)
    best_mgr = tf.train.CheckpointManager(ckpt, a.ckpt + "_best", max_to_keep=1)  # lowest-energy weights
    best_e = float("inf")
    if a.resume and mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint); print("resumed", mgr.latest_checkpoint, flush=True)

    N = img.shape[0]; step = 0; t0 = time.time()
    ia, il = 0.0, 0.0
    for ep in range(a.epochs):
        order = np.random.default_rng(ep).permutation(N)
        for s in range(0, N - a.batch + 1, a.batch):
            bi = order[s:s + a.batch]
            m.img_input.is_clamped = True; m.txt_input.is_clamped = True
            m.pass_through(tf.convert_to_tensor(img[bi]), tf.convert_to_tensor(txt[bi]), tf.convert_to_tensor(mask[bi]))
            if a.infonce_lambda > 0:
                ia, il = infonce_relax_step(m, tf.convert_to_tensor(img[bi]), tf.convert_to_tensor(txt[bi]),
                                            tf.convert_to_tensor(mask[bi]), a.relax, a.infonce_lambda, a.infonce_tau)
            else:
                m.update_states_wts_b_relaxed(num_weight_steps=1, num_relax_steps=a.relax)
            if a.train_mode == "gen" and step % a.gen_every == 0:
                generative_step(m, img[bi], txt[bi], mask[bi],
                                a.gen_relax_k1 or a.relax, a.gen_relax_k2 or a.relax, a.gen_lr)
            elif a.train_mode == "chl" and step % a.gen_every == 0:
                chl_step(m, img[bi], txt[bi], mask[bi],
                         a.gen_relax_k0 or a.relax, a.gen_relax_k1 or a.relax, a.gen_relax_k2 or a.relax,
                         a.gen_lr or a.lr)
            step += 1
            if step % a.energy_every == 0:
                e, mx = energy_stats(m)
                msg = f"[step {step} ep {ep}] energy={e:.5f} max_abs_state={mx:.3f} ({(time.time()-t0)/step:.2f}s/step)"
                if a.infonce_lambda > 0:
                    msg += f" infonce_acc={ia:.3f} infonce_loss={il:.4f}"
                print(msg, flush=True)
                if np.isfinite(e) and e < best_e:
                    best_e = e; best_mgr.save(); print(f"best @ {step} energy={e:.5f}", flush=True)
                if not np.isfinite(e) or not np.isfinite(mx) or mx > 1e6:
                    print(f"DIVERGED at step {step}", flush=True); mgr.save(); return
            if step % 1000 == 0:
                mgr.save(); print(f"ckpt @ {step}", flush=True)
    mgr.save(); print("TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()
