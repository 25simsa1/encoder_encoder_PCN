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

def chl_step(m, img_np, txt_np, mask_np, k0, k1, k2, gen_lr, latents="text", free_pi_bu=None, free_state_lr=None):
    """Contrastive-Hebbian generative step (PC-native, no backprop, no separate decoder).
    Phase 0: set the shared latents, then HOLD them fixed (unclamped but excluded from the
    decode loops) for both phases below, so the contrast varies only the image.
      latents="text" (default): caption clamped, image=zeros unclamped, relax k0 -> text-set
        latents (the original CHL; ill-posed target, the conditional mean over images).
      latents="image": BOTH clamped (recon config), relax k0 -> IMAGE-set latents. The
        latent-autoencoder variant: the latents identify the image, so the conditional mean
        given them IS the true image and the contrast trains TOP-DOWN SELF-SUFFICIENCY on a
        well-posed target. The free phase starts the image from zeros so the anti-learned
        sample is a genuine standalone top-down generation.
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

    if latents == "image":
        # Phase 0 (image-set): BOTH clamped (recon config), relax so the latents encode THIS image
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(img, txt, mask)
        for _ in range(k0):
            for L in m.trainable_layers:
                L.update_state()
        # the free phase must be a standalone top-down generation, so start the image from zeros
        m.img_input.set_state(tf.zeros_like(img))
    else:
        # Phase 0: text-drive the shared latents (image zeros, unclamped, full relax)
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(tf.zeros_like(img), txt, mask)
        m.img_input.is_clamped = False
        for _ in range(k0):
            for L in m.trainable_layers:
                L.update_state()
            m.img_input.update_state()
    # latents are now set; they stay unclamped but are excluded from the decode loops (fixed)

    # FREE phase: latents fixed, image FREE, relax the decode, then ANTI-LEARN.
    # free_pi_bu / free_state_lr (both None = byte-identical default) run this phase
    # top-down-dominant at an adequate rate, so the free sample actually EXPRESSES the
    # current top-down cascade (rate-starved at the built state_lr it stays ~zero and the
    # contrast has nothing to calibrate against).
    m.img_input.is_clamped = False
    if free_pi_bu is not None:
        for L in decode:
            L.pi_bu = free_pi_bu
    orig_slr = None
    if free_state_lr is not None:
        orig_slr = [(L, L.state_lr) for L in decode] + [(m.img_input, m.img_input.state_lr)]
        for L, _ in orig_slr:
            L.state_lr = free_state_lr
    for _ in range(k1):
        for L in decode:
            L.update_state()
        m.img_input.update_state()
    if free_pi_bu is not None:
        for L in decode:
            L.pi_bu = 1.0
    if orig_slr is not None:
        for L, s in orig_slr:
            L.state_lr = s
    _weight_step(-gen_lr)

    # CLAMPED phase: latents fixed, img_input clamped to the TRUE image, relax the decode, then LEARN
    m.img_input.set_state(img); m.img_input.is_clamped = True
    for _ in range(k2):
        for L in decode:
            L.update_state()
    _weight_step(+gen_lr)

    # restore the recon clamp config for the next recon step
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True

def ebm_step(m, img_np, txt_np, mask_np, k0, k1, k2, gen_lr, noise_t0):
    """Contrastive-divergence EBM step (PC-native): CHL with a NOISY CD-1-from-data negative
    phase. Same structure as chl_step; the only change is the negative phase starts at the TRUE
    image and relaxes with ANNEALED Langevin noise (a model SAMPLE) instead of the deterministic
    mean, so the local contrast (anti-learn the sample, learn the data) sharpens the energy into a
    proper EBM. Sampling is the noise term in update_state; the weight rule is the class's own
    local update_wts. No backprop, no separate decoder. Ends in the recon clamp config."""
    T = tf.convert_to_tensor
    img = T(img_np); txt = T(txt_np); mask = T(mask_np)
    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]
    noised = decode + [m.img_input]   # the free states sampled in the negative phase

    def _weight_step(signed_lr):
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

    # NEGATIVE phase (CD-1 from data): start at the true image, free it, noisy-relax the decode
    # with an annealed temperature to draw a SAMPLE, then ANTI-LEARN.
    m.img_input.set_state(img); m.img_input.is_clamped = False
    for i in range(k1):
        t = noise_t0 * (1.0 - i / float(max(1, k1)))   # linear anneal T0 -> ~0
        for L in noised:
            L.noise_temp = t
        for L in decode:
            L.update_state()
        m.img_input.update_state()
    for L in noised:
        L.noise_temp = 0.0                              # reset so the clamped/recon relax is noise-free
    _weight_step(-gen_lr)

    # POSITIVE phase: latents fixed, img_input clamped to the TRUE image, deterministic relax, LEARN
    m.img_input.set_state(img); m.img_input.is_clamped = True
    for _ in range(k2):
        for L in decode:
            L.update_state()
    _weight_step(+gen_lr)

    m.img_input.is_clamped = True; m.txt_input.is_clamped = True

def diffusion_step(m, img_np, txt_np, mask_np, k0, k2, gen_lr, sigma_t):
    """Diffusion denoising step (PC-native): encode a NOISED image x_t = x_0 + sigma_t*eps together
    with the caption into the latent, then take the class's local weight step to train the decode
    toward the CLEAN x_0. A denoising autoencoder, sharp for small sigma_t; positive-only (no
    contrast). Because the latent is conditioned on x_t (not only the under-determined caption
    latent), the target is sharp for small noise. No backprop, no separate decoder, no optimizer."""
    T = tf.convert_to_tensor
    img = T(img_np); txt = T(txt_np); mask = T(mask_np)
    x_t = img + sigma_t * tf.random.normal(img.shape)
    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]

    def _weight_step(signed_lr):
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

    # ENCODE phase: clamp the NOISED image x_t + caption, relax all states so the latent encodes x_t
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(x_t, txt, mask)
    for _ in range(k0):
        for L in m.trainable_layers:
            L.update_state()
    # latents now encode x_t + caption; held fixed (excluded from the decode loop below)

    # TARGET phase: clamp img_input to the CLEAN x_0, relax the decode (latent fixed), learn toward x_0
    m.img_input.set_state(img); m.img_input.is_clamped = True
    for _ in range(k2):
        for L in decode:
            L.update_state()
    _weight_step(+gen_lr)

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
    ap.add_argument("--train-mode", default="recon", choices=["recon", "gen", "chl", "ebm", "diffusion"])
    ap.add_argument("--gen-every", type=int, default=1)
    ap.add_argument("--gen-relax-k0", type=int, default=None)   # phase-0 (set the latent) relax steps
    ap.add_argument("--gen-relax-k1", type=int, default=None)
    ap.add_argument("--gen-relax-k2", type=int, default=None)
    ap.add_argument("--gen-lr", type=float, default=None)   # gentler rate for the generative weight step only
    ap.add_argument("--gen-latents", default="text", choices=["text", "image"])   # chl phase-0 latent source
    ap.add_argument("--free-pi-bu", type=float, default=None)      # chl free-phase generative precision; None = default
    ap.add_argument("--free-state-lr", type=float, default=None)   # chl free-phase relaxation rate; None = as built
    ap.add_argument("--weight-norm", action="store_true")   # PC-native weight-norm stabilizer on conv/dense layers
    ap.add_argument("--hf-weight", type=float, default=0.0)   # high-frequency boost on the bottom pixel error
    ap.add_argument("--noise-temp", type=float, default=0.0)  # initial Langevin temperature for the ebm negative phase
    ap.add_argument("--diff-levels", type=int, default=10)    # diffusion noise levels
    ap.add_argument("--diff-sigma-min", type=float, default=0.05)
    ap.add_argument("--diff-sigma-max", type=float, default=0.8)
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
    if a.hf_weight > 0:
        nhf = 0
        for L in m.trainable_layers:
            if isinstance(L, Conv2DPCNLayer) and getattr(L, "prev_layer", None) is m.img_input:
                L.hf_gamma = a.hf_weight; nhf += 1
        print(f"hf_weight={a.hf_weight} set on {nhf} bottom conv layer(s)", flush=True)
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

    WN_W = []
    wn_mgr = wn_best_mgr = None
    if a.weight_norm:
        nwn = 0
        for L in m.trainable_layers:
            if hasattr(L, "enable_weight_norm") and getattr(L, "wts", None) is not None:
                L.enable_weight_norm(); nwn += 1
        WN_W = [L.g_mag for L in m.trainable_layers if getattr(L, "weight_norm", False)]
        print(f"weight_norm enabled on {nwn} conv/dense layers", flush=True)
        if WN_W:
            wn_ckpt = tf.train.Checkpoint(**{f"g{i}": v for i, v in enumerate(WN_W)})
            wn_mgr = tf.train.CheckpointManager(wn_ckpt, a.ckpt + "_wn", max_to_keep=1)
            wn_best_mgr = tf.train.CheckpointManager(wn_ckpt, a.ckpt + "_best_wn", max_to_keep=1)
            if a.resume and wn_mgr.latest_checkpoint:
                wn_ckpt.restore(wn_mgr.latest_checkpoint)   # restore trained magnitudes over the ||wts||-derived ones
                print("resumed wn", wn_mgr.latest_checkpoint, flush=True)

    def save_latest():
        mgr.save()
        if wn_mgr is not None: wn_mgr.save()
    def save_best():
        best_mgr.save()
        if wn_best_mgr is not None: wn_best_mgr.save()

    diff_sigmas = np.geomspace(a.diff_sigma_min, a.diff_sigma_max, a.diff_levels).astype(np.float32)
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
                         a.gen_lr or a.lr, a.gen_latents, a.free_pi_bu, a.free_state_lr)
            elif a.train_mode == "ebm" and step % a.gen_every == 0:
                ebm_step(m, img[bi], txt[bi], mask[bi],
                         a.gen_relax_k0 or a.relax, a.gen_relax_k1 or a.relax, a.gen_relax_k2 or a.relax,
                         a.gen_lr or a.lr, a.noise_temp)
            elif a.train_mode == "diffusion" and step % a.gen_every == 0:
                sig = float(diff_sigmas[np.random.randint(len(diff_sigmas))])
                diffusion_step(m, img[bi], txt[bi], mask[bi],
                               a.gen_relax_k0 or a.relax, a.gen_relax_k2 or a.relax, a.gen_lr or a.lr, sig)
            step += 1
            if step % a.energy_every == 0:
                e, mx = energy_stats(m)
                msg = f"[step {step} ep {ep}] energy={e:.5f} max_abs_state={mx:.3f} ({(time.time()-t0)/step:.2f}s/step)"
                if a.infonce_lambda > 0:
                    msg += f" infonce_acc={ia:.3f} infonce_loss={il:.4f}"
                print(msg, flush=True)
                if np.isfinite(e) and e < best_e:
                    best_e = e; save_best(); print(f"best @ {step} energy={e:.5f}", flush=True)
                if not np.isfinite(e) or not np.isfinite(mx) or mx > 1e6:
                    print(f"DIVERGED at step {step}", flush=True); save_latest(); return
            if step % 1000 == 0:
                save_latest(); print(f"ckpt @ {step}", flush=True)
    save_latest(); print("TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()
