"""Overfit the bidirectional class on a COCO64 subset with its relaxed PC schedule.
The optimizer is the class's own beta-less LARS-on-weights; this script only sets the
learning rate and runs the schedule. Logs a PC-energy proxy + max|state| for divergence."""
import os, argparse, time
import tensorflow as tf, numpy as np
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from conv_pcn_layer import Conv2DPCNLayer
from pcn_config import COCO64_156M
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
    ap.add_argument("--infonce-lambda", type=float, default=0.0)
    ap.add_argument("--infonce-tau", type=float, default=0.07)
    a = ap.parse_args()

    img, txt, mask = D.load_batch(a.pairs, seed=0)
    print(f"data: img{img.shape} txt{txt.shape} mask{mask.shape}", flush=True)
    m = EncoderEncoderPCN(a.lr, config=COCO64_156M)
    if np.isfinite(a.state_clip):
        nclip = 0
        for L in m.trainable_layers:
            if hasattr(L, "state_clip"):
                L.state_clip = a.state_clip; nclip += 1
        print(f"state_clip = {a.state_clip} set on {nclip} layers", flush=True)
    if a.conv_activation != "relu":
        nc = 0
        for L in m.trainable_layers:
            if isinstance(L, Conv2DPCNLayer):
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
