"""Closed-form ridge solve for the dense top-down edges: the exact least-squares optimum of the
same objective the local d_pred/NLMS rule descends (its provable fixed point, fast-forwarded).
Collect the 2k forward states, per dense untied edge solve min ||(X-mx)W' - (Y-my)|| + lam||W'||
in the dual (n x n), set wts_td = W'^T and c_td = my - mx @ W'. Conv edges keep their trained
values (they are healthy). Saves the result as a _td checkpoint. Throwaway instrument."""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
import os
from pcn_config import COCO64_GEN, COCO64_WIDE
# PCN_TOOL_CONFIG=coco64_wide selects the wide-inter config; default unchanged
C = COCO64_WIDE if os.environ.get('PCN_TOOL_CONFIG') == 'coco64_wide' else COCO64_GEN
from conv_pcn_layer import Conv2DPCNLayer
from dense_pcn_layer import DensePCNLayer
import coco64_data as D
CLIP = 400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="ckpt_iso_best")
    ap.add_argument("--conv-td-ckpt", default="ckpt_u17_td")   # trained conv td values (healthy)
    ap.add_argument("--out-ckpt", default="ckpt_u18_td")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--ridge-conv", action="store_true")   # also closed-form solve stride-1 conv td edges
    a = ap.parse_args()
    img, txt, mask = D.load_batch(a.pairs, seed=0)
    T = tf.convert_to_tensor
    m = EncoderEncoderPCN(1e-4, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = CLIP
        if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
            L.activation = "gelu"
    b0 = T(np.asarray(img[:8], np.float32))
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(b0, T(np.asarray(txt[:8], np.float32)), T(np.asarray(mask[:8], np.float32)))
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    ck.restore(tf.train.latest_checkpoint(a.base_ckpt)).expect_partial()
    for L in m._image_path_layers:
        if hasattr(L, "enable_untied") and getattr(L, "wts", None) is not None:
            L.enable_untied()
    TD = [v for L in m._image_path_layers if getattr(L, "untied", False) for v in (L.wts_td, L.c_td)]
    if a.conv_td_ckpt and a.conv_td_ckpt.lower() != "none":
        tck = tf.train.Checkpoint(**{f"t{i}": v for i, v in enumerate(TD)})
        tck.restore(tf.train.latest_checkpoint(a.conv_td_ckpt)).expect_partial()
        print(f"restored base {a.base_ckpt} + conv td {a.conv_td_ckpt}", flush=True)
    else:
        # no compatible conv td checkpoint (fresh config): conv edges keep their
        # copy-init wts_td = wts, which is healthy (gains ~1 even tied)
        print(f"restored base {a.base_ckpt}; conv td = copy-init", flush=True)

    dense_edges = [L for L in m._image_path_layers
                   if isinstance(L, DensePCNLayer) and getattr(L, "untied", False)
                   and getattr(L, "prev_layer", None) is not None]
    conv_edges = [L for L in m._image_path_layers
                  if a.ridge_conv and isinstance(L, Conv2DPCNLayer)
                  and getattr(L, "untied", False) and getattr(L, "stride", 1) == 1
                  and getattr(L, "prev_layer", None) is not None
                  and getattr(L.prev_layer, "state", None) is not None
                  and len(L.prev_layer.state.shape) == 4]
    Xs = {id(L): [] for L in dense_edges}
    Ys = {id(L): [] for L in dense_edges}
    # conv edges: accumulate the ridge normal equations over im2col patches
    # (exact, no state storage; Z = patches of own state, X = prev forward target)
    CACC = {}
    for L in conv_edges:
        kh, kw = int(L.wts.shape[0]), int(L.wts.shape[1])
        cout, cin = int(L.wts.shape[-1]), int(L.wts.shape[-2])
        PD = kh * kw * cout
        CACC[id(L)] = dict(G=np.zeros((PD, PD)), R=np.zeros((PD, cin)),
                           XX=np.zeros((cin, cin)), mz=np.zeros(PD), mx=np.zeros(cin), n=0)
    n = a.pairs
    for s in range(0, n, 8):
        bi = slice(s, s + 8)
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(T(np.asarray(img[bi], np.float32)), T(np.asarray(txt[bi], np.float32)),
                       T(np.asarray(mask[bi], np.float32)))
        for _ in range(15):
            for L in m.trainable_layers:
                L.update_state()
        for L in dense_edges:
            Xs[id(L)].append((L.state - L.b).numpy())
            Ys[id(L)].append(L.prev_layer.predict_next().numpy())
        for L in conv_edges:
            acc = CACC[id(L)]
            kh, kw = int(L.wts.shape[0]), int(L.wts.shape[1])
            Z = tf.image.extract_patches(L.state, sizes=[1, kh, kw, 1], strides=[1, 1, 1, 1],
                                         rates=[1, 1, 1, 1], padding="SAME")
            Z = np.reshape(Z.numpy(), (-1, kh * kw * int(L.state.shape[-1]))).astype(np.float64)
            Xt = np.reshape(L.prev_layer.predict_next().numpy(), (-1, int(L.wts.shape[-2]))).astype(np.float64)
            acc["G"] += Z.T @ Z
            acc["R"] += Z.T @ Xt
            acc["XX"] += Xt.T @ Xt
            acc["mz"] += Z.sum(0)
            acc["mx"] += Xt.sum(0)
            acc["n"] += Z.shape[0]
        if s % 400 == 0:
            print(f"collected {s + 8}/{n}", flush=True)

    names = {id(L): f"{i:02d}" for i, L in enumerate(m._image_path_layers)}
    for L in dense_edges:
        X = np.concatenate(Xs[id(L)], 0).astype(np.float64)
        Y = np.concatenate(Ys[id(L)], 0).astype(np.float64)
        mx, my = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
        Xc, Yc = X - mx, Y - my
        # dual ridge: W' = Xc^T (Xc Xc^T + lam I)^-1 Yc  -> works for any width
        G = Xc @ Xc.T + a.lam * np.eye(X.shape[0])
        A = np.linalg.solve(G, Yc)
        Wp = Xc.T @ A                                  # (out_dim_of_L, prev_dim)
        ctd = (my - mx @ Wp).reshape(-1)
        pred = Xc @ Wp
        r2 = 1.0 - float(((Yc - pred) ** 2).sum() / ((Yc ** 2).sum() + 1e-9))
        L.wts_td.assign(tf.convert_to_tensor(Wp.T, tf.float32))   # wts_td shape (prev_dim, out)
        L.c_td.assign(tf.convert_to_tensor(ctd, tf.float32))
        print(f"edge {names[id(L)]}: X{X.shape} -> Y{Y.shape}  R2={r2:.4f}", flush=True)

    for L in conv_edges:
        acc = CACC[id(L)]
        kh, kw = int(L.wts.shape[0]), int(L.wts.shape[1])
        cout, cin = int(L.wts.shape[-1]), int(L.wts.shape[-2])
        n = acc["n"]
        mz, mx = acc["mz"] / n, acc["mx"] / n
        Gc = acc["G"] - n * np.outer(mz, mz)
        Rc = acc["R"] - n * np.outer(mz, mx)
        PD = Gc.shape[0]
        K = np.linalg.solve(Gc + a.lam * np.eye(PD), Rc)          # (D, cin): x ~ patches(y) @ K
        b = mx - mz @ K
        XXc = acc["XX"] - n * np.outer(mx, mx)
        sse = float(np.trace(XXc) - 2.0 * np.trace(K.T @ Rc) + np.trace(K.T @ Gc @ K))
        r2 = 1.0 - sse / (float(np.trace(XXc)) + 1e-9)
        g = K.reshape(kh, kw, cout, cin)
        f = np.transpose(g[::-1, ::-1], (0, 1, 3, 2))            # conv2d_transpose filter (kh,kw,cin,cout)
        # numeric convention check on the current batch: predict_prev vs the ridge map
        y = L.state
        ref = tf.nn.conv2d(y, tf.convert_to_tensor(g, tf.float32), strides=1, padding="SAME")
        L.wts_td.assign(tf.convert_to_tensor(f, tf.float32))
        L.c_td.assign(tf.convert_to_tensor(b.reshape(L.c_td.shape), tf.float32))
        got = L.predict_prev() - L.c_td
        rel = float(tf.norm(got - ref) / (tf.norm(ref) + 1e-9))
        if rel > 1e-3:
            f2 = np.transpose(g, (0, 1, 3, 2))                   # fallback: no spatial flip
            L.wts_td.assign(tf.convert_to_tensor(f2, tf.float32))
            got2 = L.predict_prev() - L.c_td
            rel2 = float(tf.norm(got2 - ref) / (tf.norm(ref) + 1e-9))
            if rel2 < rel:
                print(f"conv edge {names[id(L)]}: NOFLIP convention (rel {rel:.2e} -> {rel2:.2e})", flush=True)
                rel = rel2
            else:
                L.wts_td.assign(tf.convert_to_tensor(f, tf.float32))
        print(f"conv edge {names[id(L)]}: D={PD} n={n}  R2={r2:.4f}  conv_check={rel:.2e}", flush=True)

    out = tf.train.Checkpoint(**{f"t{i}": v for i, v in enumerate(TD)})
    mgr = tf.train.CheckpointManager(out, a.out_ckpt, max_to_keep=1)
    mgr.save()
    print(f"saved {a.out_ckpt}", flush=True)
    print("RIDGE_TD_DONE", flush=True)


if __name__ == "__main__":
    main()
