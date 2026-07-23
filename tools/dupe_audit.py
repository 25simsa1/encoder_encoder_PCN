"""Near-duplicate leakage audit for the E1L 20k held-out coupling hits. Loads the E1L
checkpoint + the 20k-original split, recomputes eval-pool latent retrieval, identifies the
top-1 hit items, and for each eval item finds its nearest TRAIN image (pixel L2). If hits'
nearest-train distances are systematically tiny vs non-hits, the 'generalization' is
near-duplicate memorization leakage. Saves a grid of each hit beside its nearest train
image. Throwaway instrument."""
import argparse, numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from mechanism_probe import build_vocab, encode_caps, load_P, enc_img, enc_txt, l2n, CAPLEN
DATA = "/home/slsang29/coco_scale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ntrain", type=int, default=20000); ap.add_argument("--neval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--coco", default="train2017")
    ap.add_argument("--out", default="dupe_audit.png")
    a = ap.parse_args()
    imgs = np.load(f"{DATA}/imgs_sc_{a.coco}.npy", mmap_mode="r")
    caps = open(f"{DATA}/caps_sc_{a.coco}.txt").read().split("\n")[:len(imgs)]
    N = len(imgs); perm = np.random.RandomState(a.seed + 1).permutation(N)
    tr = perm[:a.ntrain]; ev = perm[a.ntrain:a.ntrain + a.neval]
    chars, c2i = build_vocab([caps[i] for i in tr]); toks = encode_caps(caps, c2i, CAPLEN)
    P = load_P(a.ckpt)

    # eval latents (concat taps, per-tap l2n as the driver's latents() does on the concat --
    # here match the driver: l2n over the FULL concat)
    def latents(idx, bs=200):
        Z_i, Z_t = [], []
        for s in range(0, len(idx), bs):
            bi = idx[s:s + bs]
            it = enc_img(P, tf.constant(imgs[bi].astype("float32")))
            tt = enc_txt(P, tf.constant(toks[bi]))
            Z_i.append(tf.concat(it, 1).numpy()); Z_t.append(tf.concat(tt, 1).numpy())
        return l2n(np.concatenate(Z_i, 0)), l2n(np.concatenate(Z_t, 0))
    zi, zt = latents(ev)
    S = zi @ zt.T
    t2i_hit = np.argmax(S, 0) == np.arange(len(ev))     # text -> image top-1 (the paper metric family)
    i2t_hit = np.argmax(S, 1) == np.arange(len(ev))
    hits = np.where(t2i_hit | i2t_hit)[0]
    print(f"eval hits: t2i={int(t2i_hit.sum())} i2t={int(i2t_hit.sum())} union={len(hits)} / {len(ev)} (chance ~1 each)", flush=True)

    # nearest-train pixel distance for every eval item (chunked GPU matmul on flattened pixels)
    evf = tf.constant(imgs[ev].reshape(len(ev), -1).astype("float32"))
    ev_sq = tf.reduce_sum(evf * evf, 1, keepdims=True)
    best_d = np.full(len(ev), np.inf, "float32"); best_j = np.zeros(len(ev), "int64")
    CH = 1000
    for s in range(0, len(tr), CH):
        bj = tr[s:s + CH]
        trf = tf.constant(imgs[bj].reshape(len(bj), -1).astype("float32"))
        d = (ev_sq + tf.reduce_sum(trf * trf, 1)[None, :] - 2.0 * evf @ tf.transpose(trf)).numpy()
        m = d.min(1); am = d.argmin(1)
        upd = m < best_d
        best_d[upd] = m[upd]; best_j[upd] = bj[am[upd]]
        if (s // CH) % 5 == 0: print(f"  nn {s + CH}/{len(tr)}", flush=True)
    best_rmse = np.sqrt(np.maximum(best_d, 0) / evf.shape[1])   # per-pixel rmse in [0,1] units

    nh = np.setdiff1d(np.arange(len(ev)), hits)
    print(f"nearest-train per-pixel RMSE: hits mean={best_rmse[hits].mean():.4f} median={np.median(best_rmse[hits]):.4f}"
          f" | non-hits mean={best_rmse[nh].mean():.4f} median={np.median(best_rmse[nh]):.4f}", flush=True)
    for q in [0.05, 0.10]:
        thr = np.quantile(best_rmse, q)
        frac = (best_rmse[hits] <= thr).mean() if len(hits) else 0.0
        print(f"  hits in the closest {int(q*100)}% of eval-to-train distances: {frac:.2f} (chance {q:.2f})", flush=True)
    for i in hits:
        print(f"  hit ev[{i}] rmse={best_rmse[i]:.4f}  evcap='{caps[ev[i]][:60]}'  nncap='{caps[best_j[i]][:60]}'", flush=True)

    k = len(hits)
    if k:
        fig, ax = plt.subplots(2, k, figsize=(k * 1.7, 3.6))
        if k == 1: ax = ax.reshape(2, 1)
        for c, i in enumerate(hits):
            ax[0][c].imshow(imgs[ev[i]]); ax[0][c].axis("off"); ax[0][c].set_title(f"hit rmse={best_rmse[i]:.3f}", fontsize=7)
            ax[1][c].imshow(imgs[best_j[i]]); ax[1][c].axis("off")
        ax[0][0].set_ylabel("eval hit"); ax[1][0].set_ylabel("nearest train")
        plt.tight_layout(); plt.savefig(a.out, dpi=110)
        print(f"saved {a.out}", flush=True)
    print("DUPE_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
