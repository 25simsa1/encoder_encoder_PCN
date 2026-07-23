"""Category-level cross-modal transfer probe. The dupe audit showed BP's held-out hits
cluster in frequent visual categories (planes, bathrooms, field animals) -- category-level
binding, not instance binding. This measures it directly: on HELD-OUT pairs, for each image
latent, retrieve the top-k caption latents and score the fraction sharing the query's
(keyword-derived) category, vs that category's base rate. Compare BP vs PC checkpoints at
matched training fit. Analysis-only, no training. Throwaway instrument."""
import argparse, numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from mechanism_probe import build_vocab, encode_caps, load_P, enc_img, enc_txt, l2n, CAPLEN
DATA = "/home/slsang29/coco_scale"

CATS = {
    "plane":   ["plane", "jet", "airliner", "airport", "flying", "helicopter"],
    "bathroom":["toilet", "bathroom", "sink", "shower", "urinal"],
    "train":   ["train", "railroad", "locomotive", "rail "],
    "bus":     ["bus "],
    "car_road":["car ", "truck", "motorcycle", "street", "road", "traffic", "highway"],
    "boat":    ["boat", "ship", "harbor", "marina", "water"],
    "zebra":   ["zebra"],
    "giraffe": ["giraffe"],
    "elephant":["elephant"],
    "cow_sheep":["cow", "sheep", "cattle", "horse"],
    "dog":     ["dog ", "dogs", "puppy"],
    "cat":     ["cat ", "cats", "kitten"],
    "bird":    ["bird", "duck", "goose"],
    "food":    ["plate", "food", "pizza", "sandwich", "salad", "meat", "veggies", "fruit", "cake", "bread"],
    "sports":  ["frisbee", "skate", "surf", "ski", "snowboard", "tennis", "baseball", "soccer", "kite"],
    "person":  ["man ", "woman", "people", "person", "boy ", "girl "],
    "kitchen": ["kitchen", "oven", "refrigerator", "microwave"],
    "clock":   ["clock"],
    "sign":    ["sign", "hydrant"],
}


def cat_of(caption):
    c = caption.lower()
    for name, kws in CATS.items():
        for kw in kws:
            if kw in c:
                return name
    return None


def probe(path, imgs, toks, caps, ev, label, topk=10):
    P = load_P(path)
    Z_i, Z_t = [], []
    for s in range(0, len(ev), 200):
        bi = ev[s:s + 200]
        Z_i.append(tf.concat(enc_img(P, tf.constant(imgs[bi].astype("float32"))), 1).numpy())
        Z_t.append(tf.concat(enc_txt(P, tf.constant(toks[bi])), 1).numpy())
    zi = l2n(np.concatenate(Z_i, 0)); zt = l2n(np.concatenate(Z_t, 0))
    cats = np.array([cat_of(caps[i]) or "" for i in ev])
    valid = cats != ""
    S = zi @ zt.T
    print(f"\n===== {label} ({path.split('/')[-2]}/{path.split('/')[-1]}) | eval={len(ev)}, categorized={int(valid.sum())} =====", flush=True)
    # per-query top-k category precision (image query -> caption pool), vs category base rate
    order = np.argsort(-S, axis=1)[:, :topk]
    prec, base = [], []
    for q in np.where(valid)[0]:
        cq = cats[q]
        prec.append(float(np.mean(cats[order[q]] == cq)))
        base.append(float(np.mean(cats[valid] == cq)))
    prec, base = np.array(prec), np.array(base)
    print(f"  i->t category precision@{topk}: {prec.mean():.4f}  (base rate {base.mean():.4f}, lift {prec.mean()/max(base.mean(),1e-9):.2f}x)", flush=True)
    # per-category breakdown (categories with >=20 eval members)
    for name in CATS:
        m = valid & (cats == name)
        if m.sum() < 20: continue
        qidx = np.where(m)[0]
        p = float(np.mean([np.mean(cats[order[q]] == name) for q in qidx]))
        b = float(np.mean(cats[valid] == name))
        print(f"    {name:10s} n={int(m.sum()):4d}  prec@{topk}={p:.3f}  base={b:.3f}  lift={p/max(b,1e-9):.1f}x", flush=True)
    return prec.mean(), base.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True); ap.add_argument("--pc", required=True)
    ap.add_argument("--ntrain", type=int, default=20000); ap.add_argument("--neval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--coco", default="train2017")
    a = ap.parse_args()
    imgs = np.load(f"{DATA}/imgs_sc_{a.coco}.npy", mmap_mode="r")
    caps = open(f"{DATA}/caps_sc_{a.coco}.txt").read().split("\n")[:len(imgs)]
    N = len(imgs); perm = np.random.RandomState(a.seed + 1).permutation(N)
    tr = perm[:a.ntrain]; ev = perm[a.ntrain:a.ntrain + a.neval]
    chars, c2i = build_vocab([caps[i] for i in tr]); toks = encode_caps(caps, c2i, CAPLEN)
    probe(a.bp, imgs, toks, caps, ev, "BP (E1L, 20k)")
    probe(a.pc, imgs, toks, caps, ev, "PC (jointw=1.0 lr5e-3, 20k)")
    print("CATEGORY_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
