"""Seed-1 forensics: WHY does the PC 8k matched-epochs seed-1 arm A show train lat_retr 0.008
(~12/1500, ~11 sigma in-sample) while held-out stays at 1/2000?

Recomputes the train latent retrieval from the checkpoint, identifies WHICH pairs hit, and profiles
them against three explanations: near-duplicate images (pixel MSE to nearest train neighbors),
near-duplicate captions (difflib ratio to nearest train caption), and latent-norm outliers
(pre-normalization concat norms vs population percentiles). Distinguishes memorization-without-transfer
(hits look ordinary, model simply memorized a few pairs) from a data artifact (hits are dups/outliers).

RUN ON THE CLUSTER. ENV: FOREN_CKPT (default ~/runs/8k_150ep_s1/cs_A_seed1.npz), RUNS1_DATA, seed fixed
at 1, N 8000/2000, READTRAIN 1500 (the same subsample the driver scored).
OUT: seed1_forensics.json + printed two-sentence conclusion.
"""
import os, json, difflib, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__)); HOME = os.path.expanduser("~")
DATA = os.environ.get("RUNS1_DATA", os.path.join(HOME, "coco_scale"))
COCO = os.environ.get("RUNS1_COCO", "train2017")
CKPT = os.environ.get("FOREN_CKPT", os.path.join(HOME, "runs", "8k_150ep_s1", "cs_A_seed1.npz"))
SEED, N_TRAIN, N_EVAL, READTRAIN, RES, CAPLEN = 1, 8000, 2000, 1500, 64, 64
HEADS, NBLK, NS, READB = 4, 4, 4, 128

def gelu(z): return tf.nn.gelu(z)
def build_vocab(cs):
    chs = sorted(set("".join(cs)) | {"\0"}); return chs, {c:i for i,c in enumerate(chs)}
def encode_caps(cs, c2i, caplen):
    nul = c2i["\0"]; tk = np.full((len(cs), caplen), nul, "int32")
    for n,cp in enumerate(cs):
        for t in range(caplen):
            if t < len(cp): tk[n,t] = c2i.get(cp[t], nul)
    return tk
def load_cache():
    im = np.load(os.path.join(DATA, f"imgs_sc_{COCO}.npy"))
    cp = open(os.path.join(DATA, f"caps_sc_{COCO}.txt")).read().split("\n")[:len(im)]
    print(f"[data] cache {im.shape}", flush=True); return im, cp
def make_encoders(P):
    DM = P["emb"].shape[1]; HEAD = DM // HEADS
    def enc_img(x):
        h=gelu(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        h=gelu(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        f0=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        f1=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]); h=tf.nn.max_pool2d(h,2,2,"SAME")
        f2=tf.reshape(h,[tf.shape(x)[0],-1]); f3=gelu(f2@P["wbn"]+P["bbn"])
        return [gelu(f0@P["Wi0"]+P["bi0"]),gelu(f1@P["Wi1"]+P["bi1"]),gelu(f2@P["Wi2"]+P["bi2"]),gelu(f3@P["Wi3"]+P["bi3"])]
    def enc_txt(tk):
        B=tf.shape(tk)[0]; x=tf.gather(P["emb"],tk)+P["pos"][None]; tt=[]
        for b in range(NBLK):
            q,k_,v=x@P[f"Wq{b}"],x@P[f"Wk{b}"],x@P[f"Wv{b}"]
            sp=lambda t: tf.transpose(tf.reshape(t,[B,CAPLEN,HEADS,HEAD]),[0,2,1,3])
            a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
            ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,CAPLEN,DM])
            x=x+ctx@P[f"Wo{b}"]; x=x+(gelu(x@P[f"f1_{b}"]+P[f"fb1_{b}"])@P[f"f2_{b}"]+P[f"fb2_{b}"])
            tt.append(gelu(tf.reduce_mean(x,1)@P[f"Wt{b}"]+P[f"bt{b}"]))
        return tt
    return enc_img, enc_txt
def forward_scales(enc_img, enc_txt, im, tk, idx):
    ZI = [[] for _ in range(NS)]; ZT = [[] for _ in range(NS)]
    for st in range(0, len(idx), READB):
        bi = idx[st:st+READB]
        it = enc_img(tf.constant(im[bi])); tt = enc_txt(tf.constant(tk[bi]))
        for k in range(NS): ZI[k].append(it[k].numpy()); ZT[k].append(tt[k].numpy())
    return [np.concatenate(z,0) for z in ZI], [np.concatenate(z,0) for z in ZT]
l2n = lambda Z: Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

imgs, caps = load_cache(); N_HAVE = len(imgs)
perm = np.random.RandomState(SEED+1).permutation(N_HAVE)
tr_idx = perm[:N_TRAIN]; ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]
chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)
toks = encode_caps(caps, c2i, CAPLEN)
npz = np.load(CKPT); P = {k: tf.constant(npz[k]) for k in npz.files}
assert P["emb"].shape[0] == V, f"vocab mismatch: ckpt {P['emb'].shape[0]} vs split {V}"
enc_img, enc_txt = make_encoders(P)
tr_sub = tr_idx[np.random.RandomState(SEED+3).choice(N_TRAIN, READTRAIN, replace=False)]   # driver's scored subsample

ZI_s, ZT_s = forward_scales(enc_img, enc_txt, imgs, toks, tr_sub)
ZI_raw = np.concatenate(ZI_s, 1); ZT_raw = np.concatenate(ZT_s, 1)
ZI, ZT = l2n(ZI_raw), l2n(ZT_raw)
pred = np.argmax(ZT @ ZI.T, 1)
hit_pos = np.where(pred == np.arange(len(tr_sub)))[0]
print(f"recomputed train lat_retr = {len(hit_pos)}/{len(tr_sub)} = {len(hit_pos)/len(tr_sub):.4f} "
      f"(driver reported 0.0080 = 12/1500)", flush=True)

# population profiles
img_flat = imgs[tr_sub].reshape(len(tr_sub), -1)
norms_i, norms_t = np.linalg.norm(ZI_raw, axis=1), np.linalg.norm(ZT_raw, axis=1)
def pct(v, pop): return float(np.mean(pop <= v) * 100)

profiles = []
for j in hit_pos:
    gi = int(tr_sub[j])
    d = ((img_flat - img_flat[j])**2).mean(1); d[j] = np.inf
    nn = int(np.argmin(d)); nn_mse = float(d[nn])
    cap_j = caps[gi]
    ratios = [(difflib.SequenceMatcher(None, cap_j, caps[int(tr_sub[o])]).ratio(), o)
              for o in range(len(tr_sub)) if o != j]
    best_ratio, bo = max(ratios)
    profiles.append(dict(
        subset_pos=int(j), global_idx=gi, caption=cap_j[:80],
        nn_img_mse=nn_mse,
        nn_caption_ratio=float(best_ratio), nn_caption=caps[int(tr_sub[int(bo)])][:80],
        img_norm_pct=pct(norms_i[j], norms_i), txt_norm_pct=pct(norms_t[j], norms_t),
        margin=float((ZT[j] @ ZI[j]) - np.partition(ZT[j] @ ZI.T, -2)[-2])))
    print(f"  HIT pair global={gi} | nn-img MSE={nn_mse:.5f} | nn-cap ratio={best_ratio:.2f} "
          f"| norm pct img/txt={pct(norms_i[j], norms_i):.0f}/{pct(norms_t[j], norms_t):.0f} "
          f"| margin={profiles[-1]['margin']:.4f} | '{cap_j[:60]}'", flush=True)

# population baselines for the profile columns
rng = np.random.RandomState(0); samp = rng.choice(len(tr_sub), 200, replace=False)
pop_nn_mse = []
for o in samp:
    d = ((img_flat - img_flat[o])**2).mean(1); d[o] = np.inf; pop_nn_mse.append(float(d.min()))
pop_cap_ratio = []
for o in samp[:60]:
    rr = max(difflib.SequenceMatcher(None, caps[int(tr_sub[o])], caps[int(tr_sub[q])]).ratio()
             for q in rng.choice(len(tr_sub), 300, replace=False) if q != o)
    pop_cap_ratio.append(rr)
base = dict(pop_nn_img_mse_median=float(np.median(pop_nn_mse)), pop_nn_img_mse_p10=float(np.percentile(pop_nn_mse, 10)),
            pop_nn_cap_ratio_median=float(np.median(pop_cap_ratio)), pop_nn_cap_ratio_p90=float(np.percentile(pop_cap_ratio, 90)))

dupish = [p for p in profiles if p["nn_img_mse"] < base["pop_nn_img_mse_p10"] or p["nn_caption_ratio"] > base["pop_nn_cap_ratio_p90"]]
outliers = [p for p in profiles if p["img_norm_pct"] > 97 or p["img_norm_pct"] < 3 or p["txt_norm_pct"] > 97 or p["txt_norm_pct"] < 3]
if len(dupish) >= max(1, len(profiles)//2):
    concl = (f"DATA-ARTIFACT PROFILE: {len(dupish)}/{len(profiles)} hit pairs are near-duplicates "
             f"(image MSE below the population p10 or caption similarity above the p90); the weak in-sample "
             f"signal rides on duplicated content, not learned coupling.")
elif len(outliers) >= max(1, len(profiles)//2):
    concl = (f"OUTLIER PROFILE: {len(outliers)}/{len(profiles)} hit pairs are latent-norm outliers; the "
             f"signal comes from a few extreme-norm examples dominating the cosine geometry.")
else:
    concl = (f"MEMORIZATION-WITHOUT-TRANSFER PROFILE: the {len(profiles)} hit pairs look ordinary "
             f"(images not near-duplicates vs population baselines, captions not near-duplicates, norms in "
             f"the normal range); the model simply memorized a handful of arbitrary training pairs, and that "
             f"memorization does not transfer (held-out stays 1/2000).")
print("\nCONCLUSION: " + concl, flush=True)

out = dict(ckpt=CKPT, seed=SEED, recomputed_train_hits=len(hit_pos), readtrain=len(tr_sub),
           heldout_reference="res_8k_150ep_s1.json arm_A heldout lat_retr 0.0005 (1/2000)",
           population_baselines=base, hits=profiles, conclusion=concl)
jp = os.path.join(HERE, "seed1_forensics.json")
with open(jp + ".tmp", "w") as fh: json.dump(out, fh, indent=2)
os.replace(jp + ".tmp", jp)
print("saved: seed1_forensics.json", flush=True)
