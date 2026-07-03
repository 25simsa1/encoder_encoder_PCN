"""E4 -- LATENT-GEOMETRY BATTERY: turn "mean-collapse alignment" into a demonstrated mechanism.

WHAT. For each trained system's checkpoint (PC arms, BPonF, E1, E1L, free-latent BPonF), forward the
capped train set (1500, same subsample law as the drivers) and the FULL held-out set through both
encoders, dump the 8 per-scale latents plus the per-modality concats, and compute:
  - effective rank (participation ratio of centered singular values) per modality, per scale and concat
  - matched-pair vs derangement cosine gap with a permutation sigma (200 derangements)
  - Wang-Isola alignment (mean matched cosine and E||zi-zt||^2) and uniformity (log mean exp(-2||x-y||^2))
    per modality, held-out
  - a retrieval ladder fitted on TRAIN pairs only, evaluated held-out: raw cosine / per-modality
    mean-centered / centered + orthogonal Procrustes (factored, never materializes DxD) / dual-form ridge
    text->image; hits at each rung vs the pre-registered bar >3/N_eval
  - per-scale whitened rung (each scale centered + L2-normed before concat) to test scale-drowning
  - bag-of-chars ridge probe from text latents (held-out R^2 + top-char accuracy vs a shuffled-pair null)

ORDERED INTERPRETATION (printed per system):
  mean-direction collapse (erank ~1, centering rescues nothing) -> misrotation (Procrustes rescues) ->
  scale-drowned (centering / per-scale whitening rescues) -> features-without-coupling (probe works,
  no ladder rung rescues).
PREDICTION under adjudication: PC and BPonF cluster together (low erank, near-zero matched-derangement
gap, no ladder rescue) while E1 and E1L differ on every axis.

RUN ON THE CLUSTER (ckpts + COCO cache live there). CPU is fine; a GPU makes the forwards faster.
ENV: RUNS1_DATA (~/coco_scale), GEOM_SPECS (semicolon list of name=path:seed entries; a default spec
covers the standard locations and skips missing files loudly), RUNS1_COCO(train2017),
RUNS1_NTRAIN(8000) RUNS1_NEVAL(2000) RUNS1_READTRAIN(1500).
OUT: analysis_latent_geometry_results.json + analysis_latent_geometry.md (compact table).
"""
import os, sys, json, math, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE   = os.path.dirname(os.path.abspath(__file__))
HOME   = os.path.expanduser("~")
DATA   = os.environ.get("RUNS1_DATA", os.path.join(HOME, "coco_scale"))
COCO   = os.environ.get("RUNS1_COCO", "train2017")
N_TRAIN= int(os.environ.get("RUNS1_NTRAIN", 8000))
N_EVAL = int(os.environ.get("RUNS1_NEVAL", 2000))
READTRAIN = int(os.environ.get("RUNS1_READTRAIN", 1500))
RES, CAPLEN, WMUL = 64, 64, 1.5
HEADS, NBLK, NS = 4, 4, 4
READB = 128
BAR = 3  # pre-registered: hits > 3/N_eval

DEFAULT_SPECS = ";".join([
    f"PC_armA=%s/runs/8k_150ep/cs_A_seed0.npz:0" % HOME,
    f"PC_armB=%s/runs/8k_150ep/cs_B_seed0.npz:0" % HOME,
    f"BPonF=%s/runs/geom_ckpts/bpf_seed0.npz:0" % HOME,
    f"E1_adam=%s/runs/geom_ckpts/e1_seed0.npz:0" % HOME,
    f"E1L_lars=%s/runs/geom_ckpts/e1l_seed0.npz:0" % HOME,
    f"BPonF_free=%s/runs/bpff_8k/bpf_free_seed0.npz:0" % HOME,
])
SPECS = []
for ent in os.environ.get("GEOM_SPECS", DEFAULT_SPECS).split(";"):
    name, rest = ent.split("="); path, seed = rest.rsplit(":", 1)
    SPECS.append((name, path, int(seed)))

def gelu(z): return tf.nn.gelu(z)

def build_vocab(caps):
    chars = sorted(set("".join(caps)) | {"\0"}); return chars, {c:i for i,c in enumerate(chars)}
def encode_caps(caps, c2i, caplen):
    nul = c2i["\0"]; toks = np.full((len(caps), caplen), nul, "int32")
    for n,cp in enumerate(caps):
        for t in range(caplen):
            if t < len(cp): toks[n,t] = c2i.get(cp[t], nul)
    return toks

def load_cache():
    f_img, f_cap = os.path.join(DATA, f"imgs_sc_{COCO}.npy"), os.path.join(DATA, f"caps_sc_{COCO}.txt")
    imgs = np.load(f_img); caps = open(f_cap).read().split("\n")[:len(imgs)]
    print(f"[data] cache {imgs.shape}", flush=True); return imgs, caps

# ---------------- encoders rebuilt from a checkpoint dict (identical across every driver) ----------------
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

def forward_scales(enc_img, enc_txt, imgs, toks, idx):
    ZI = [[] for _ in range(NS)]; ZT = [[] for _ in range(NS)]
    for st in range(0, len(idx), READB):
        bi = idx[st:st+READB]
        it = enc_img(tf.constant(imgs[bi])); tt = enc_txt(tf.constant(toks[bi]))
        for k in range(NS): ZI[k].append(it[k].numpy()); ZT[k].append(tt[k].numpy())
    return [np.concatenate(z,0) for z in ZI], [np.concatenate(z,0) for z in ZT]

l2n = lambda Z: Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

def erank(Z, center=True):
    # centered = dimensionality of variation AROUND the mean; uncentered = dimensionality of the cloud
    # itself (a ray-collapsed cloud has uncentered erank ~1 but can be full rank after centering).
    Zc = Z - Z.mean(0, keepdims=True) if center else Z
    s = np.linalg.svd(Zc, compute_uv=False)
    s2 = s**2
    return float((s2.sum()**2) / ((s2**2).sum() + 1e-30))

def mu_norm2(Z):
    # for L2-normalized rows, ||mean||^2 in [0,1] is the concentration-on-one-direction measure
    m = Z.mean(0); return float(m @ m)

def hits_of(ZT, ZI):
    M = len(ZT); return int(np.sum(np.argmax(ZT @ ZI.T, 1) == np.arange(M)))

def derangement_gap(ZI, ZT, n_perm=200, rng=None):
    rng = rng or np.random.RandomState(0)
    M = len(ZI); matched = float(np.mean(np.sum(ZI*ZT, 1)))
    mis = []
    for _ in range(n_perm):
        pi = rng.permutation(M)
        fx = np.where(pi == np.arange(M))[0]
        for j in fx: pi[j], pi[(j+1) % M] = pi[(j+1) % M], pi[j]     # derange fixed points
        mis.append(float(np.mean(np.sum(ZI*ZT[pi], 1))))
    mis = np.array(mis)
    sd = float(mis.std() + 1e-12)
    return dict(matched_cos=matched, mismatched_cos=float(mis.mean()),
                gap=matched - float(mis.mean()), gap_sigma=(matched - float(mis.mean())) / sd)

def uniformity(Z, t=2.0, cap=2000, rng=None):
    rng = rng or np.random.RandomState(1)
    if len(Z) > cap: Z = Z[rng.choice(len(Z), cap, replace=False)]
    d2 = np.maximum(2.0 - 2.0*(Z @ Z.T), 0.0)
    iu = np.triu_indices(len(Z), 1)
    return float(np.log(np.mean(np.exp(-t * d2[iu])) + 1e-30))

def procrustes_retrieval(ZT_tr, ZI_tr, ZT_ev, ZI_ev):
    # centered inputs. R = argmin ||ZT R - ZI||_F over orthogonal R, computed factored:
    # M = ZT_tr^T ZI_tr = Q1 (R1 R2^T) Q2^T ; SVD(R1 R2^T) = U' S V'^T ; R = (Q1 U')(Q2 V')^T.
    Q1, R1 = np.linalg.qr(ZT_tr.T)                                    # D x n, n x n
    Q2, R2 = np.linalg.qr(ZI_tr.T)
    U_, _, Vt_ = np.linalg.svd(R1 @ R2.T)
    A = Q1 @ U_                                                       # D x n
    B = Q2 @ Vt_.T                                                    # D x n
    # (ZT_ev @ R) @ ZI_ev^T = (ZT_ev @ A) @ (ZI_ev @ B)^T  -- never materialize R (D x D)
    S = (ZT_ev @ A) @ (ZI_ev @ B).T
    return int(np.sum(np.argmax(S, 1) == np.arange(len(ZT_ev))))

def ridge_retrieval(ZT_tr, ZI_tr, ZT_ev, ZI_ev, lams=(1e-2, 1e-1, 1.0)):
    # dual-form ridge text->image: W = ZT^T (K + lam I)^-1 ZI, K = ZT ZT^T (n x n). Pick lam on TRAIN.
    K = ZT_tr @ ZT_tr.T; n = len(K)
    best = None
    for lam in lams:
        alpha = np.linalg.solve(K + lam*np.eye(n), ZI_tr)             # n x D
        pred_tr = K @ alpha                                           # = ZT_tr W
        tr_hits = hits_of(l2n(pred_tr), l2n(ZI_tr))
        if best is None or tr_hits > best[0]: best = (tr_hits, lam, alpha)
    _, lam, alpha = best
    pred_ev = (ZT_ev @ ZT_tr.T) @ alpha
    return int(np.sum(np.argmax(l2n(pred_ev) @ l2n(ZI_ev).T, 1) == np.arange(len(ZT_ev)))), lam

def bag_probe(ZT_tr, toks_tr, ZT_ev, toks_ev, V, nul, lam=1.0, rng=None):
    rng = rng or np.random.RandomState(2)
    def bag(tk):
        B = np.zeros((len(tk), V), "float64")
        for i, row in enumerate(tk):
            for c in row:
                if c != nul: B[i, c] += 1
        return B
    Ytr, Yev = bag(toks_tr), bag(toks_ev)
    K = ZT_tr @ ZT_tr.T; n = len(K)
    alpha = np.linalg.solve(K + lam*np.eye(n), Ytr - Ytr.mean(0))
    pred = (ZT_ev @ ZT_tr.T) @ alpha + Ytr.mean(0)
    ss_res = float(((Yev - pred)**2).sum()); ss_tot = float(((Yev - Yev.mean(0))**2).sum())
    r2 = 1 - ss_res/(ss_tot + 1e-30)
    top_acc = float(np.mean(pred.argmax(1) == Yev.argmax(1)))
    # shuffled-pair null (refit with permuted captions)
    r2n = []
    for _ in range(3):
        piN = rng.permutation(n)
        alphaN = np.linalg.solve(K + lam*np.eye(n), Ytr[piN] - Ytr[piN].mean(0))
        predN = (ZT_ev @ ZT_tr.T) @ alphaN + Ytr[piN].mean(0)
        r2n.append(1 - float(((Yev - predN)**2).sum())/(ss_tot + 1e-30))
    return dict(r2=r2, r2_null=float(np.mean(r2n)), top_char_acc=top_acc,
                null_top_base=float(np.mean(Yev.argmax(1) == np.bincount(Ytr.argmax(1)).argmax())))

# ---------------- main ----------------
imgs, caps = load_cache(); N_HAVE = len(imgs)
results = {}
for name, path, seed in SPECS:
    if not os.path.exists(path):
        print(f"!! SKIP {name}: checkpoint missing at {path}", flush=True); continue
    t0 = time.time()
    perm = np.random.RandomState(seed+1).permutation(N_HAVE)
    tr_idx = perm[:N_TRAIN]; ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]
    chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars); nul = c2i["\0"]
    toks = encode_caps(caps, c2i, CAPLEN)
    npz = np.load(path)
    P = {k: tf.constant(npz[k]) for k in npz.files}
    assert P["emb"].shape[0] == V, f"{name}: vocab mismatch (ckpt {P['emb'].shape[0]} vs split {V})"
    enc_img, enc_txt = make_encoders(P)
    tr_sub = tr_idx if N_TRAIN <= READTRAIN else tr_idx[np.random.RandomState(seed+3).choice(N_TRAIN, READTRAIN, replace=False)]
    ZI_tr_s, ZT_tr_s = forward_scales(enc_img, enc_txt, imgs, toks, tr_sub)
    ZI_ev_s, ZT_ev_s = forward_scales(enc_img, enc_txt, imgs, toks, ev_idx)
    ZI_tr, ZT_tr = np.concatenate(ZI_tr_s, 1), np.concatenate(ZT_tr_s, 1)
    ZI_ev, ZT_ev = np.concatenate(ZI_ev_s, 1), np.concatenate(ZT_ev_s, 1)
    ZI_trn, ZT_trn, ZI_evn, ZT_evn = map(l2n, (ZI_tr, ZT_tr, ZI_ev, ZT_ev))

    r = dict(seed=seed, ckpt=path, n_train_sub=len(tr_sub), n_eval=len(ev_idx), V=V)
    r["erank"] = dict(
        img_concat=erank(ZI_evn), txt_concat=erank(ZT_evn),
        img_concat_unc=erank(ZI_evn, center=False), txt_concat_unc=erank(ZT_evn, center=False),
        img_mu_norm2=mu_norm2(ZI_evn), txt_mu_norm2=mu_norm2(ZT_evn),
        img_scales=[erank(l2n(z)) for z in ZI_ev_s], txt_scales=[erank(l2n(z)) for z in ZT_ev_s])
    r["gap"] = derangement_gap(ZI_evn, ZT_evn)
    r["wang_isola"] = dict(align_cos=r["gap"]["matched_cos"],
                           l_align=float(np.mean(np.sum((ZI_evn-ZT_evn)**2, 1))),
                           unif_img=uniformity(ZI_evn), unif_txt=uniformity(ZT_evn))
    # retrieval ladder (fit on train only)
    mi, mt = ZI_tr.mean(0, keepdims=True), ZT_tr.mean(0, keepdims=True)
    ZI_tr_c, ZT_tr_c = l2n(ZI_tr-mi), l2n(ZT_tr-mt)
    ZI_ev_c, ZT_ev_c = l2n(ZI_ev-mi), l2n(ZT_ev-mt)
    # per-scale whitening rung: center+norm each scale with TRAIN stats, then concat
    def scale_white(Zs, means): return l2n(np.concatenate([l2n(z-m) for z, m in zip(Zs, means)], 1))
    mis = [z.mean(0, keepdims=True) for z in ZI_tr_s]; mts = [z.mean(0, keepdims=True) for z in ZT_tr_s]
    ridge_hits, ridge_lam = ridge_retrieval(ZT_tr_c, ZI_tr_c, ZT_ev_c, ZI_ev_c)
    r["ladder"] = dict(
        raw=hits_of(ZT_evn, ZI_evn),
        centered=hits_of(ZT_ev_c, ZI_ev_c),
        centered_procrustes=procrustes_retrieval(ZT_tr_c, ZI_tr_c, ZT_ev_c, ZI_ev_c),
        scale_whitened=hits_of(scale_white(ZT_ev_s, mts), scale_white(ZI_ev_s, mis)),
        ridge=ridge_hits, ridge_lam=ridge_lam, bar=BAR, n_eval=len(ev_idx))
    r["bag_probe"] = bag_probe(ZT_tr_c, toks[tr_sub], ZT_ev_c, toks[ev_idx], V, nul)
    # ordered interpretation
    lad = r["ladder"]; rescued = {k: lad[k] > BAR for k in ("raw","centered","centered_procrustes","scale_whitened","ridge")}
    probe_ok = r["bag_probe"]["r2"] > max(3*abs(r["bag_probe"]["r2_null"]), 0.05)
    collapsed = (r["erank"]["txt_concat_unc"] < 3 or r["erank"]["img_concat_unc"] < 3
                 or max(r["erank"]["txt_mu_norm2"], r["erank"]["img_mu_norm2"]) > 0.8)
    if collapsed and not any(rescued.values()):
        verdict = "mean-direction collapse (uncentered erank ~1 / mean carries the energy, nothing rescues)"
    elif rescued["centered_procrustes"] and not rescued["raw"]:
        verdict = "misrotation (Procrustes rescues held-out retrieval)"
    elif (rescued["centered"] or rescued["scale_whitened"]) and not rescued["raw"]:
        verdict = "scale-drowned or mean-offset (centering/whitening rescues)"
    elif probe_ok and not any(rescued.values()):
        verdict = "features-without-coupling (caption info present, no linear map to image latents)"
    elif rescued["raw"]:
        verdict = "coupled (raw retrieval above bar)"
    else:
        verdict = "no caption information in text latents (probe fails, nothing rescues)"
    r["verdict"] = verdict
    results[name] = r
    print(f"[{name}] ({time.time()-t0:.0f}s) erank(c) img/txt={r['erank']['img_concat']:.1f}/{r['erank']['txt_concat']:.1f} "
          f"erank(u)={r['erank']['img_concat_unc']:.1f}/{r['erank']['txt_concat_unc']:.1f} "
          f"mu2={r['erank']['img_mu_norm2']:.2f}/{r['erank']['txt_mu_norm2']:.2f} "
          f"gap={r['gap']['gap']:.4f} ({r['gap']['gap_sigma']:.1f} sig) align={r['wang_isola']['align_cos']:.3f} "
          f"unif(i/t)={r['wang_isola']['unif_img']:.2f}/{r['wang_isola']['unif_txt']:.2f} | "
          f"ladder raw/cent/proc/white/ridge = {lad['raw']}/{lad['centered']}/{lad['centered_procrustes']}/"
          f"{lad['scale_whitened']}/{lad['ridge']} (bar >{BAR}/{lad['n_eval']}) | "
          f"probe R2={r['bag_probe']['r2']:.3f} (null {r['bag_probe']['r2_null']:.3f}) | VERDICT: {verdict}", flush=True)

# ---------------- prediction adjudication + outputs ----------------
pred = None
have = lambda n: n in results
if have("PC_armA") and have("BPonF") and have("E1_adam"):
    def sig(n):
        r = results[n]
        best_rescue = max(r["ladder"][k] for k in ("centered","centered_procrustes","scale_whitened","ridge"))
        return dict(mu2=max(r["erank"]["img_mu_norm2"], r["erank"]["txt_mu_norm2"]),
                    gap_sigma=r["gap"]["gap_sigma"], raw=r["ladder"]["raw"], best_rescue=best_rescue)
    pc, bpf, e1 = sig("PC_armA"), sig("BPonF"), sig("E1_adam")
    fail_geom = lambda s: s["raw"] <= BAR and s["best_rescue"] <= BAR
    same_cluster = fail_geom(pc) and fail_geom(bpf) and (abs(pc["mu2"]-bpf["mu2"]) < 0.3)
    e1_differs = (e1["raw"] > BAR) or (e1["best_rescue"] > BAR) or (e1["gap_sigma"] > 10*max(pc["gap_sigma"], 1e-9))
    pred = dict(pc=pc, bponf=bpf, e1=e1, pc_bponf_cluster=bool(same_cluster), e1_differs=bool(e1_differs),
                adjudication=("PREDICTION HOLDS: PC and BPonF share the failure geometry while the InfoNCE systems differ"
                              if same_cluster and e1_differs else
                              "PREDICTION PARTIAL OR FAILED: inspect per-system rows"))
    print(f"\nPREDICTION: {pred['adjudication']}", flush=True)

out = dict(config=dict(n_train=N_TRAIN, n_eval=N_EVAL, readtrain=READTRAIN, coco=COCO, bar=BAR),
           systems=results, prediction=pred)
jp = os.path.join(HERE, "analysis_latent_geometry_results.json")
with open(jp + ".tmp", "w") as fh: json.dump(out, fh, indent=2)
os.replace(jp + ".tmp", jp)

rows = ["| system | erank img/txt (mu2 i/t) | matched-derangement gap (sigma) | align_cos | unif img/txt | raw | centered | Procrustes | whitened | ridge | probe R2 (null) | verdict |",
        "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
for n, r in results.items():
    lad = r["ladder"]; g = r["gap"]; wi = r["wang_isola"]; bp = r["bag_probe"]
    rows.append(f"| {n} | {r['erank']['img_concat']:.1f}/{r['erank']['txt_concat']:.1f} "
                f"({r['erank']['img_mu_norm2']:.2f}/{r['erank']['txt_mu_norm2']:.2f}) | {g['gap']:.4f} ({g['gap_sigma']:.1f}) | "
                f"{wi['align_cos']:.3f} | {wi['unif_img']:.2f}/{wi['unif_txt']:.2f} | {lad['raw']} | {lad['centered']} | "
                f"{lad['centered_procrustes']} | {lad['scale_whitened']} | {lad['ridge']} | {bp['r2']:.3f} ({bp['r2_null']:.3f}) | {r['verdict']} |")
mp = os.path.join(HERE, "analysis_latent_geometry.md")
open(mp, "w").write(f"# E4 latent-geometry battery (8k, held-out N={N_EVAL}, ladder bar >{BAR} hits)\n\n" + "\n".join(rows) +
                    ("\n\n" + pred["adjudication"] if pred else "") + "\n")
print(f"\nsaved: analysis_latent_geometry_results.json + analysis_latent_geometry.md ({len(results)} systems)", flush=True)
