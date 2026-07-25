"""PC-MAX driver -- scale BIDIRECTIONAL PC with the two published scaling methods, fork of
run_coupling_capacity.py (the original is untouched). New experimental arm for the paper: give local
PC its best shot per muPC (Innocenti et al., NeurIPS 2025, Depth-muP parameterization) and HEP
(Mohammadi & Ororbia 2026, error highways), then re-ask the coupling question at 156M/20k against
the banked bar and the category-transfer probe. Spec: docs/superpowers/specs/
2026-07-24-pcmax-mupc-hep-design.md.

WHAT IS DIFFERENT FROM THE CAPACITY DRIVER (each one a documented deviation or an addition; the
frozen recipe constants are untouched and arms A/B reproduce it under PCMAX_PARITY=1):
  1. muP PARAMETERIZATION (arms Bmu and PCMAX). Weights are drawn N(0,1) with the SAME generator and
     draw order as the baseline; the forward applies explicit premultipliers MULT[k] (hidden
     1/sqrt(fan_in), decoders/proj 1/fan_in). RMSNorm with learnable gains at block inputs (image
     blocks 2-4, text pre-attention + pre-FFN); residual branches scaled RSCALE=1/sqrt(2*NBLK).
     Gains are new ckpt keys; checkpoints carry an __pcmax arch marker for the probes.
  2. BIDIRECTIONAL INTERIOR STATES (arm PCMAX). Free states at every block output (image Z1..Z4,
     text Z0..Z3) plus the 4 shared tap states. Reciprocal edges: bottom-up prediction = the block
     forward; top-down prediction = UNTIED weights (td_c* conv2d_transpose, td_t* position-wise
     dense; per the 2026-07-17 untied-td design rule). Bottom clamps: the image pixels and the
     discrete tokens (emb+pos fold into text block 0 so they train). Energy = bu errors +
     PCMAX_BIDIR_W * td errors + the family cross/gen terms with taps computed FROM the states.
  3. HEP ERROR HIGHWAYS (arm PCMAX). eps = -grad InfoNCE wrt the pre-norm concat latents (a small
     tape over the tap outputs only, recomputed at every inference step), split at the DIMS
     boundaries; text segment b -> text block b, image seg0->Z2 seg1->Z3 seg2->Z4 (the block feeding
     that tap) and seg3 (bottleneck) -> Z1 so every block has a highway. Fixed random V matrices map
     the segment to the state's channel/DM axis and broadcast over positions (the taps are
     mean/flatten-pooled, so the tap error is position-uniform; full flattened-state maps would be
     ~90 GB per text block at 7.7B). FA-fork discipline: dedicated Generator(seed+20011), fixed draw
     order, non-trainable, NOT in P, never saved. Tap states get no highway (the cross terms already
     deliver their error). Injection: state += STATE_LR * ALPHA * (seg @ V) each inference step.
  4. INFERENCE (arm PCMAX). Feedforward init (bu errors exactly zero at t=0), PCMAX_T_INFER steps of
     hand-rolled Adam-on-states (m,v reset every batch; PCMAX_STATE_OPT=gd is the plain-GD fallback
     for the strict monotone gate). The relaxation reduces with a SUM over the batch (family rule).
  5. FULLY LOCAL WEIGHT UPDATE (arm PCMAX). The relaxed states enter the weight tape as constants,
     so each energy term touches exactly one block's weights and one tape over the summed energy is
     automatically block-local -- NO cross-layer backprop exists anywhere in the PCMAX arm. The
     JOINTW*InfoNCE term is computed from taps of constant states, so its gradient reaches only the
     boundary-adjacent tap heads (Wi*, wbn, Wt* and their biases) -- the PC output layer. There is
     NO InfoNCE warmup phase for PCMAX (warmup is backprop; PCMAX is backprop-free by construction).
     Deviation from the family jointw mechanics: the InfoNCE term rides the weight objective on the
     SAME batch instead of a separate warmup_step batch (the separate step would be a full backprop).
     LARS trust ratio byte-copied. Locality granularity is the BLOCK: attention mixes positions, so
     the transformer term is local in depth, not in neurons -- pinned here once.
  6. PCMAX_PARITY=1 gate mode: baseline init law (stddev inside the draw), no gains/td/marker keys,
     RMSNorm off, RSCALE=1, premultipliers not applied. Arms A/B then reproduce
     run_coupling_capacity.py digit-for-digit (the FA transpose-gate standard); banner/verdict prose
     may differ, numeric trace lines must not.
  7. DIAGNOSTIC (PCMAX_DIAG=1): at the first joint step, per-inference-step per-state
     RMS(dZ)/RMS(Z) is printed twice on the same batch -- alpha=0 then alpha=PCMAX_ALPHA -- the HEP
     Fig-1 analog. Expected: alpha=0 leaves deep blocks near-silent early; alpha>0 moves every block
     from step 1.
  8. PCMAX_FITSTOP>0: early-stop at matched fit (train lat_retr >= FITSTOP at an epoch boundary) --
     the budget lever; epochs beyond fit do not change the verdict. Off by default.
  9. Checkpoint/resume extended to the single-arm PCMAX run (it consumes only the epoch-shuffle RNG,
     like arm A; Adam-on-states resets per batch so nothing extra persists).

ENV: RUNS1_* and CAP_* exactly as run_coupling_capacity.py, RUNS1_ARMS gains "Bmu" and "PCMAX",
plus PCMAX_ALPHA(0.1) PCMAX_T_INFER(16) PCMAX_SIGMA_V(1e-3) PCMAX_STATE_LR(1e-2)
PCMAX_STATE_OPT(adam|gd) PCMAX_BIDIR_W(1.0) PCMAX_PARITY(0) PCMAX_DIAG(1) PCMAX_FITSTOP(0).
OUT: pcmax_capacity_w{WMUL}_seed{SEED}.json, checkpoints cap_{arm}_w{WMUL}_seed{SEED}.npz in
RUNS1_CKPT (68 baseline keys + gains/td + __pcmax marker for Bmu/PCMAX; parity arms = baseline
layout).
"""
import os, sys, time, json, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNS1_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE  = os.environ.get("RUNS1_SMOKE", "0") == "1"
SEED   = int(os.environ.get("RUNS1_SEED", 0))
RES    = int(os.environ.get("RUNS1_RES", 16 if SMOKE else 64))
CAPLEN = int(os.environ.get("RUNS1_CAPLEN", 16 if SMOKE else 64))
N_TRAIN= int(os.environ.get("RUNS1_NTRAIN", 12 if SMOKE else 8000))
N_EVAL = int(os.environ.get("RUNS1_NEVAL", 6 if SMOKE else 2000))
N_WANT = N_TRAIN + N_EVAL
PAIRS  = int(os.environ.get("RUNS1_PAIRS", N_WANT + 300))
WMUL   = float(os.environ.get("RUNS1_WMUL", 0.1 if SMOKE else 1.5))
LR     = float(os.environ.get("RUNS1_LR", 2e-2))
BATCHJ = int(os.environ.get("RUNS1_BATCHJ", 2 if SMOKE else 128))
EPOCHS = int(os.environ.get("RUNS1_EPOCHS", 4 if SMOKE else 150))
WARMUP = int(os.environ.get("RUNS1_WARMUP", 6 if SMOKE else 1500))
BATCH  = int(os.environ.get("RUNS1_BATCH", 4 if SMOKE else 64))
TEMP   = float(os.environ.get("RUNS1_TEMP", 0.07))
WARMLR = float(os.environ.get("RUNS1_WARMLR", 2e-3 if not SMOKE else LR))
RAMP   = int(os.environ.get("RUNS1_RAMP", 2 if SMOKE else 300))
JOINTW = float(os.environ.get("RUNS1_JOINTW", 0.0))
READB  = int(os.environ.get("RUNS1_READB", 3 if SMOKE else 128))
CKPT   = os.environ.get("RUNS1_CKPT", "/tmp/cap_ckpt" if SMOKE else "/root")
DATA   = os.environ.get("RUNS1_DATA", "/tmp/cap_data" if SMOKE else "/root/coco_scale")
COCO   = os.environ.get("RUNS1_COCO", "val2017" if SMOKE else "train2017")
READTRAIN = int(os.environ.get("RUNS1_READTRAIN", 4 if SMOKE else 1500))
ARMS   = [a.strip() for a in os.environ.get("RUNS1_ARMS", "PCMAX" if not SMOKE else "PCMAX").split(",") if a.strip()]
CKPT_EVERY = int(os.environ.get("CAP_CKPT_EVERY", 0))
RESUME = os.environ.get("CAP_RESUME", "auto")
AUTOBATCH = os.environ.get("CAP_AUTOBATCH", "1") == "1"
LOWHOST = os.environ.get("CAP_LOWHOST", "0") == "1"
# PCMAX knobs (see header)
PARITY   = os.environ.get("PCMAX_PARITY", "0") == "1"
# PCMAX_WOPT=adamw: the published weight optimizer (muPC uses Adam, HEP uses AdamW wd=1e-4).
# ADDED 2026-07-24 after the first GPU wave: every muP arm (Bmu included, no states/highways)
# diverged under LARS at 5e-3 with F healthy and move% compounding -- RMSNorm makes the forward
# invariant to pre-norm weight scale, so the trust ratio (step ~ ||w||) pushes along the loss-flat
# radial direction and norm inflates exponentially (the project's known ~ep13 mode, isolated to
# LARS x normalization). Decoupled weight decay is the standard cure. Default stays lars so the
# parity gate and the family comparison are untouched.
WOPT     = os.environ.get("PCMAX_WOPT", "lars")
WD       = float(os.environ.get("PCMAX_WD", 1e-4))           # adamw decoupled weight decay; RUNS1_LR is the lr for BOTH optimizers (ramp semantics unchanged)
# PCMAX_ANORM=1: the pre-registered RMS-normalized-alpha fallback (docs/runbooks/PCMAX.md watch-items).
# ADDED 2026-07-25 after 9605: raw injection is proportional to the InfoNCE latent gradient, which
# GROWS as the highway drives mean-collapse (discrimination worsens) -> positive feedback -> NaN at
# a config (alpha=9.7e4, ratio 0.1) that survived 15ep under a different GPU-noise draw (9579), so
# the raw-alpha stability window is a knife edge. With ANORM the eps segment is normalized to unit
# RMS per example before the highway matrix, so alpha sets the injection scale directly, immune to
# InfoNCE-scale drift. Never a silent change: off by default, banner-printed when on.
ANORM    = os.environ.get("PCMAX_ANORM", "0") == "1"
ALPHA    = float(os.environ.get("PCMAX_ALPHA", 0.1))
T_INFER  = int(os.environ.get("PCMAX_T_INFER", 16))
SIGMA_V  = float(os.environ.get("PCMAX_SIGMA_V", 1e-3))
STATE_LR = float(os.environ.get("PCMAX_STATE_LR", 1e-2))
STATE_OPT= os.environ.get("PCMAX_STATE_OPT", "adam")
BIDIR_W  = float(os.environ.get("PCMAX_BIDIR_W", 1.0))
DIAG     = os.environ.get("PCMAX_DIAG", "1") == "1"
FITSTOP  = float(os.environ.get("PCMAX_FITSTOP", 0.0))
MUP      = not PARITY                      # muP parameterization active for Bmu/PCMAX builds
os.makedirs(CKPT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"
assert all(a in ("A","B","A_long","Bmu","PCMAX") for a in ARMS), "RUNS1_ARMS must be a subset of A,B,A_long,Bmu,PCMAX"
assert STATE_OPT in ("adam","gd")
assert WOPT in ("lars","adamw")
if PARITY: assert all(a in ("A","B","A_long") for a in ARMS), "parity mode exists to reproduce the baseline arms"
if PARITY: assert WOPT=="lars", "parity mode is the frozen LARS recipe"
if CKPT_EVERY: assert ARMS in (["A"],["PCMAX"]), "checkpoint/resume is supported for single-arm A or PCMAX runs"
if LOWHOST: assert ARMS in (["A"],["PCMAX"]), "low-host mode assumes a single-arm run (reset is a no-op)"

# recipe constants (identical to run_coupling_scale.py / run_coupling_capacity.py)
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER  = int(os.environ.get("RUNS1_NINFER", 2 if SMOKE else 8))
GEN_INFER = 3 if SMOKE else 25
DIVERGE_W = 1e3
MOVE_MIN = 0.40
LOG_EVERY = 2 if SMOKE else 50
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]
RSCALE = 1.0 if PARITY else 1.0/math.sqrt(2.0*NBLK)

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)   # -> [B]

# ============================ DATA (byte-copied: one caption per image; split by image) ============================
SMOKE_CAPS = ["a dog runs across a grassy field","a red bus on a city street","two people sit on a wooden bench",
    "a plate of food on a table","a cat sleeping on a couch","a man riding a surfboard on a wave",
    "a clock tower against a blue sky","a bowl of fruit near a window"]

def build_vocab(caps):
    chars = sorted(set("".join(caps)) | {"\0"}); return chars, {c:i for i,c in enumerate(chars)}
def encode_caps(caps, c2i, caplen):
    nul = c2i["\0"]; toks = np.full((len(caps), caplen), nul, "int32")
    for n,cp in enumerate(caps):
        for t in range(caplen):
            if t < len(cp): toks[n,t] = c2i.get(cp[t], nul)
    return toks

def load_synthetic():
    rs = np.random.RandomState(SEED)
    caps = [c.lower() for c in (SMOKE_CAPS * 4)[:N_WANT]]
    imgs = np.zeros((len(caps), RES, RES, 3), "float32")
    for i in range(len(caps)):
        imgs[i] = rs.rand(RES,RES,3).astype("float32")*0.3
        c = rs.rand(3); y,x = rs.randint(0,RES,2); r = RES//4
        yy,xx = np.ogrid[:RES,:RES]; m = (yy-y)**2+(xx-x)**2 <= r*r
        imgs[i][m] = c
    return imgs, caps

def load_coco():
    f_img,f_cap = os.path.join(DATA,f"imgs_sc_{COCO}.npy"), os.path.join(DATA,f"caps_sc_{COCO}.txt")
    if os.path.exists(f_img) and os.path.exists(f_cap):
        imgs=np.load(f_img); caps=open(f_cap).read().split("\n")[:len(imgs)]
        print(f"[data] reused cache {imgs.shape}",flush=True); return imgs, caps
    import urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    ANN="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"; IMG="http://images.cocodataset.org/"+COCO+"/{}"
    imgdir=os.path.join(DATA,"img"); os.makedirs(imgdir,exist_ok=True); capj=os.path.join(DATA,"cap.json"); t0=time.time()
    if not os.path.exists(capj):
        z=os.path.join(DATA,"ann.zip")
        if not os.path.exists(z): print("[data] downloading COCO annotations (~241MB)...",flush=True); urllib.request.urlretrieve(ANN,z)
        with zipfile.ZipFile(z) as zf, zf.open(f"annotations/captions_{COCO}.json") as s, open(capj,"wb") as d: d.write(s.read())
    cap=json.load(open(capj)); id2cap={}
    for a in cap["annotations"]: id2cap.setdefault(a["image_id"], a["caption"])
    id2file={im["id"]:im["file_name"] for im in cap["images"]}
    ids=[i for i in id2cap if i in id2file][:PAIRS]
    def dl(iid):
        p=os.path.join(imgdir,id2file[iid])
        if not os.path.exists(p):
            try: urllib.request.urlretrieve(IMG.format(id2file[iid]),p)
            except Exception: pass
    with ThreadPoolExecutor(max_workers=48) as ex: list(ex.map(dl,ids))
    print(f"[data] downloads done ({time.time()-t0:.0f}s)",flush=True)
    imgs,caps=[],[]
    for iid in ids:
        p=os.path.join(imgdir,id2file[iid])
        if not os.path.exists(p): continue
        try:
            im=Image.open(p).convert("RGB").resize((RES,RES))
            imgs.append(np.asarray(im,"float32")/255.0); caps.append(id2cap[iid].strip().lower())
            if len(imgs)>=N_WANT: break
        except Exception: pass
    imgs=np.asarray(imgs,"float32")
    np.save(f_img,imgs); open(f_cap,"w").write("\n".join(caps))
    print(f"[data] COCO ready {imgs.shape} ({time.time()-t0:.0f}s)",flush=True)
    return imgs, caps

imgs_all, caps_all = (load_synthetic() if SMOKE else load_coco())
N_HAVE = len(imgs_all)
assert N_HAVE >= N_TRAIN + 1, f"only {N_HAVE} pairs, need >= {N_TRAIN}+1"
perm = np.random.RandomState(SEED+1).permutation(N_HAVE)
tr_idx = perm[:N_TRAIN]; ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]
imgs = imgs_all; caps = caps_all
NTR, NEV = len(tr_idx), len(ev_idx)
PIX = RES*RES*3; CH = 3
chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)
toks = encode_caps(caps, c2i, CAPLEN); toks_oh = tf.one_hot(toks, V).numpy().astype("float32")
train_mean_img = imgs[tr_idx].mean(0)

# ============================ MODEL ============================
def cfg(wmul):
    r=lambda x:max(4,int(round(x*wmul)))
    DM=r(B_DM); DM-=DM%HEADS
    return dict(DM=max(HEADS,DM),C1=r(B_C1),C2=r(B_C2),C3=r(B_C3),C4=r(B_C4),BN=r(B_BN),
                DIMS=[r(d) for d in B_DIMS],FFN=r(B_FFN),HEAD=max(1,(max(HEADS,DM))//HEADS))

def build(wmul, seed):
    """muP build: SAME generator and draw order as the baseline for the 68 baseline keys; under
    PCMAX_PARITY the stddev rides the draw (byte-identical baseline build, MULT all 1.0, no extra
    keys). Under muP the draws are N(0,1) and MULT carries the scale; gains (ones, no draws) and
    untied td weights (drawn AFTER all baseline keys) are appended."""
    c=cfg(wmul); DM,C1,C2,C3,C4,BN,DIMS,FFN=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"]
    s2=RES//4; s3=RES//8; s4=RES//16; f0d,f1d,f2d=s2*s2*C2, s3*s3*C3, s4*s4*C4
    c["f0d"],c["f1d"],c["f2d"]=f0d,f1d,f2d
    g=tf.random.Generator.from_seed(seed)
    MULT={}
    def W(shape,key=""):
        sd=DEC_SD if (key.startswith("proj") or key in ("W_DI","W_DT")) else 1.0/np.sqrt(np.prod(shape[:-1]))
        if PARITY:
            MULT[key]=1.0; return tf.Variable(g.normal(shape,stddev=sd))
        MULT[key]=(1.0/np.prod(shape[:-1])) if (key.startswith("proj") or key in ("W_DI","W_DT")) \
                  else 1.0/np.sqrt(np.prod(shape[:-1]))
        return tf.Variable(g.normal(shape,stddev=1.0))
    def Z(shape): return tf.Variable(tf.zeros(shape))
    P=dict(c1=W([3,3,CH,C1],"c1"),cb1=Z([C1]),c2=W([3,3,C1,C2],"c2"),cb2=Z([C2]),c3=W([3,3,C2,C3],"c3"),cb3=Z([C3]),
           c4=W([3,3,C3,C4],"c4"),cb4=Z([C4]),wbn=W([f2d,BN],"wbn"),bbn=Z([BN]),
           Wi0=W([f0d,DIMS[0]],"Wi0"),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]],"Wi1"),bi1=Z([DIMS[1]]),
           Wi2=W([f2d,DIMS[2]],"Wi2"),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]],"Wi3"),bi3=Z([DIMS[3]]),
           emb=W([V,DM],"emb"),pos=W([CAPLEN,DM],"pos"))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM],f"Wq{b}");P[f"Wk{b}"]=W([DM,DM],f"Wk{b}");P[f"Wv{b}"]=W([DM,DM],f"Wv{b}");P[f"Wo{b}"]=W([DM,DM],f"Wo{b}")
        P[f"f1_{b}"]=W([DM,FFN],f"f1_{b}");P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM],f"f2_{b}");P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]],f"Wt{b}");P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],f"proj{k}")
    P["W_DI"]=W([NS*CODE,PIX],"W_DI");P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,CAPLEN*V],"W_DT");P["B_DT"]=Z([CAPLEN*V])
    if MUP:
        # RMSNorm gains (ones, no RNG draws): image block-2..4 inputs (channel axis), text
        # pre-attention/pre-FFN per block. Baseline draw order above is untouched.
        P["gc2"]=tf.Variable(tf.ones([C1])); P["gc3"]=tf.Variable(tf.ones([C2])); P["gc4"]=tf.Variable(tf.ones([C3]))
        for b in range(NBLK):
            P[f"ga{b}"]=tf.Variable(tf.ones([DM])); P[f"gf{b}"]=tf.Variable(tf.ones([DM]))
        # untied top-down prediction weights (drawn after every baseline key; N(0,1) + MULT)
        tdc=[(1,CH,C1),(2,C1,C2),(3,C2,C3),(4,C3,C4)]                     # (block, ch_below, ch_this)
        for l,cb,ct in tdc:
            P[f"td_c{l}"]=W([3,3,cb,ct],f"td_c{l}"); P[f"tdb_c{l}"]=Z([cb])
        for b in range(1,NBLK):
            P[f"td_t{b}"]=W([DM,DM],f"td_t{b}"); P[f"tdb_t{b}"]=Z([DM])
    return P,c,MULT

def build_highways(seed, c):
    """HEP highway matrices, FA-fork discipline: dedicated generator, fixed draw order, non-
    trainable, never in P, never saved. Image seg->channel maps for the blocks (routing seg3->Z1),
    text seg b -> block b DM maps."""
    DIMS=c["DIMS"]; DM=c["DM"]
    gv=tf.random.Generator.from_seed(seed+20011)
    VHW={}
    img_route={1:3, 2:0, 3:1, 4:2}                                        # block -> eps segment
    chs={1:c["C1"],2:c["C2"],3:c["C3"],4:c["C4"]}
    for l in (1,2,3,4):
        VHW[f"Vi{l}"]=tf.Variable(gv.normal([DIMS[img_route[l]],chs[l]],stddev=SIGMA_V),trainable=False)
    for b in range(NBLK):
        VHW[f"Vt{b}"]=tf.Variable(gv.normal([DIMS[b],DM],stddev=SIGMA_V),trainable=False)
    VHW["__img_route"]=img_route
    return VHW

def make_ops(P,c,MULT,VHW):
    DM,C1,C2,C3,C4,BN,DIMS,FFN,HEAD=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"],c["HEAD"]
    betas=[REL_C*d for d in DIMS]
    ALL_W=[v for k,v in P.items()]
    if WOPT=="adamw":
        # slots double weight memory: fine to 1.5B, needs sharding thought before the 7.7B rung
        WSLOT_M=[tf.Variable(tf.zeros_like(v),trainable=False) for v in ALL_W]
        WSLOT_V=[tf.Variable(tf.zeros_like(v),trainable=False) for v in ALL_W]
        WSTEP_T=tf.Variable(0.0,trainable=False)
    def apply_wgrads(gr,lr):
        if WOPT=="lars":
            for v,gg in zip(ALL_W,gr):
                if gg is None: continue
                tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
        else:
            WSTEP_T.assign_add(1.0); t=WSTEP_T
            for v,gg,m,vv in zip(ALL_W,gr,WSLOT_M,WSLOT_V):
                if gg is None: continue
                if isinstance(gg,tf.IndexedSlices): gg=tf.convert_to_tensor(gg)   # emb gather grad
                m.assign(0.9*m+0.1*gg); vv.assign(0.999*vv+0.001*gg*gg)
                mh=m/(1.0-tf.pow(0.9,t)); vh=vv/(1.0-tf.pow(0.999,t))
                v.assign_sub(lr*(mh/(tf.sqrt(vh)+1e-8)+WD*v))
    def mm(key,t):
        m=MULT.get(key,1.0); return t if m==1.0 else t*m
    def rmsn(x,gk):
        if (not MUP) or gk is None: return x
        return P[gk]*x*tf.math.rsqrt(tf.reduce_mean(x*x,axis=-1,keepdims=True)+1e-8)
    # ---- blocks (the SAME functions serve the plain forward and the PCMAX per-block predictions;
    #      under PARITY they reduce byte-for-byte to the baseline forward) ----
    def img_block(l,h):
        h=rmsn(h,f"gc{l}" if l>1 else None)
        h=gelu(mm(f"c{l}",tf.nn.conv2d(h,P[f"c{l}"],1,"SAME"))+P[f"cb{l}"])
        return tf.nn.max_pool2d(h,2,2,"SAME")
    def img_taps(Z2,Z3,Z4):
        B=tf.shape(Z2)[0]
        f0=tf.reshape(Z2,[B,-1]); f1=tf.reshape(Z3,[B,-1]); f2=tf.reshape(Z4,[B,-1])
        f3=gelu(mm("wbn",f2@P["wbn"])+P["bbn"])
        return [gelu(mm("Wi0",f0@P["Wi0"])+P["bi0"]),gelu(mm("Wi1",f1@P["Wi1"])+P["bi1"]),
                gelu(mm("Wi2",f2@P["Wi2"])+P["bi2"]),gelu(mm("Wi3",f3@P["Wi3"])+P["bi3"])]
    def txt_embed(tk): return mm("emb",tf.gather(P["emb"],tk))+mm("pos",P["pos"])[None]
    def txt_block(b,x):
        B=tf.shape(x)[0]
        rs=lambda t: t if RSCALE==1.0 else RSCALE*t                        # no-op under PARITY: exact baseline graph
        xin=rmsn(x,f"ga{b}")
        q,k_,v=mm(f"Wq{b}",xin@P[f"Wq{b}"]),mm(f"Wk{b}",xin@P[f"Wk{b}"]),mm(f"Wv{b}",xin@P[f"Wv{b}"])
        sp=lambda t: tf.transpose(tf.reshape(t,[B,CAPLEN,HEADS,HEAD]),[0,2,1,3])
        a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
        ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,CAPLEN,DM])
        x=x+rs(mm(f"Wo{b}",ctx@P[f"Wo{b}"]))
        h=rmsn(x,f"gf{b}")
        x=x+rs(mm(f"f2_{b}",gelu(mm(f"f1_{b}",h@P[f"f1_{b}"])+P[f"fb1_{b}"])@P[f"f2_{b}"])+P[f"fb2_{b}"])
        return x
    def txt_tap(b,x): return gelu(mm(f"Wt{b}",tf.reduce_mean(x,1)@P[f"Wt{b}"])+P[f"bt{b}"])
    def enc_img(x):
        Z1=img_block(1,x); Z2=img_block(2,Z1); Z3=img_block(3,Z2); Z4=img_block(4,Z3)
        return img_taps(Z2,Z3,Z4)
    def enc_txt(tk):
        x=txt_embed(tk); tt=[]
        for b in range(NBLK):
            x=txt_block(b,x); tt.append(txt_tap(b,x))
        return tt
    # ---- untied top-down predictions (PCMAX only) ----
    def g_img(l,Zl,shp_below):
        B=tf.shape(Zl)[0]
        out=tf.nn.conv2d_transpose(Zl,P[f"td_c{l}"],tf.stack([B,shp_below[0],shp_below[1],shp_below[2]]),2,"SAME")
        return mm(f"td_c{l}",out)+P[f"tdb_c{l}"]
    def g_txt(b,Zb): return mm(f"td_t{b}",Zb@P[f"td_t{b}"])+P[f"tdb_t{b}"]
    # ---- family ops (byte-matched math, muP forward) ----
    def code_of(S): return tf.concat([gelu(mm(f"proj{k}",S[k]@P[f"proj{k}"])) for k in range(NS)],axis=1)
    def dec_img(S): return tf.nn.sigmoid(mm("W_DI",code_of(S)@P["W_DI"])+P["B_DI"])
    def dec_txt(S): return mm("W_DT",code_of(S)@P["W_DT"])+P["B_DT"]
    def F_energy(S,it,tt,igt,tgt,red):
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        return 0.5*red(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
    @tf.function
    def get_taps(x,tk): return enc_img(x),enc_txt(tk)
    def relax_full(S,it,tt,igt,tgt,n):
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tp: tp.watch(Sv); f=F_energy(Sv,it,tt,igt,tgt,tf.reduce_sum)
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    # CAP_EAGER_WSTEP / CAP_RECOMPUTE inherited verbatim (see run_coupling_capacity.py for the FFN
    # NaN history). Under RMSNorm the NaN is expected moot for Bmu/PCMAX -- the smoke prints confirm.
    RECOMPUTE = os.environ.get("CAP_RECOMPUTE","0")=="1"
    EAGER_WSTEP = os.environ.get("CAP_EAGER_WSTEP","0")=="1" or RECOMPUTE
    enc_img_w = tf.recompute_grad(enc_img) if RECOMPUTE else enc_img
    enc_txt_w = tf.recompute_grad(enc_txt) if RECOMPUTE else enc_txt
    def _maybe_compile(f): return f if EAGER_WSTEP else tf.function(f)
    @_maybe_compile
    def weight_step(x,tk,S,igt,tgt,lr):
        with tf.GradientTape() as t: t.watch(ALL_W); F=F_energy(S,enc_img_w(x),enc_txt_w(tk),igt,tgt,tf.reduce_mean)
        gr=t.gradient(F,ALL_W)
        apply_wgrads(gr,lr)
        return F, tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    def relax_mono(S,taps,decfn,tgt,n):
        Sv=[tf.identity(s) for s in S]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_sum(tf.add_n([mse(Sv[k]-taps[k]) for k in range(NS)])+A_GEN*mse(decfn(Sv)-tgt))
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    def l2n(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))
    def infonce(zi,zt,temp):
        logits=tf.matmul(zi,zt,transpose_b=True)/temp; B=tf.shape(zi)[0]; lab=tf.range(B)
        return 0.5*(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=logits))
                   +tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=tf.transpose(logits))))
    @tf.function
    def warmup_step(xb,tkb,lr,temp):
        with tf.GradientTape() as t: t.watch(ALL_W); zi,zt=latents(xb,tkb); L=infonce(zi,zt,temp)
        gr=t.gradient(L,ALL_W)
        apply_wgrads(gr,lr)
        return L
    # ============================ PCMAX arm ops ============================
    s1,s2,s3,s4=RES//2,RES//4,RES//8,RES//16
    IMG_SHPS=[(RES,RES,CH),(s1,s1,C1),(s2,s2,C2),(s3,s3,C3),(s4,s4,C4)]    # index l = shape of Z_l (0 = clamped input)
    def F_pcmax(Zi,Zt,S,x,tk,igt,tgt,red):
        e_bu = mse(Zi[0]-img_block(1,x))+mse(Zi[1]-img_block(2,Zi[0]))\
              +mse(Zi[2]-img_block(3,Zi[1]))+mse(Zi[3]-img_block(4,Zi[2]))
        x0=txt_embed(tk)
        e_bu += mse(Zt[0]-txt_block(0,x0))+mse(Zt[1]-txt_block(1,Zt[0]))\
               +mse(Zt[2]-txt_block(2,Zt[1]))+mse(Zt[3]-txt_block(3,Zt[2]))
        e_td = mse(x-g_img(1,Zi[0],IMG_SHPS[0]))+mse(Zi[0]-g_img(2,Zi[1],IMG_SHPS[1]))\
              +mse(Zi[1]-g_img(3,Zi[2],IMG_SHPS[2]))+mse(Zi[2]-g_img(4,Zi[3],IMG_SHPS[3]))
        for b in range(1,NBLK): e_td += mse(Zt[b-1]-g_txt(b,Zt[b]))
        it=img_taps(Zi[1],Zi[2],Zi[3]); tt=[txt_tap(b,Zt[b]) for b in range(NBLK)]
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        gen=mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)
        return 0.5*red(e_bu + BIDIR_W*e_td + A_CROSS*cross + A_GEN*gen)
    def taps_from_states(Zi,Zt):
        return img_taps(Zi[1],Zi[2],Zi[3]),[txt_tap(b,Zt[b]) for b in range(NBLK)]
    @tf.function
    def ff_states(x,tk):
        Z1=img_block(1,x); Z2=img_block(2,Z1); Z3=img_block(3,Z2); Z4=img_block(4,Z3)
        x0=txt_embed(tk); T=[]
        h=x0
        for b in range(NBLK): h=txt_block(b,h); T.append(h)
        it=img_taps(Z2,Z3,Z4); tt=[txt_tap(b,T[b]) for b in range(NBLK)]
        S=[0.5*(it[k]+tt[k]) for k in range(NS)]
        return [Z1,Z2,Z3,Z4],T,S
    @tf.function
    def hep_step(Zi,Zt,S,Mi,Mt,Ms,Vi,Vt,Vs,x,tk,igt,tgt,tstep,alpha):
        states=list(Zi)+list(Zt)+list(S)
        with tf.GradientTape() as tp:
            tp.watch(states); F=F_pcmax(states[:4],states[4:8],states[8:],x,tk,igt,tgt,tf.reduce_sum)
        gr=tp.gradient(F,states)
        mom=list(Mi)+list(Mt)+list(Ms); vel=list(Vi)+list(Vt)+list(Vs)
        new_s=[]; new_m=[]; new_v=[]
        for z,gg,m,v in zip(states,gr,mom,vel):
            if STATE_OPT=="adam":
                m=0.9*m+0.1*gg; v=0.999*v+0.001*gg*gg
                mh=m/(1.0-tf.pow(0.9,tstep)); vh=v/(1.0-tf.pow(0.999,tstep))
                z=z-STATE_LR*mh/(tf.sqrt(vh)+1e-8)
            else:
                z=z-STATE_LR*gg
            new_s.append(z); new_m.append(m); new_v.append(v)
        Zi2,Zt2,S2=new_s[:4],new_s[4:8],new_s[8:]
        if alpha>0.0:
            it,tt=taps_from_states(Zi2,Zt2)
            zi_raw=tf.concat(it,1); zt_raw=tf.concat(tt,1)
            with tf.GradientTape() as tq:
                tq.watch([zi_raw,zt_raw]); L=infonce(l2n(zi_raw),l2n(zt_raw),tf.constant(TEMP,tf.float32))
            gi,gt=tq.gradient(L,[zi_raw,zt_raw])
            ei=[-tf.stop_gradient(s) for s in tf.split(gi,DIMS,axis=1)]
            et=[-tf.stop_gradient(s) for s in tf.split(gt,DIMS,axis=1)]
            if ANORM:
                nrm=lambda e: e*tf.math.rsqrt(tf.reduce_mean(e*e,axis=1,keepdims=True)+1e-30)
                ei=[nrm(e) for e in ei]; et=[nrm(e) for e in et]
            route=VHW["__img_route"]
            hwi=[STATE_LR*alpha*(ei[route[l]]@VHW[f"Vi{l}"])[:,None,None,:] for l in (1,2,3,4)]
            hwt=[STATE_LR*alpha*(et[b]@VHW[f"Vt{b}"])[:,None,:] for b in range(NBLK)]
            # highway-to-local-update ratio per interior state: the calibration number for alpha
            # (injection is LINEAR in alpha, so one measured ratio fixes the whole grid)
            loc=[o-n for o,n in zip(states[:8],new_s[:8])]
            ratios=tf.stack([tf.norm(h)/(tf.norm(l_)+1e-30) for h,l_ in
                             zip([tf.broadcast_to(h,tf.shape(z)) for h,z in zip(hwi+hwt,new_s[:8])],loc)])
            Zi2=[Zi2[l-1]+hwi[l-1] for l in (1,2,3,4)]
            Zt2=[Zt2[b]+hwt[b] for b in range(NBLK)]
        else:
            ratios=tf.zeros([8])
        return Zi2,Zt2,S2,new_m[:4],new_m[4:8],new_m[8:],new_v[:4],new_v[4:8],new_v[8:],F,ratios
    # alpha rides hep_step as a PYTHON float: it is a trace-time constant (two traces at most, 0.0
    # and ALPHA for the diagnostic), so the highway branch is compiled in or out, never a tensor bool.
    def relax_hep(x,tk,igt,tgt,n,alpha,diag=False):
        Zi,Zt,S=ff_states(x,tk)
        Mi=[tf.zeros_like(z) for z in Zi]; Mt=[tf.zeros_like(z) for z in Zt]; Ms=[tf.zeros_like(z) for z in S]
        Vi=[tf.zeros_like(z) for z in Zi]; Vt=[tf.zeros_like(z) for z in Zt]; Vs=[tf.zeros_like(z) for z in S]
        rows=[]; Ftr=[]; rrows=[]
        for i in range(n):
            prev=(list(Zi)+list(Zt)+list(S)) if diag else None
            Zi,Zt,S,Mi,Mt,Ms,Vi,Vt,Vs,F,ratios=hep_step(Zi,Zt,S,Mi,Mt,Ms,Vi,Vt,Vs,x,tk,igt,tgt,
                                                        tf.constant(float(i+1),tf.float32),float(alpha))
            Ftr.append(float(F))
            if diag:
                cur=list(Zi)+list(Zt)+list(S)
                rows.append([float(tf.norm(c-p)/(tf.norm(p)+1e-9)) for c,p in zip(cur,prev)])
                rrows.append([float(v) for v in ratios.numpy()])
        return Zi,Zt,S,Ftr,rows,rrows
    @_maybe_compile
    def weight_step_local(x,tk,Zi,Zt,S,igt,tgt,lr,jointw):
        with tf.GradientTape() as t:
            t.watch(ALL_W)
            F=F_pcmax(Zi,Zt,S,x,tk,igt,tgt,tf.reduce_mean)
            L=F
            if JOINTW>0:
                it,tt=taps_from_states(Zi,Zt)
                L=F+jointw*infonce(l2n(tf.concat(it,1)),l2n(tf.concat(tt,1)),tf.constant(TEMP,tf.float32))
        gr=t.gradient(L,ALL_W)
        apply_wgrads(gr,lr)
        return F, tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    return dict(get_taps=get_taps,relax_full=relax_full,weight_step=weight_step,relax_mono=relax_mono,
                dec_img=dec_img,dec_txt=dec_txt,latents=latents,warmup_step=warmup_step,
                ff_states=ff_states,relax_hep=relax_hep,weight_step_local=weight_step_local,
                enc_img=enc_img,enc_txt=enc_txt)

def movement(P,P0):
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

# E4 uniformity metric, byte-copied from analysis_latent_geometry.py (t=2.0 fixed)
def unif_np(Z, t=2.0, cap=2000, rng=None):
    rng = rng or np.random.RandomState(1)
    if len(Z) > cap: Z = Z[rng.choice(len(Z), cap, replace=False)]
    d2 = np.maximum(2.0 - 2.0*(Z @ Z.T), 0.0)
    iu = np.triu_indices(len(Z), 1)
    return float(np.log(np.mean(np.exp(-t * d2[iu])) + 1e-30))

# ============================ BUILD ONCE, snapshot init ============================
P,c,MULT=build(WMUL,SEED)
VHW=build_highways(SEED,c) if ("PCMAX" in ARMS and ALPHA>0) else {"__img_route":{1:3,2:0,3:1,4:2}}
ops=make_ops(P,c,MULT,VHW)
NP=int(sum(int(np.prod(v.shape)) for v in P.values()))
INIT_DIR=os.path.join(CKPT, f"init_w{WMUL}_seed{SEED}")
if LOWHOST:
    os.makedirs(INIT_DIR, exist_ok=True)
    for k in P:                                                            # one tensor at a time: host peak = largest tensor
        fp=os.path.join(INIT_DIR, f"{k}.npy")
        if not os.path.exists(fp): np.save(fp, P[k].numpy())
    P_init=None
    print(f"[lowhost] init streamed to {INIT_DIR}; no host-resident weight copy",flush=True)
else:
    P_init={k:v.numpy().copy() for k,v in P.items()}
def reset():
    if LOWHOST: return                                                     # single fresh-process arm: weights ARE the init
    [P[k].assign(P_init[k]) for k in P]
def movement_now():
    if not LOWHOST: return movement(P,P_init)
    try:
        num=0.0; den=0.0
        for k in P:
            b=np.load(os.path.join(INIT_DIR, f"{k}.npy"), mmap_mode="r")
            shp=P[k].shape
            rows=int(shp[0]) if len(shp)>0 else 1
            per_row=int(np.prod(shp[1:])) if len(shp)>1 else 1
            step=max(1, int(64e6)//max(per_row,1))
            for i in range(0, rows, step):
                a=P[k][i:i+step].numpy().astype("float64")
                bb=np.asarray(b[i:i+step], dtype="float64")
                num+=float(((a-bb)**2).sum()); den+=float((bb**2).sum())
        return math.sqrt(num)/(math.sqrt(den)+1e-9)
    except Exception as e:
        print(f"    !! movement_now failed ({type(e).__name__}: {e}); recording movement=None",flush=True)
        return None

# ============================ AUTO-BATCH FALLBACK ============================
OOM_ERRS=(tf.errors.ResourceExhaustedError, tf.errors.InternalError)
def restore_init():
    if LOWHOST:
        for k in P: P[k].assign(np.load(os.path.join(INIT_DIR, f"{k}.npy")))
    else:
        [P[k].assign(P_init[k]) for k in P]

def pick_batch(requested):
    if not AUTOBATCH: return requested
    B=requested
    while B>=1:
        try:
            bi=tr_idx[:min(B,NTR)]
            x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi])
            igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
            if "PCMAX" in ARMS:
                Zi,Zt,S,_,_,_=ops["relax_hep"](x,tk,igt,tgt,1,ALPHA)
                _,mxw=ops["weight_step_local"](x,tk,[tf.constant(z.numpy()) for z in Zi],
                                               [tf.constant(z.numpy()) for z in Zt],
                                               [tf.constant(z.numpy()) for z in S],
                                               igt,tgt,tf.constant(0.0,tf.float32),tf.constant(JOINTW,tf.float32))
            else:
                it,tt=ops["get_taps"](x,tk)
                Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
                _,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,tf.constant(0.0,tf.float32))  # lr=0: no INTENDED weight change
            if not np.isfinite(float(mxw)):
                print(f"[autobatch] trial at BATCHJ={B} produced non-finite weights (Inf gradient met the "
                      f"lr=0 update; 0*Inf=NaN); restoring init. The rung will surface the divergence "
                      f"honestly at step 0 instead of training on corrupted weights.",flush=True)
                restore_init()
            else:
                print(f"[autobatch] BATCHJ={B} fits (trial step ok, weights finite)",flush=True)
            return B
        except OOM_ERRS as e:
            print(f"[autobatch] BATCHJ={B} OOM ({type(e).__name__}), halving",flush=True)
            B//=2
    raise RuntimeError("no batch size fits, even BATCHJ=1")
BATCHJ_REQ=BATCHJ; BATCHJ=pick_batch(BATCHJ)
steps_per_epoch = max(1, math.ceil(NTR/BATCHJ)); JOINT_STEPS = EPOCHS*steps_per_epoch
WARM_EPOCHS_EQ = math.ceil(WARMUP*BATCH/max(1,NTR))
LONG_STEPS = (EPOCHS+WARM_EPOCHS_EQ)*steps_per_epoch
print(f"=== PCMAX CAPACITY === smoke={SMOKE} arms={ARMS} parity={int(PARITY)} | wmul={WMUL} params={NP/1e6:.1f}M ({NP:,}) | "
      f"train={NTR} eval={NEV} V={V} | BATCHJ={BATCHJ} (requested {BATCHJ_REQ}) EPOCHS={EPOCHS} "
      f"({JOINT_STEPS} steps) lr={LR} | ckpt_every={CKPT_EVERY} resume={RESUME} | chance={1/max(NEV,1):.5f} bar >3/{NEV}",flush=True)
if "PCMAX" in ARMS:
    print(f"[pcmax] alpha={ALPHA} T={T_INFER} state_opt={STATE_OPT} state_lr={STATE_LR} sigma_v={SIGMA_V} "
          f"bidir_w={BIDIR_W} jointw={JOINTW} fitstop={FITSTOP} rscale={RSCALE:.4f} wopt={WOPT} wd={WD} anorm={int(ANORM)}",flush=True)

# ============================ READOUTS (byte-matched + uniformity) ============================
def readouts(idx):
    M=len(idx); t2i=np.zeros((M,RES,RES,CH)); i2i=np.zeros((M,RES,RES,CH)); i2t_hits=0; i2t_tot=0
    for st in range(0,M,READB):
        bi=[int(idx[j]) for j in range(st,min(st+READB,M))]
        x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); it,tt=ops["get_taps"](x,tk)
        igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
        St=ops["relax_mono"]([tf.identity(tt[k]) for k in range(NS)], tt, ops["dec_txt"], tgt, GEN_INFER)
        t2i[st:st+len(bi)]=ops["dec_img"](St).numpy().reshape(len(bi),RES,RES,CH)
        Si=ops["relax_mono"]([tf.identity(it[k]) for k in range(NS)], it, ops["dec_img"], igt, GEN_INFER)
        i2i[st:st+len(bi)]=ops["dec_img"](Si).numpy().reshape(len(bi),RES,RES,CH)
        pred=ops["dec_txt"](Si).numpy().reshape(len(bi),CAPLEN,V).argmax(-1)
        i2t_hits+=int((pred==toks[bi]).sum()); i2t_tot+=pred.size
    real=imgs[idx]
    diversity=float(np.mean(np.std(t2i,0))/(np.std(real)+1e-9))
    out_range=float((t2i.max(0)-t2i.min(0)).mean())
    A=t2i.reshape(M,-1).astype("float32"); Bm=real.reshape(M,-1).astype("float32"); Bn=(Bm**2).sum(1)
    nn=np.empty(M,"int64")
    for st in range(0,M,256): nn[st:st+256]=np.argmin(Bn[None,:]-2.0*(A[st:st+256]@Bm.T),1)
    retr=float(np.mean(nn==np.arange(M)))
    recon=float(np.mean((i2i-real)**2)); recon_base=float(np.mean((train_mean_img[None]-real)**2))
    i2t=i2t_hits/max(1,i2t_tot)
    ZIl=[]; ZTl=[]
    for st in range(0,M,READB):
        bi=idx[st:st+READB]; ZI,ZT=ops["latents"](tf.constant(imgs[bi]), tf.constant(toks[bi])); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
    ZI=np.concatenate(ZIl,0); ZT=np.concatenate(ZTl,0)
    align_cos=float(np.mean(np.sum(ZI*ZT,1)))
    lat_hits=int(np.sum(np.argmax(ZT@ZI.T,1)==np.arange(M)))
    return dict(M=M,diversity=diversity,out_range=out_range,retr=retr,hits=int(round(retr*M)),chance=1.0/M,
                recon=recon,recon_base=recon_base,i2t=i2t,align_cos=align_cos,
                lat_retr=lat_hits/max(M,1),lat_hits=lat_hits,
                unif_img=unif_np(ZI),unif_txt=unif_np(ZT)), t2i

mode_char=int(np.bincount(toks[tr_idx].reshape(-1),minlength=V).argmax())
def i2t_base_on(idx): return float(np.mean(toks[idx]==mode_char))

def latent_readout(idx):
    M=len(idx); ZIl=[]; ZTl=[]
    for st in range(0,M,READB):
        bi=idx[st:st+READB]; ZI,ZT=ops["latents"](tf.constant(imgs[bi]), tf.constant(toks[bi])); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
    ZI=np.concatenate(ZIl,0); ZT=np.concatenate(ZTl,0)
    return dict(align_cos=float(np.mean(np.sum(ZI*ZT,1))), lat_retr=float(np.mean(np.argmax(ZT@ZI.T,1)==np.arange(M))))

def sigma_above_chance(hits, M):
    p=1.0/M; expd=M*p; sd=math.sqrt(M*p*(1-p))
    return (hits-expd)/sd if sd>0 else float("nan")

# ============================ CHECKPOINT / RESUME (single-arm A or PCMAX) ============================
def state_path(arm): return os.path.join(CKPT, f"cap_state_{arm}_w{WMUL}_seed{SEED}.npz")

def save_state(arm, step, order, ptr, rs, Fhist):
    st=rs.get_state()                                                        # ('MT19937', keys, pos, has_gauss, cached)
    meta=dict(__step=np.int64(step), __order=order.astype("int64"), __ptr=np.int64(ptr),
              __rng_keys=st[1].astype("uint32"), __rng_pos=np.int64(st[2]),
              __rng_hg=np.int64(st[3]), __rng_cg=np.float64(st[4]),
              __fhist=np.asarray(Fhist[-200:], "float64"),
              __alpha=np.float64(ALPHA), __t_infer=np.int64(T_INFER))
    if LOWHOST:
        sd=state_path(arm)+".dir"; os.makedirs(sd, exist_ok=True)
        for k in P: np.save(os.path.join(sd, f"{k}.npy"), P[k].numpy())     # streamed: host peak = one tensor
        tmp=os.path.join(sd, "meta.tmp.npz"); np.savez(tmp, **meta)
        os.replace(tmp, os.path.join(sd, "meta.npz"))                       # meta last = commit marker
        print(f"    [ckpt] saved state at step {step} (streamed dir)",flush=True)
    else:
        payload={f"W__{k}": P[k].numpy() for k in P}; payload.update(meta)
        tmp=state_path(arm)+".tmp.npz"
        np.savez(tmp, **payload); os.replace(tmp, state_path(arm))
        print(f"    [ckpt] saved state at step {step} ({os.path.getsize(state_path(arm))/2**30:.1f} GB)",flush=True)

def load_state(arm, rs):
    if RESUME in ("0","no"): return None
    sd=state_path(arm)+".dir"; mp=os.path.join(sd, "meta.npz")
    if LOWHOST and os.path.exists(mp):
        z=np.load(mp)
        for k in P: P[k].assign(np.load(os.path.join(sd, f"{k}.npy")))      # streamed
    elif (not LOWHOST) and os.path.exists(state_path(arm)):
        z=np.load(state_path(arm))
        for k in P: P[k].assign(z[f"W__{k}"])
    else:
        return None
    rs.set_state(("MT19937", z["__rng_keys"], int(z["__rng_pos"]), int(z["__rng_hg"]), float(z["__rng_cg"])))
    step=int(z["__step"]); order=z["__order"].astype("int64").copy(); ptr=int(z["__ptr"])
    fh=list(z["__fhist"])
    print(f"    [ckpt] RESUMED arm {arm} from step {step} (data order preserved; bit-exact continuation "
          f"not guaranteed under GPU nondeterminism)",flush=True)
    return step, order, ptr, fh

def save_final(name):
    try:
        if LOWHOST:
            fd=os.path.join(CKPT,f"cap_{name}_w{WMUL}_seed{SEED}.dir"); os.makedirs(fd, exist_ok=True)
            for k in P: np.save(os.path.join(fd, f"{k}.npy"), P[k].numpy())  # streamed final ckpt (dir layout)
            if MUP: np.save(os.path.join(fd,"__pcmax.npy"), np.int64(1))
        else:
            payload={k:P[k].numpy() for k in P}
            if MUP: payload["__pcmax"]=np.int64(1)                           # arch marker for the probes
            np.savez(os.path.join(CKPT,f"cap_{name}_w{WMUL}_seed{SEED}.npz"), **payload)
    except Exception as e: print(f"    !! ckpt save failed: {e}",flush=True)

# ============================ PHASES ============================
warm_rs=np.random.RandomState(SEED+11); ep_rs=np.random.RandomState(SEED+7)
def warmup_phase(steps):
    if steps<=0: return
    lrt=tf.constant(WARMLR,tf.float32); tmp=tf.constant(TEMP,tf.float32); t0=time.time()
    for s in range(steps):
        b=warm_rs.choice(NTR, size=min(BATCH,NTR), replace=False); bi=tr_idx[b]
        L=float(ops["warmup_step"](tf.constant(imgs[bi]), tf.constant(toks[bi]), lrt, tmp))
        if (s+1)%LOG_EVERY==0: print(f"    [warmup] {s+1:5d} infonce={L:.4f} t={(time.time()-t0)/60:.1f}m",flush=True)

def joint_phase(arm, total_steps, jointw):
    tmp=tf.constant(TEMP,tf.float32); diverged=False; t0=time.time()
    resumed=load_state(arm, ep_rs) if (CKPT_EVERY and arm=="A") else None
    if resumed: start,order,ptr,Fhist=resumed
    else: start,order,ptr,Fhist=0,ep_rs.permutation(NTR),0,[]
    for s in range(start, total_steps):
        if ptr+BATCHJ>NTR: order=ep_rs.permutation(NTR); ptr=0
        bi=tr_idx[order[ptr:ptr+BATCHJ]]; ptr+=BATCHJ
        cur=LR*min(1.0,(s+1)/RAMP) if RAMP>0 else LR; lrt=tf.constant(cur,tf.float32)
        x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
        it,tt=ops["get_taps"](x,tk)
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt); F=float(F); mxw=float(mxw); Fhist.append(F)
        if jointw>0:
            wb=warm_rs.choice(NTR, size=min(BATCH,NTR), replace=False); wi=tr_idx[wb]
            ops["warmup_step"](tf.constant(imgs[wi]), tf.constant(toks[wi]), tf.constant(cur*jointw,tf.float32), tmp)
        if not (np.isfinite(F) and mxw<DIVERGE_W):
            diverged=True; print(f"    !! DIVERGENCE step {s}: F={F:.3e} max|w|={mxw:.2e}",flush=True); break
        if (s+1)%LOG_EVERY==0:
            mv="" if LOWHOST else f" move={movement(P,P_init)*100:.1f}%"     # lowhost: movement only at the end (host cost)
            print(f"    [joint] {s+1:5d}/{total_steps} F={F:.4e}{mv} lr={cur:.1e} t={(time.time()-t0)/60:.1f}m",flush=True)
        if CKPT_EVERY and arm=="A" and (s+1)%CKPT_EVERY==0 and (s+1)<total_steps:
            save_state(arm, s+1, order, ptr, ep_rs, Fhist)
    return Fhist, diverged

def diag_print(rows, tag, rrows=None):
    names=["Zi1","Zi2","Zi3","Zi4","Zt0","Zt1","Zt2","Zt3","S0","S1","S2","S3"]
    print(f"    [diag {tag}] per-step RMS(dZ)/RMS(Z):",flush=True)
    print("      step " + " ".join(f"{n:>8s}" for n in names),flush=True)
    for i,r in enumerate(rows):
        print(f"      {i+1:4d} " + " ".join(f"{v:8.1e}" for v in r),flush=True)
    if rrows and any(any(v>0 for v in r) for r in rrows):
        print(f"    [diag {tag}] highway/local-update norm ratio (interior states; linear in alpha):",flush=True)
        print("      step " + " ".join(f"{n:>8s}" for n in names[:8]),flush=True)
        for i,r in enumerate(rrows):
            print(f"      {i+1:4d} " + " ".join(f"{v:8.1e}" for v in r),flush=True)
        mr=float(np.mean(rrows[0]))
        if mr>0: print(f"    [diag] step-1 mean ratio={mr:.3e} at alpha={ALPHA} -> alpha for ratio 0.3 ~= {0.3*ALPHA/mr:.3e}",flush=True)

def joint_phase_pcmax(total_steps):
    diverged=False; t0=time.time(); fit_stop_step=None
    resumed=load_state("PCMAX", ep_rs) if CKPT_EVERY else None
    if resumed: start,order,ptr,Fhist=resumed
    else: start,order,ptr,Fhist=0,ep_rs.permutation(NTR),0,[]
    tr_sub = tr_idx if NTR<=READTRAIN else tr_idx[np.random.RandomState(SEED+3).choice(NTR,READTRAIN,replace=False)]
    for s in range(start, total_steps):
        if ptr+BATCHJ>NTR: order=ep_rs.permutation(NTR); ptr=0
        bi=tr_idx[order[ptr:ptr+BATCHJ]]; ptr+=BATCHJ
        cur=LR*min(1.0,(s+1)/RAMP) if RAMP>0 else LR; lrt=tf.constant(cur,tf.float32)
        x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
        if DIAG and s==start:
            # HEP Fig-1 analog on the first batch: same batch, alpha=0 then alpha=ALPHA
            _,_,_,F0,r0,_  =ops["relax_hep"](x,tk,igt,tgt,T_INFER,0.0,diag=True);   diag_print(r0,"alpha=0")
            _,_,_,F1,r1,rr1=ops["relax_hep"](x,tk,igt,tgt,T_INFER,ALPHA,diag=True); diag_print(r1,f"alpha={ALPHA}",rr1)
            print(f"    [diag] F trace alpha=0:      " + " ".join(f"{f:.4e}" for f in F0),flush=True)
            print(f"    [diag] F trace alpha={ALPHA}: " + " ".join(f"{f:.4e}" for f in F1),flush=True)
        Zi,Zt,S,Ftr,_,_=ops["relax_hep"](x,tk,igt,tgt,T_INFER,ALPHA)
        F,mxw=ops["weight_step_local"](x,tk,[tf.stop_gradient(z) for z in Zi],[tf.stop_gradient(z) for z in Zt],
                                       [tf.stop_gradient(z) for z in S],igt,tgt,lrt,tf.constant(JOINTW,tf.float32))
        F=float(F); mxw=float(mxw); Fhist.append(F)
        if not (np.isfinite(F) and mxw<DIVERGE_W):
            diverged=True; print(f"    !! DIVERGENCE step {s}: F={F:.3e} max|w|={mxw:.2e}",flush=True); break
        if (s+1)%LOG_EVERY==0:
            mv="" if LOWHOST else f" move={movement(P,P_init)*100:.1f}%"
            print(f"    [pcmax] {s+1:5d}/{total_steps} F={F:.4e} relaxF {Ftr[0]:.3e}->{Ftr[-1]:.3e}{mv} "
                  f"lr={cur:.1e} t={(time.time()-t0)/60:.1f}m",flush=True)
        if FITSTOP>0 and (s+1)%steps_per_epoch==0:
            fit=latent_readout(tr_sub)
            print(f"    [fitgate] epoch {(s+1)//steps_per_epoch}: train lat_retr={fit['lat_retr']:.4f} "
                  f"align={fit['align_cos']:.3f} (stop at >={FITSTOP})",flush=True)
            if fit["lat_retr"]>=FITSTOP:
                fit_stop_step=s+1; print(f"    [fitgate] FIT-STOP at step {s+1} (matched-fit criterion reached; "
                                         f"epochs beyond fit do not change the verdict)",flush=True); break
        if CKPT_EVERY and (s+1)%CKPT_EVERY==0 and (s+1)<total_steps:
            save_state("PCMAX", s+1, order, ptr, ep_rs, Fhist)
    return Fhist, diverged, fit_stop_step

def run_arm(name, do_warmup, joint_steps, jointw):
    print(f"\n----- ARM {name} (warmup={'yes' if do_warmup else 'no'}, joint_steps={joint_steps}, jointw={jointw}) -----",flush=True)
    reset(); t0=time.time()
    postwarm=None
    if do_warmup:
        warmup_phase(WARMUP)
        if NEV:
            postwarm=latent_readout(ev_idx)
            print(f"  ARM {name} POST-WARMUP held-out: align_cos={postwarm['align_cos']:.3f} lat_retr={postwarm['lat_retr']:.3f}",flush=True)
    fit_stop_step=None
    if name=="PCMAX":
        Fhist,diverged,fit_stop_step=joint_phase_pcmax(joint_steps)
    else:
        Fhist,diverged=joint_phase("A" if name=="A" else name, joint_steps, jointw)
    move=movement_now(); elapsed=(time.time()-t0)/60
    save_final(name)
    try: peak=tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: peak=None
    if diverged: return dict(name=name,diverged=True,move=move,elapsed=elapsed,postwarm=postwarm,peak_gpu_gb=peak)
    tr_sub = tr_idx if NTR<=READTRAIN else tr_idx[np.random.RandomState(SEED+3).choice(NTR,READTRAIN,replace=False)]
    m_tr,_=readouts(tr_sub); m_ev,_=(readouts(ev_idx) if NEV else (None,None))
    mv_s = "n/a" if move is None else f"{move*100:.1f}%"
    print(f"  ARM {name}: move={mv_s} | HELD-OUT lat {m_ev['lat_hits']}/{NEV} "
          f"({sigma_above_chance(m_ev['lat_hits'],NEV):+.1f} sigma, bar >3) align={m_ev['align_cos']:.3f} "
          f"unif_img/txt={m_ev['unif_img']:.2f}/{m_ev['unif_txt']:.2f} | gen retr={m_ev['retr']:.5f} "
          f"({m_ev['hits']}/{NEV}) diversity={m_ev['diversity']:.3f} recon={m_ev['recon']:.4f} "
          f"(base {m_ev['recon_base']:.4f}) i2t={m_ev['i2t']:.3f} | {elapsed:.1f} min",flush=True)
    return dict(name=name,diverged=False,move=move,elapsed=elapsed,train=m_tr,heldout=m_ev,postwarm=postwarm,
                peak_gpu_gb=peak,lat_hits=m_ev["lat_hits"],sigma=sigma_above_chance(m_ev["lat_hits"],NEV),
                fit_stop_step=fit_stop_step)

# ============================ RUN ARMS ============================
results={}
if "A" in ARMS:      results["arm_A"]=run_arm("A", False, JOINT_STEPS, 0.0)
if "B" in ARMS:      results["arm_B"]=run_arm("B", True,  JOINT_STEPS, JOINTW)
if "Bmu" in ARMS:    results["arm_Bmu"]=run_arm("Bmu", True, JOINT_STEPS, JOINTW)   # arm-B recipe under muP parameterization
if "A_long" in ARMS: results["arm_A_long"]=run_arm("A_long", False, LONG_STEPS, 0.0)
if "PCMAX" in ARMS:  results["arm_PCMAX"]=run_arm("PCMAX", False, JOINT_STEPS, JOINTW)  # no warmup: backprop-free by construction

dump=dict(config=dict(smoke=SMOKE,arms=ARMS,wmul=WMUL,params=NP,N_have=N_HAVE,N_train=NTR,N_eval=NEV,
                      RES=RES,CAPLEN=CAPLEN,V=V,lr=LR,batchj=BATCHJ,batchj_requested=BATCHJ_REQ,
                      epochs=EPOCHS,joint_steps=JOINT_STEPS,ramp=RAMP,jointw=JOINTW,seed=SEED,
                      n_infer=N_INFER,ckpt_every=CKPT_EVERY,lowhost=LOWHOST,
                      parity=PARITY,alpha=ALPHA,t_infer=T_INFER,sigma_v=SIGMA_V,state_lr=STATE_LR,
                      state_opt=STATE_OPT,bidir_w=BIDIR_W,fitstop=FITSTOP,rscale=RSCALE),
          **results, i2t_base_train=i2t_base_on(tr_idx), i2t_base_eval=(i2t_base_on(ev_idx) if NEV else None))
out=os.path.join(HERE,f"pcmax_capacity_w{WMUL}_seed{SEED}.json")
with open(out+".tmp","w") as fh: json.dump(dump,fh,indent=2)
os.replace(out+".tmp",out)
print(f"\nsaved: {out}",flush=True)

# ============================ PER-RUNG VERDICT (pre-registered branches) ============================
print(f"\n==================== PCMAX RUNG VERDICT (wmul={WMUL}, {NP/1e9:.2f}B params, bar >3/{NEV}) ====================",flush=True)
_prim="arm_PCMAX" if "arm_PCMAX" in results else ("arm_Bmu" if "arm_Bmu" in results else
      ("arm_A" if "arm_A" in results else ("arm_B" if "arm_B" in results else None)))
a=results.get(_prim) if _prim else None
if a is not None and not a.get("diverged") and a.get("train") is not None:
    _fit=a["train"].get("lat_retr")
    if _fit is not None: print(f"[{_prim}] MATCHED-FIT GATE: train lat_retr={_fit:.4f} (rung counts only if >=0.95 alongside the BP baseline's)",flush=True)
if a is None: print("VERDICT: no primary arm in this run.",flush=True)
elif a["diverged"]: print("VERDICT: DIVERGED. Report with trace; this run is a stability datum, not a coupling datum.",flush=True)
elif a["move"] is not None and a["move"]<MOVE_MIN: print(f"VERDICT: VOID (move {a['move']*100:.0f}% < {MOVE_MIN*100:.0f}% floor). Undertrained, not a negative.",flush=True)
else:
    ho=a["heldout"]
    if a["lat_hits"]>3:
        print(f"VERDICT: BEST-SHOT PC CROSSES THE BAR. held-out latent {a['lat_hits']}/{NEV} at {NP/1e9:.2f}B under "
              f"muPC+HEP. Do NOT write the word cured: queue seeds 1,2 and category_probe vs the banked BP/PC ckpts first.",flush=True)
    else:
        mvv = "n/a" if a["move"] is None else f"{a['move']*100:.0f}%"
        print(f"VERDICT: best-shot PC flat at this rung ({a['lat_hits']}/{NEV}, {a['sigma']:+.1f} sigma) with move {mvv}, "
              f"align {ho['align_cos']:.3f}, unif {ho['unif_img']:.2f}/{ho['unif_txt']:.2f}. The dissociation survives "
              f"the published scaling methods at this size; category_probe adjudicates the transfer claim.",flush=True)
if SMOKE:
    if MUP:
        # ground truth for the probe-compatibility gate: the driver's own tap outputs on 8 examples
        gi=[int(i) for i in tr_idx[:min(8,NTR)]]
        it,tt=ops["get_taps"](tf.constant(imgs[gi]),tf.constant(toks[gi]))
        np.savez(os.path.join(CKPT,"pcmax_smoke_taps.npz"), idx=np.asarray(gi),
                 **{f"it{k}":it[k].numpy() for k in range(NS)}, **{f"tt{k}":tt[k].numpy() for k in range(NS)})
        print(f"[SMOKE] tap ground truth saved to {os.path.join(CKPT,'pcmax_smoke_taps.npz')}",flush=True)
    print("\n[SMOKE] mechanics only. Confirms arm selection, autobatch trial, relax/diag path, ckpt save, "
          "json, verdict.",flush=True)
