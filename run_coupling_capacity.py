"""CAPACITY LADDER driver -- the banked 8k recipe at larger WMUL, scratch variant of
run_coupling_scale.py (the original is untouched; byte-matched semantics, arm A reproduces the banked
arm A exactly at the same WMUL/BATCHJ/seed). New experimental axis for the paper: does the coupling
failure persist as capacity grows from 156M toward 7.7B, with latent retrieval primary and the banked
bar (>3/N_eval) unchanged.

WHAT IS DIFFERENT FROM THE BANKED DRIVER (each one a documented deviation or an addition, not a change
to the frozen recipe):
  1. ARM SELECTION. RUNS1_ARMS (default "A"): the ladder runs arm A per rung; arm B and the A_long
     control are available where budget is spare.
  2. UNIFORMITY READOUT. readouts() additionally records the Wang-Isola uniformity of the held-out
     encoder concats (unif_img/unif_txt, t=2.0 fixed, function byte-copied from
     analysis_latent_geometry.py) so the capacity trend of mean-collapse alignment is a figure.
  3. REDUCED BATCHJ IS EXPECTED at large WMUL. The relaxation reduces with a SUM over the batch, so it
     is per-example and batch-invariant by construction; the weight step is a batch MEAN and gets
     noisier at small batch, which LARS normalizes in magnitude but not in direction. Deviation noted
     here once; every rung records its batch in the JSON. Epochs are matched across rungs (the epochs
     rule), so smaller batches mean more steps, not less exposure.
  4. AUTO-BATCH FALLBACK (CAP_AUTOBATCH=1). Before training, a trial joint step runs at the requested
     BATCHJ; on OOM the batch halves (floor 1) and the trial repeats. Protects multi-day chained jobs
     from a marginal probe pick; the chosen batch is recorded.
  5. CHECKPOINT AND RESUME (CAP_CKPT_EVERY steps, 0=off; use for any job projected over 24h). Saves
     weights plus the exact loop state (step, epoch order, pointer, epoch RNG state) atomically
     (tmp+rename, latest only). On restart with CAP_RESUME=auto the run continues from the saved step
     with the data order exactly preserved. RNG caveat: continuation is exact in data order and in
     every algorithmic choice (arm A's joint loop consumes no randomness beyond the epoch shuffle),
     but bit-exact equality with an unbroken run is not guaranteed under GPU reduction nondeterminism.
     Resume is supported for arm A only (warm-up arms consume a second RNG stream mid-run).

ENV: RUNS1_* exactly as run_coupling_scale.py (WMUL is the size knob; defaults here are the ladder
scale N_TRAIN=8000 N_EVAL=2000 EPOCHS=150) plus RUNS1_ARMS("A") CAP_CKPT_EVERY(0) CAP_RESUME(auto|0)
CAP_AUTOBATCH(1).
OUT: coupling_capacity_w{WMUL}_seed{SEED}.json, checkpoints cap_{arm}_w{WMUL}_seed{SEED}.npz in
RUNS1_CKPT (same key layout as cs_*.npz so analysis_move_decomp.py applies).
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
ARMS   = [a.strip() for a in os.environ.get("RUNS1_ARMS", "A,B" if SMOKE else "A").split(",") if a.strip()]
CKPT_EVERY = int(os.environ.get("CAP_CKPT_EVERY", 0))
RESUME = os.environ.get("CAP_RESUME", "auto")
AUTOBATCH = os.environ.get("CAP_AUTOBATCH", "1") == "1"
os.makedirs(CKPT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"
assert all(a in ("A","B","A_long") for a in ARMS), "RUNS1_ARMS must be a subset of A,B,A_long"
if CKPT_EVERY: assert ARMS == ["A"], "checkpoint/resume is supported for the arm-A-only ladder runs"

# recipe constants (identical to run_coupling_scale.py)
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

# ============================ MODEL (byte-copied) ============================
def cfg(wmul):
    r=lambda x:max(4,int(round(x*wmul)))
    DM=r(B_DM); DM-=DM%HEADS
    return dict(DM=max(HEADS,DM),C1=r(B_C1),C2=r(B_C2),C3=r(B_C3),C4=r(B_C4),BN=r(B_BN),
                DIMS=[r(d) for d in B_DIMS],FFN=r(B_FFN),HEAD=max(1,(max(HEADS,DM))//HEADS))

def build(wmul, seed):
    c=cfg(wmul); DM,C1,C2,C3,C4,BN,DIMS,FFN=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"]
    s2=RES//4; s3=RES//8; s4=RES//16; f0d,f1d,f2d=s2*s2*C2, s3*s3*C3, s4*s4*C4
    c["f0d"],c["f1d"],c["f2d"]=f0d,f1d,f2d
    g=tf.random.Generator.from_seed(seed)
    def W(shape,key=""):
        sd=DEC_SD if (key.startswith("proj") or key in ("W_DI","W_DT")) else 1.0/np.sqrt(np.prod(shape[:-1]))
        return tf.Variable(g.normal(shape,stddev=sd))
    def Z(shape): return tf.Variable(tf.zeros(shape))
    P=dict(c1=W([3,3,CH,C1]),cb1=Z([C1]),c2=W([3,3,C1,C2]),cb2=Z([C2]),c3=W([3,3,C2,C3]),cb3=Z([C3]),
           c4=W([3,3,C3,C4]),cb4=Z([C4]),wbn=W([f2d,BN]),bbn=Z([BN]),
           Wi0=W([f0d,DIMS[0]]),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]]),bi1=Z([DIMS[1]]),
           Wi2=W([f2d,DIMS[2]]),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]]),bi3=Z([DIMS[3]]),
           emb=W([V,DM]),pos=W([CAPLEN,DM]))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM]);P[f"Wk{b}"]=W([DM,DM]);P[f"Wv{b}"]=W([DM,DM]);P[f"Wo{b}"]=W([DM,DM])
        P[f"f1_{b}"]=W([DM,FFN]);P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM]);P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]]);P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],f"proj{k}")
    P["W_DI"]=W([NS*CODE,PIX],"W_DI");P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,CAPLEN*V],"W_DT");P["B_DT"]=Z([CAPLEN*V])
    return P,c

def make_ops(P,c):
    DM,C1,C2,C3,C4,BN,DIMS,FFN,HEAD=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"],c["HEAD"]
    betas=[REL_C*d for d in DIMS]; ALL_W=list(P.values())
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
    def code_of(S): return tf.concat([gelu(S[k]@P[f"proj{k}"]) for k in range(NS)],axis=1)
    def dec_img(S): return tf.nn.sigmoid(code_of(S)@P["W_DI"]+P["B_DI"])
    def dec_txt(S): return code_of(S)@P["W_DT"]+P["B_DT"]
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
    @tf.function
    def weight_step(x,tk,S,igt,tgt,lr):
        with tf.GradientTape() as t: t.watch(ALL_W); F=F_energy(S,enc_img(x),enc_txt(tk),igt,tgt,tf.reduce_mean)
        gr=t.gradient(F,ALL_W)
        for v,gg in zip(ALL_W,gr):
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
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
        for v,gg in zip(ALL_W,gr):
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
        return L
    return dict(get_taps=get_taps,relax_full=relax_full,weight_step=weight_step,relax_mono=relax_mono,
                dec_img=dec_img,dec_txt=dec_txt,latents=latents,warmup_step=warmup_step)

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
P,c=build(WMUL,SEED); P_init={k:v.numpy().copy() for k,v in P.items()}; ops=make_ops(P,c)
NP=int(sum(int(np.prod(v.shape)) for v in P.values()))
def reset(): [P[k].assign(P_init[k]) for k in P]

# ============================ AUTO-BATCH FALLBACK ============================
OOM_ERRS=(tf.errors.ResourceExhaustedError, tf.errors.InternalError)
def pick_batch(requested):
    if not AUTOBATCH: return requested
    B=requested
    while B>=1:
        try:
            bi=tr_idx[:min(B,NTR)]
            x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi])
            igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
            it,tt=ops["get_taps"](x,tk)
            Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
            ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,tf.constant(0.0,tf.float32))  # lr=0: no weight change
            print(f"[autobatch] BATCHJ={B} fits (trial step ok)",flush=True)
            return B
        except OOM_ERRS as e:
            print(f"[autobatch] BATCHJ={B} OOM ({type(e).__name__}), halving",flush=True)
            B//=2
    raise RuntimeError("no batch size fits, even BATCHJ=1")
BATCHJ_REQ=BATCHJ; BATCHJ=pick_batch(BATCHJ)
steps_per_epoch = max(1, math.ceil(NTR/BATCHJ)); JOINT_STEPS = EPOCHS*steps_per_epoch
WARM_EPOCHS_EQ = math.ceil(WARMUP*BATCH/max(1,NTR))
LONG_STEPS = (EPOCHS+WARM_EPOCHS_EQ)*steps_per_epoch
print(f"=== CAPACITY === smoke={SMOKE} arms={ARMS} | wmul={WMUL} params={NP/1e6:.1f}M ({NP:,}) | "
      f"train={NTR} eval={NEV} V={V} | BATCHJ={BATCHJ} (requested {BATCHJ_REQ}) EPOCHS={EPOCHS} "
      f"({JOINT_STEPS} steps) lr={LR} | ckpt_every={CKPT_EVERY} resume={RESUME} | chance={1/max(NEV,1):.5f} bar >3/{NEV}",flush=True)

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

# ============================ CHECKPOINT / RESUME (arm A only) ============================
def state_path(arm): return os.path.join(CKPT, f"cap_state_{arm}_w{WMUL}_seed{SEED}.npz")

def save_state(arm, step, order, ptr, rs, Fhist):
    st=rs.get_state()                                                        # ('MT19937', keys, pos, has_gauss, cached)
    payload={f"W__{k}": P[k].numpy() for k in P}
    payload.update(__step=np.int64(step), __order=order.astype("int64"), __ptr=np.int64(ptr),
                   __rng_keys=st[1].astype("uint32"), __rng_pos=np.int64(st[2]),
                   __rng_hg=np.int64(st[3]), __rng_cg=np.float64(st[4]),
                   __fhist=np.asarray(Fhist[-200:], "float64"))
    tmp=state_path(arm)+".tmp.npz"
    np.savez(tmp, **payload); os.replace(tmp, state_path(arm))
    print(f"    [ckpt] saved state at step {step} ({os.path.getsize(state_path(arm))/2**30:.1f} GB)",flush=True)

def load_state(arm, rs):
    p=state_path(arm)
    if RESUME in ("0","no") or not os.path.exists(p): return None
    z=np.load(p)
    for k in P: P[k].assign(z[f"W__{k}"])
    rs.set_state(("MT19937", z["__rng_keys"], int(z["__rng_pos"]), int(z["__rng_hg"]), float(z["__rng_cg"])))
    step=int(z["__step"]); order=z["__order"].astype("int64").copy(); ptr=int(z["__ptr"])
    fh=list(z["__fhist"])
    print(f"    [ckpt] RESUMED arm {arm} from step {step} (data order preserved; bit-exact continuation "
          f"not guaranteed under GPU nondeterminism)",flush=True)
    return step, order, ptr, fh

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
        if (s+1)%LOG_EVERY==0: print(f"    [joint] {s+1:5d}/{total_steps} F={F:.4e} move={movement(P,P_init)*100:.1f}% lr={cur:.1e} t={(time.time()-t0)/60:.1f}m",flush=True)
        if CKPT_EVERY and arm=="A" and (s+1)%CKPT_EVERY==0 and (s+1)<total_steps:
            save_state(arm, s+1, order, ptr, ep_rs, Fhist)
    return Fhist, diverged

def run_arm(name, do_warmup, joint_steps, jointw):
    print(f"\n----- ARM {name} (warmup={'yes' if do_warmup else 'no'}, joint_steps={joint_steps}, jointw={jointw}) -----",flush=True)
    reset(); t0=time.time()
    postwarm=None
    if do_warmup:
        warmup_phase(WARMUP)
        if NEV:
            postwarm=latent_readout(ev_idx)
            print(f"  ARM {name} POST-WARMUP held-out: align_cos={postwarm['align_cos']:.3f} lat_retr={postwarm['lat_retr']:.3f}",flush=True)
    Fhist,diverged=joint_phase(name, joint_steps, jointw)
    move=movement(P,P_init); elapsed=(time.time()-t0)/60
    try: np.savez(os.path.join(CKPT,f"cap_{name}_w{WMUL}_seed{SEED}.npz"), **{k:P[k].numpy() for k in P})
    except Exception as e: print(f"    !! ckpt save failed: {e}",flush=True)
    try: peak=tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: peak=None
    if diverged: return dict(name=name,diverged=True,move=move,elapsed=elapsed,postwarm=postwarm,peak_gpu_gb=peak)
    tr_sub = tr_idx if NTR<=READTRAIN else tr_idx[np.random.RandomState(SEED+3).choice(NTR,READTRAIN,replace=False)]
    m_tr,_=readouts(tr_sub); m_ev,_=(readouts(ev_idx) if NEV else (None,None))
    print(f"  ARM {name}: move={move*100:.1f}% | HELD-OUT lat {m_ev['lat_hits']}/{NEV} "
          f"({sigma_above_chance(m_ev['lat_hits'],NEV):+.1f} sigma, bar >3) align={m_ev['align_cos']:.3f} "
          f"unif_img/txt={m_ev['unif_img']:.2f}/{m_ev['unif_txt']:.2f} | gen retr={m_ev['retr']:.5f} "
          f"({m_ev['hits']}/{NEV}) diversity={m_ev['diversity']:.3f} recon={m_ev['recon']:.4f} "
          f"(base {m_ev['recon_base']:.4f}) i2t={m_ev['i2t']:.3f} | {elapsed:.1f} min",flush=True)
    return dict(name=name,diverged=False,move=move,elapsed=elapsed,train=m_tr,heldout=m_ev,postwarm=postwarm,
                peak_gpu_gb=peak,lat_hits=m_ev["lat_hits"],sigma=sigma_above_chance(m_ev["lat_hits"],NEV))

# ============================ RUN ARMS ============================
results={}
if "A" in ARMS:      results["arm_A"]=run_arm("A", False, JOINT_STEPS, 0.0)
if "B" in ARMS:      results["arm_B"]=run_arm("B", True,  JOINT_STEPS, JOINTW)
if "A_long" in ARMS: results["arm_A_long"]=run_arm("A_long", False, LONG_STEPS, 0.0)

dump=dict(config=dict(smoke=SMOKE,arms=ARMS,wmul=WMUL,params=NP,N_have=N_HAVE,N_train=NTR,N_eval=NEV,
                      RES=RES,CAPLEN=CAPLEN,V=V,lr=LR,batchj=BATCHJ,batchj_requested=BATCHJ_REQ,
                      epochs=EPOCHS,joint_steps=JOINT_STEPS,ramp=RAMP,jointw=JOINTW,seed=SEED,
                      n_infer=N_INFER,ckpt_every=CKPT_EVERY),
          **results, i2t_base_train=i2t_base_on(tr_idx), i2t_base_eval=(i2t_base_on(ev_idx) if NEV else None))
out=os.path.join(HERE,f"coupling_capacity_w{WMUL}_seed{SEED}.json")
with open(out+".tmp","w") as fh: json.dump(dump,fh,indent=2)
os.replace(out+".tmp",out)
print(f"\nsaved: {out}",flush=True)

# ============================ PER-RUNG VERDICT (pre-registered branches) ============================
print(f"\n==================== CAPACITY RUNG VERDICT (wmul={WMUL}, {NP/1e9:.2f}B params, bar >3/{NEV}) ====================",flush=True)
a=results.get("arm_A")
if a is None: print("VERDICT: no arm A in this run.",flush=True)
elif a["diverged"]: print("VERDICT: DIVERGED. Report with trace; this rung is a stability datum, not a coupling datum.",flush=True)
elif a["move"]<MOVE_MIN: print(f"VERDICT: VOID (move {a['move']*100:.0f}% < {MOVE_MIN*100:.0f}% floor). Undertrained, not a negative.",flush=True)
else:
    ho=a["heldout"]
    if a["lat_hits"]>3:
        print(f"VERDICT: BRANCH (b) CANDIDATE AT THIS RUNG. PC held-out latent {a['lat_hits']}/{NEV} crosses the bar "
              f"at {NP/1e9:.2f}B. Do NOT write the word emerges: queue seeds 1,2 and a 20k rung at this size first.",flush=True)
    else:
        print(f"VERDICT: PC flat at this rung ({a['lat_hits']}/{NEV}, {a['sigma']:+.1f} sigma) with move {a['move']*100:.0f}%, "
              f"align {ho['align_cos']:.3f}, unif {ho['unif_img']:.2f}/{ho['unif_txt']:.2f}. Branch (a) row; the BP "
              f"baseline at this WMUL adjudicates the pair.",flush=True)
if SMOKE: print("\n[SMOKE] mechanics only. Confirms arm selection, autobatch trial, ckpt save/resume path, "
                "uniformity readouts, json, verdict.",flush=True)
