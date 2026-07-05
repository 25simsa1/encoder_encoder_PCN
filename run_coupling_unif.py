"""F_unif -- the constructive test implied by the diagnosis: add the missing repulsive term to the PC
energy and ask whether the PC rule can consume it. Scratch variant of run_coupling_scale.py (the
original is untouched). The paper's mechanism claim is that F fails because its cross term is pure
alignment with no repulsion (shown three ways: the PC arms, pinned BPonF under backprop, and deeper
relaxation driving alignment higher). This script tests the implied repair.

THE ENERGY. For each example i in the batch, with z_i the L2-normalized latent code:
    F_unif(S_i) = F(S_i) + UNIF_LAMBDA * u_i
    u_i = log mean_{j!=i} exp(-UNIF_T * ||z_i - z_j||^2)          (Wang-Isola uniformity, t=2)
z COMES FROM code_of(S), NOT latents(). Reason: during relaxation only S moves; the encoder taps that
latents() reads are constants, so a uniformity on latents() would have ZERO gradient into the relax
loop and the whole question ("can the relaxation consume a repulsive term") would be vacuous.
code_of(S) is the differentiable function of the relaxing state, and it is the exact NS*CODE=64-dim
code both decoders read, so spreading it is spreading the generative bottleneck. The E4-COMPARABLE
METRIC is still recorded on the encoder-concat pathway (see MEASURE) so the bands line up.

STOP-GRADIENT STRUCTURE. During relaxation the OTHER examples' codes z_j are constants
(tf.stop_gradient on the second operand of the pairwise similarity), so each example descends its own
per-example energy given the others' current codes (Jacobi-style simultaneous relaxation; z_j refresh
between relax steps as S_j moves). In the weight step the same energy is evaluated at the relaxed
(detached) states with NO stop-gradient on z_j: standard minibatch semantics, gradients to all weights.
Note the weight-path of the u term is through proj{k} only (S is detached there); the encoders feel the
repulsion indirectly, through the cross term chasing the repulsion-spread S. That indirection IS the
mechanism under test: inference moves states, learning chases them.

THE ONE DEVIATION. Strict batch-invariance of the relaxation no longer holds: the negatives z_j come
from the batch, so an example's relaxation depends on which batch it lands in. BATCHJ stays fixed at
128 for every arm so the negative-pool size is at least consistent across arms. Everything else keeps
the banked semantics (relax reduce_sum, weight reduce_mean, LARS, ramp, split law, readouts).

ARMS (one arm+seed per process; UNIF_ARM=pc|bp|both, both only sensible in smoke):
  pc  PC-unif: the PC rule (relax-then-step with LARS) on F_unif. Seeds 0,1,2. LR=RUNS1_LR (2e-2;
      one pre-registered retry at 5e-3 if unstable).
  bp  BP-unif: backprop (Adam, UNIF_BPLR=1e-4) through the unrolled N_INFER-step relaxation on F_unif,
      the free-latent pattern from run_BPonF_freelatent.py. Seed 0 only. The ceiling arm: it asks
      whether F_unif's optimum even contains transferable coupling.

MEASURE (byte-matched to the banked 8k runs, N_TRAIN=8000 N_EVAL=2000 150ep):
  PRIMARY  held-out latent retrieval, raw hits vs the pre-registered bar >3/2000.
  MECHANISM held-out uniformity on the E4 surface: unif_img/unif_txt = log mean exp(-2 d^2) over the
  L2-normed concatenated ENCODER latents (function byte-copied from analysis_latent_geometry.py,
  t=2.0 FIXED for comparability even if UNIF_T differs). E4 bands measured at 8k: F-family
  unif_img/txt in [-0.52, -0.01] (PC_armA -0.20/-0.01, PC_armB -0.07/-0.01, BPonF -0.10/-0.02,
  BPonF_free -0.52/-0.36); InfoNCE systems -3.81/-3.78 and -3.75/-3.69, and those transfer.
  PRE-REGISTERED "optimized the term": mean(unif_img, unif_txt) held-out < -1.0 (clearly out of the
  F-family band, toward the InfoNCE band). Also recorded: unif on the relaxed code (the surface the
  term acts on), train-side unif, u trajectory during training, alignment, generation secondary.

PRE-REGISTERED DECISION RULES (adjudicated over the merged records; first true branch fires):
  1 REPAIR WORKS          PC-unif held-out lat hits > 3/2000 on >=2 of 3 seeds.
  2 NECESSARY NOT SUFFICIENT  PC-unif optimizes the term (mean held-out encoder unif < -1.0 on >=2
    seeds) but retrieval stays at chance.
  3 RULE CLAUSE, SECOND INSTANCE  PC-unif fails to move uniformity (stays in the F-family band) while
    BP-unif succeeds (< -1.0): the local rule cannot consume repulsive terms either.
  4 REPAIR REFUTED        BP-unif also fails to transfer (hits <= 3): F_unif's optimum lacks coupling.
  Divergence: report with the trace; one LR retry at 5e-3 for the PC arm, no tuning past that.
  Selection among configurations uses TRAIN-side criteria only (term optimized, stability); held-out
  retrieval is evaluated once per final configuration and is never a selection criterion.

SMOKE CHECKS (RUNS1_SMOKE=1, CPU): (a) stop-gradient probe prints that z_j receives no relaxation
gradient (off-example grad rows exactly zero); (b) UNIF_LAMBDA=0 gates the term out entirely so the
graph is identical to run_coupling_scale.py arm A and the F trace must match within float noise;
(c) per-term gradient-into-S balance printed at init; (d) both arms end-to-end.

ENV: RUNS1_* exactly as run_coupling_scale.py, plus UNIF_LAMBDA(1.0) UNIF_T(2.0) UNIF_ARM(pc; both in
smoke) UNIF_BPLR(1e-4) UNIF_EVAL_EVERY(100, bp train tracker).
OUT: appends one record per (arm, lambda, seed) to coupling_unif_results.json (atomic append-merge);
weights to RUNS1_CKPT/unif_{arm}_l{lambda}_seed{s}.npz.
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
BATCHJ = int(os.environ.get("RUNS1_BATCHJ", 2 if SMOKE else 128))        # FIXED across arms (negative pool size)
EPOCHS = int(os.environ.get("RUNS1_EPOCHS", 4 if SMOKE else 150))
RAMP   = int(os.environ.get("RUNS1_RAMP", 2 if SMOKE else 300))
READB  = int(os.environ.get("RUNS1_READB", 3 if SMOKE else 128))
CKPT   = os.environ.get("RUNS1_CKPT", "/tmp/unif_ckpt" if SMOKE else "/root")
DATA   = os.environ.get("RUNS1_DATA", "/tmp/unif_data" if SMOKE else "/root/coco_scale")
COCO   = os.environ.get("RUNS1_COCO", "val2017" if SMOKE else "train2017")
READTRAIN = int(os.environ.get("RUNS1_READTRAIN", 4 if SMOKE else 1500))
LAMBDA = float(os.environ.get("UNIF_LAMBDA", 1.0))
UNIF_T = float(os.environ.get("UNIF_T", 2.0))
ARM    = os.environ.get("UNIF_ARM", "both" if SMOKE else "pc")
BPLR   = float(os.environ.get("UNIF_BPLR", 1e-4))
EVAL_EVERY = int(os.environ.get("UNIF_EVAL_EVERY", 4 if SMOKE else 100))
os.makedirs(CKPT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"
assert ARM in ("pc", "bp", "both"), "UNIF_ARM must be pc|bp|both"
assert BATCHJ >= 2, "uniformity needs at least one negative per example"

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
perm = np.random.RandomState(SEED+1).permutation(N_HAVE)                   # same split law as every banked run
tr_idx = perm[:N_TRAIN]; ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]
imgs = imgs_all; caps = caps_all
NTR, NEV = len(tr_idx), len(ev_idx)
PIX = RES*RES*3; CH = 3
chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)
toks = encode_caps(caps, c2i, CAPLEN); toks_oh = tf.one_hot(toks, V).numpy().astype("float32")
train_mean_img = imgs[tr_idx].mean(0)
steps_per_epoch = max(1, math.ceil(NTR/BATCHJ)); JOINT_STEPS = EPOCHS*steps_per_epoch
print(f"=== F_UNIF === smoke={SMOKE} arm={ARM} lambda={LAMBDA} t={UNIF_T} | N_have={N_HAVE} -> train={NTR} eval={NEV} | "
      f"img {imgs.shape[1:]} | V={V} | BATCHJ={BATCHJ} EPOCHS={EPOCHS} ({JOINT_STEPS} steps) | pc lr={LR} bp lr={BPLR} | "
      f"chance eval={1/max(NEV,1):.5f} bar >3/{NEV}",flush=True)

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
    def F_energy(S,it,tt,igt,tgt,red):                                    # the banked energy, unchanged
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        return 0.5*red(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
    def l2rows(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def unif_vec(S, stop_negatives):
        # per-example Wang-Isola term u_i = log mean_{j!=i} exp(-t ||z_i - z_j||^2) on z = l2n(code_of(S)).
        # stop_negatives=True freezes the z_j operand (relaxation semantics); False leaves all rows live
        # (weight-step / unrolled-outer semantics).
        Zl = l2rows(code_of(S))                                           # [B, NS*CODE]
        Zc = tf.stop_gradient(Zl) if stop_negatives else Zl
        d2 = tf.maximum(2.0 - 2.0*tf.matmul(Zl, Zc, transpose_b=True), 0.0)
        B  = tf.shape(Zl)[0]; Bf = tf.cast(B, tf.float32)
        off = 1.0 - tf.eye(B)
        E  = tf.exp(-UNIF_T*d2)*off
        return tf.math.log(tf.reduce_sum(E,axis=1)/tf.maximum(Bf-1.0,1.0) + 1e-30)  # [B]
    def F_relax_obj(S,it,tt,igt,tgt):
        # relaxation objective: reduce_sum F (per-example) + lambda * sum(u). Gated so LAMBDA=0 is
        # graph-identical to the banked relax_full (the equivalence smoke check depends on this).
        f = F_energy(S,it,tt,igt,tgt,tf.reduce_sum)
        if LAMBDA != 0.0: f = f + LAMBDA*tf.reduce_sum(unif_vec(S, stop_negatives=True))
        return f
    @tf.function
    def get_taps(x,tk): return enc_img(x),enc_txt(tk)
    def relax_full(S,it,tt,igt,tgt,n):                                    # banked relax_full on F_unif
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tp: tp.watch(Sv); f=F_relax_obj(Sv,it,tt,igt,tgt)
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    @tf.function
    def weight_step(x,tk,S,igt,tgt,lr):
        # banked LARS step on the same energy at the relaxed (detached) states; u with all rows live.
        with tf.GradientTape() as t:
            t.watch(ALL_W)
            Fb=F_energy(S,enc_img(x),enc_txt(tk),igt,tgt,tf.reduce_mean)
            um=tf.reduce_mean(unif_vec(S, stop_negatives=False)) if LAMBDA != 0.0 else tf.constant(0.0)
            F=Fb + LAMBDA*um
        gr=t.gradient(F,ALL_W)
        for v,gg in zip(ALL_W,gr):
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
        return Fb, um, tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    def relax_unrolled(S,it,tt,igt,tgt,n):
        # BP-unif inner dynamics: the SAME Jacobi relaxation (stop-grad negatives) kept on the outer tape.
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tpi:
                tpi.watch(Sv); f=F_relax_obj(Sv,it,tt,igt,tgt)
            gri=tpi.gradient(f,Sv)
            Sv=[Sv[k]-betas[k]*gri[k] for k in range(NS)]
        return Sv
    def relax_mono(S,taps,decfn,tgt,n):                                   # readout-only, byte-matched
        Sv=[tf.identity(s) for s in S]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_sum(tf.add_n([mse(Sv[k]-taps[k]) for k in range(NS)])+A_GEN*mse(decfn(Sv)-tgt))
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    def l2n(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))
    return dict(get_taps=get_taps,relax_full=relax_full,weight_step=weight_step,relax_unrolled=relax_unrolled,
                relax_mono=relax_mono,dec_img=dec_img,dec_txt=dec_txt,latents=latents,unif_vec=unif_vec,
                code_of=code_of,enc_img=enc_img,enc_txt=enc_txt,F_energy=F_energy,l2rows=l2rows,betas=betas)

def movement(P,P0):
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

# E4 uniformity metric, byte-copied from analysis_latent_geometry.py (t=2.0 FIXED for band comparability)
def unif_np(Z, t=2.0, cap=2000, rng=None):
    rng = rng or np.random.RandomState(1)
    if len(Z) > cap: Z = Z[rng.choice(len(Z), cap, replace=False)]
    d2 = np.maximum(2.0 - 2.0*(Z @ Z.T), 0.0)
    iu = np.triu_indices(len(Z), 1)
    return float(np.log(np.mean(np.exp(-t * d2[iu])) + 1e-30))

# ============================ BUILD ONCE, snapshot init ============================
P,c=build(WMUL,SEED); P_init={k:v.numpy().copy() for k,v in P.items()}; ops=make_ops(P,c)
NP=int(sum(int(np.prod(v.shape)) for v in P.values()))
print(f"model: {NP/1e6:.1f}M params | DM={c['DM']} DIMS={c['DIMS']}",flush=True)
def reset(): [P[k].assign(P_init[k]) for k in P]

# ============================ PROBES (eager, before any training) ============================
def probes():
    nb=min(3,NTR); bi=tr_idx[:nb]
    x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi])
    igt=tf.constant(imgs[bi].reshape(nb,-1)); tgt=tf.constant(toks_oh[bi].reshape(nb,-1))
    it,tt=ops["get_taps"](x,tk)
    Sv=[0.5*(it[k]+tt[k]) for k in range(NS)]
    # (a) stop-gradient structure: grad of example 0's u_0 wrt S must be zero on rows 1..nb-1
    with tf.GradientTape() as tp:
        tp.watch(Sv); u=ops["unif_vec"](Sv, stop_negatives=True); u0=u[0]
    gr=tp.gradient(u0,Sv)
    own=float(sum(float(tf.norm(g[0])) for g in gr))
    other=float(sum(float(tf.norm(g[1:])) for g in gr))
    print(f"[probe a] stop-gradient: grad(u_0) wrt S -- own-row norm {own:.3e}, other-row norm {other:.3e} "
          f"(z_j receives {'NO' if other==0.0 else 'A NONZERO (BUG)'} relaxation gradient)",flush=True)
    assert other == 0.0, "stop-gradient structure broken: negatives received relaxation gradient"
    # (b) per-term gradient-into-S balance at init (visibility, not asserted)
    def gnorm(obj_fn):
        with tf.GradientTape() as t2:
            t2.watch(Sv); f=obj_fn()
        g=t2.gradient(f,Sv); return [0.0 if gk is None else float(tf.norm(gk)) for gk in g]
    g_cross=gnorm(lambda: 0.5*tf.reduce_sum(A_CROSS*tf.add_n([mse(Sv[k]-it[k])+mse(Sv[k]-tt[k]) for k in range(NS)])))
    g_gen  =gnorm(lambda: 0.5*tf.reduce_sum(A_GEN*(mse(ops["dec_img"](Sv)-igt)+mse(ops["dec_txt"](Sv)-tgt))))
    g_unif =gnorm(lambda: LAMBDA*tf.reduce_sum(ops["unif_vec"](Sv, stop_negatives=True)))
    print(f"[probe b] grad-into-S at init, per scale: cross={['%.2e'%v for v in g_cross]} "
          f"gen={['%.2e'%v for v in g_gen]} lambda*unif={['%.2e'%v for v in g_unif]}",flush=True)
probes()

# ============================ READOUTS (byte-matched + uniformity surfaces) ============================
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
    # E4 surface: uniformity of the encoder concats (t=2.0 fixed, the band-comparable numbers)
    unif_img=unif_np(ZI); unif_txt=unif_np(ZT)
    # the surface the term acts on: uniformity of the code at the training operating point (relax_full
    # from the tap average with true targets, N_INFER steps), computed batchwise at the training BATCHJ
    # so the negative pool matches training
    Zcodes=[]
    for st in range(0,M,BATCHJ):
        bi=[int(idx[j]) for j in range(st,min(st+BATCHJ,M))]
        if len(bi)<2: continue
        x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); it,tt=ops["get_taps"](x,tk)
        igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        Zcodes.append(ops["l2rows"](ops["code_of"](Sv)).numpy())
    unif_code=unif_np(np.concatenate(Zcodes,0)) if Zcodes else None
    return dict(M=M,diversity=diversity,out_range=out_range,retr=retr,hits=int(round(retr*M)),chance=1.0/M,
                recon=recon,recon_base=recon_base,i2t=i2t,align_cos=align_cos,
                lat_retr=lat_hits/max(M,1),lat_hits=lat_hits,
                unif_img=unif_img,unif_txt=unif_txt,unif_code=unif_code), t2i

mode_char=int(np.bincount(toks[tr_idx].reshape(-1),minlength=V).argmax())
def i2t_base_on(idx): return float(np.mean(toks[idx]==mode_char))

def latent_readout(idx):                                                  # cheap tracker (forward only)
    M=len(idx); ZIl=[]; ZTl=[]
    for st in range(0,M,READB):
        bi=idx[st:st+READB]; ZI,ZT=ops["latents"](tf.constant(imgs[bi]), tf.constant(toks[bi])); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
    ZI=np.concatenate(ZIl,0); ZT=np.concatenate(ZTl,0)
    return dict(align_cos=float(np.mean(np.sum(ZI*ZT,1))), lat_retr=float(np.mean(np.argmax(ZT@ZI.T,1)==np.arange(M))))

def sigma_above_chance(hits, M):
    p=1.0/M; expd=M*p; sd=math.sqrt(M*p*(1-p))
    return (hits-expd)/sd if sd>0 else float("nan")

# ============================ ARM RUNNERS ============================
ep_rs=np.random.RandomState(SEED+7)                                        # same batch-order law as the banked driver

def run_pc():
    print(f"\n----- ARM PC-unif (LARS relax-then-step on F_unif, lambda={LAMBDA}, lr={LR}) -----",flush=True)
    reset(); t0=time.time(); Fhist=[]; Uhist=[]; diverged=False
    order=ep_rs.permutation(NTR); ptr=0
    for s in range(JOINT_STEPS):
        if ptr+BATCHJ>NTR: order=ep_rs.permutation(NTR); ptr=0
        bi=tr_idx[order[ptr:ptr+BATCHJ]]; ptr+=BATCHJ
        cur=LR*min(1.0,(s+1)/RAMP) if RAMP>0 else LR; lrt=tf.constant(cur,tf.float32)
        x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
        it,tt=ops["get_taps"](x,tk)
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        Fb,um,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt)
        Fb=float(Fb); um=float(um); mxw=float(mxw); Fhist.append(Fb); Uhist.append(um)
        if not (np.isfinite(Fb) and np.isfinite(um) and mxw<DIVERGE_W):
            diverged=True; print(f"    !! DIVERGENCE step {s}: F={Fb:.3e} u={um:.3e} max|w|={mxw:.2e}",flush=True); break
        if (s+1)%LOG_EVERY==0:
            print(f"    [pc] {s+1:5d}/{JOINT_STEPS} F={Fb:.4e} u={um:+.4f} move={movement(P,P_init)*100:.1f}% lr={cur:.1e} t={(time.time()-t0)/60:.1f}m",flush=True)
    move=movement(P,P_init); wall=(time.time()-t0)/60
    try: np.savez(os.path.join(CKPT,f"unif_pc_l{LAMBDA}_seed{SEED}.npz"), **{k:P[k].numpy() for k in P})
    except Exception: pass
    return finish("pc", LR, diverged, move, wall, Fhist, Uhist)

def run_bp():
    print(f"\n----- ARM BP-unif (Adam lr={BPLR} through the unrolled {N_INFER}-step relaxation on F_unif, lambda={LAMBDA}) -----",flush=True)
    reset(); t0=time.time(); Fhist=[]; Uhist=[]; diverged=False; best_tr=0.0
    names=list(P.keys()); ALL_W=list(P.values())
    # grad-coverage probe through the unroll (free-latent pattern): every tensor must receive grads
    nb=min(2,NTR); bi0=tr_idx[:nb]
    xb=tf.constant(imgs[bi0]); tkb=tf.constant(toks[bi0])
    igt0=tf.constant(imgs[bi0].reshape(nb,-1)); tgt0=tf.constant(toks_oh[bi0].reshape(nb,-1))
    with tf.GradientTape() as tpr:
        tpr.watch(ALL_W)
        it0,tt0=ops["enc_img"](xb),ops["enc_txt"](tkb)
        Sv0=ops["relax_unrolled"]([0.5*(it0[k]+tt0[k]) for k in range(NS)],it0,tt0,igt0,tgt0,N_INFER)
        F0=ops["F_energy"](Sv0,it0,tt0,igt0,tgt0,tf.reduce_mean)
        if LAMBDA != 0.0: F0=F0+LAMBDA*tf.reduce_mean(ops["unif_vec"](Sv0, stop_negatives=False))
    g0=tpr.gradient(F0,ALL_W)
    got=[k for k,gg in zip(names,g0) if gg is not None]
    print(f"  [probe c] grad coverage through the unroll: {len(got)}/{len(ALL_W)} tensors",flush=True)
    assert len(got)==len(ALL_W), f"expected every tensor to train through the unroll, got {len(got)}/{len(ALL_W)}"
    TV=ALL_W
    M_=[tf.Variable(tf.zeros_like(v),trainable=False) for v in TV]
    Vv=[tf.Variable(tf.zeros_like(v),trainable=False) for v in TV]
    B1,B2,EPS=0.9,0.999,1e-8
    @tf.function
    def bp_step(x,tk,igt,tgt,lr,tstep):
        with tf.GradientTape() as tp:
            tp.watch(TV)
            it,tt=ops["enc_img"](x),ops["enc_txt"](tk)
            Sv=ops["relax_unrolled"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
            Fb=ops["F_energy"](Sv,it,tt,igt,tgt,tf.reduce_mean)
            um=tf.reduce_mean(ops["unif_vec"](Sv, stop_negatives=False)) if LAMBDA != 0.0 else tf.constant(0.0)
            F=Fb+LAMBDA*um
        gr=tp.gradient(F,TV)
        for v,gg,m,s2 in zip(TV,gr,M_,Vv):
            gg=tf.convert_to_tensor(gg)
            m.assign(B1*m+(1-B1)*gg); s2.assign(B2*s2+(1-B2)*tf.square(gg))
            v.assign_sub(lr*(m/(1-tf.pow(B1,tstep)))/(tf.sqrt(s2/(1-tf.pow(B2,tstep)))+EPS))
        return Fb,um,tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in TV]))
    tr_sub_l=tr_idx if NTR<=READTRAIN else tr_idx[np.random.RandomState(SEED+3).choice(NTR,READTRAIN,replace=False)]
    order=ep_rs.permutation(NTR); ptr=0
    for s in range(1,JOINT_STEPS+1):
        if ptr+BATCHJ>NTR: order=ep_rs.permutation(NTR); ptr=0
        bi=tr_idx[order[ptr:ptr+BATCHJ]]; ptr+=BATCHJ
        cur=BPLR*min(1.0,s/RAMP) if RAMP>0 else BPLR
        Fb,um,mxw=bp_step(tf.constant(imgs[bi]),tf.constant(toks[bi]),
                          tf.constant(imgs[bi].reshape(len(bi),-1)),tf.constant(toks_oh[bi].reshape(len(bi),-1)),
                          tf.constant(cur,tf.float32),tf.constant(float(s),tf.float32))
        Fb=float(Fb); um=float(um); mxw=float(mxw); Fhist.append(Fb); Uhist.append(um)
        if not (np.isfinite(Fb) and np.isfinite(um) and mxw<DIVERGE_W):
            diverged=True; print(f"    !! DIVERGENCE step {s}: F={Fb:.3e} u={um:.3e} max|w|={mxw:.2e}",flush=True); break
        if s%LOG_EVERY==0:
            print(f"    [bp] {s:5d}/{JOINT_STEPS} F={Fb:.4e} u={um:+.4f} move={movement(P,P_init)*100:.1f}% lr={cur:.1e} t={(time.time()-t0)/60:.1f}m",flush=True)
        if s%EVAL_EVERY==0 or s==JOINT_STEPS:
            m_tr=latent_readout(tr_sub_l); best_tr=max(best_tr,m_tr["lat_retr"])
            print(f"    [bp] {s:5d}/{JOINT_STEPS} train lat_retr={m_tr['lat_retr']:.3f} align={m_tr['align_cos']:.3f} (best {best_tr:.3f})",flush=True)
    move=movement(P,P_init); wall=(time.time()-t0)/60
    try: np.savez(os.path.join(CKPT,f"unif_bp_l{LAMBDA}_seed{SEED}.npz"), **{k:P[k].numpy() for k in P})
    except Exception: pass
    rec=finish("bp", BPLR, diverged, move, wall, Fhist, Uhist)
    rec["train_lat_retr_best"]=best_tr
    return rec

def finish(arm, lr, diverged, move, wall, Fhist, Uhist):
    try: peak=tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: peak=None
    m_tr=m_ev=None
    if not diverged:
        tr_sub=tr_idx if NTR<=READTRAIN else tr_idx[np.random.RandomState(SEED+3).choice(NTR,READTRAIN,replace=False)]
        m_tr,_=readouts(tr_sub)
        m_ev,_=readouts(ev_idx) if NEV else (None,None)
    lat_hits=m_ev["lat_hits"] if m_ev else 0
    rec=dict(arm=arm,lam=LAMBDA,unif_t=UNIF_T,seed=SEED,n_train=NTR,n_eval=NEV,coco=("smoke" if SMOKE else COCO),
             params=NP,lr=lr,batchj=BATCHJ,epochs=EPOCHS,ramp=RAMP,n_infer=N_INFER,diverged=diverged,move=move,
             F_first=(Fhist[0] if Fhist else None),F_final=(Fhist[-1] if Fhist else None),
             u_first=(Uhist[0] if Uhist else None),u_final=(Uhist[-1] if Uhist else None),
             train=m_tr,heldout=m_ev,hits=lat_hits,chance=(1.0/NEV if NEV else None),
             sigma=(sigma_above_chance(lat_hits,NEV) if (m_ev and NEV) else None),
             i2t_base_train=i2t_base_on(tr_idx),i2t_base_eval=(i2t_base_on(ev_idx) if NEV else None),
             wall_time_min=wall,peak_gpu_gb=peak)
    if diverged:
        print(f"  ARM {arm}: DIVERGED (move={move*100:.1f}%)",flush=True); return rec
    print(f"  ARM {arm}: move={move*100:.1f}% | u {rec['u_first']:+.4f} -> {rec['u_final']:+.4f} | "
          f"HELD-OUT lat {lat_hits}/{NEV} (chance {1.0/NEV:.5f}, {rec['sigma']:+.1f} sigma, bar >3) "
          f"align={m_ev['align_cos']:.3f} unif_img/txt={m_ev['unif_img']:.2f}/{m_ev['unif_txt']:.2f} "
          f"unif_code={m_ev['unif_code'] if m_ev['unif_code'] is None else '%.2f'%m_ev['unif_code']} | "
          f"gen retr={m_ev['retr']:.5f} ({m_ev['hits']}/{NEV}) diversity={m_ev['diversity']:.3f} "
          f"recon={m_ev['recon']:.4f} (base {m_ev['recon_base']:.4f}) i2t={m_ev['i2t']:.3f} | {wall:.1f} min",flush=True)
    return rec

# ============================ RUN ============================
records=[]
if ARM in ("pc","both"): records.append(run_pc())
if ARM in ("bp","both"): records.append(run_bp())

out_path=os.path.join(HERE,"coupling_unif_results.json")
try:
    existing=json.load(open(out_path)); assert isinstance(existing.get("records"),list)
except Exception:
    existing={"records":[]}
existing["records"].extend(records)
tmp=out_path+".tmp"
with open(tmp,"w") as fh: json.dump(existing,fh,indent=2)
os.replace(tmp,out_path)
print(f"\nsaved: coupling_unif_results.json (+{len(records)}, total {len(existing['records'])})",flush=True)

# ============================ PRE-REGISTERED VERDICT (over every merged record at this lambda/t) ============================
def adjudicate(recs):
    recs=[r for r in recs if r.get("lam")==LAMBDA and r.get("unif_t")==UNIF_T and r.get("n_train")==NTR and not r.get("diverged")]
    pc=[r for r in recs if r["arm"]=="pc"]; bp=[r for r in recs if r["arm"]=="bp"]
    seen=set(); pc_u=[]
    for r in reversed(pc):                                                 # newest record per seed wins
        if r["seed"] not in seen: seen.add(r["seed"]); pc_u.append(r)
    pc=sorted(pc_u,key=lambda r:r["seed"]); bp=bp[-1:] if bp else []
    def opt(r): return (r["heldout"]["unif_img"]+r["heldout"]["unif_txt"])/2.0 < -1.0
    print(f"\n==================== F_UNIF VERDICT (lambda={LAMBDA}, t={UNIF_T}, n_train={NTR}, bar >3/{NEV}) ====================",flush=True)
    for r in pc+bp:
        ho=r["heldout"]
        print(f"  {r['arm']} seed {r['seed']}: hits {r['hits']}/{r['n_eval']} ({r['sigma']:+.1f} sigma) | "
              f"unif_img/txt {ho['unif_img']:.2f}/{ho['unif_txt']:.2f} (mean {(ho['unif_img']+ho['unif_txt'])/2:.2f}, "
              f"optimized(<-1.0)={'YES' if opt(r) else 'no'}) | unif_code "
              f"{'n/a' if ho['unif_code'] is None else '%.2f'%ho['unif_code']} | align {ho['align_cos']:.3f} | "
              f"u {r['u_first']:+.3f} -> {r['u_final']:+.3f} | move {r['move']*100:.0f}%",flush=True)
    crossed=[r for r in pc if r["hits"]>3]
    pc_opt=[r for r in pc if opt(r)]
    bp_opt=[r for r in bp if opt(r)]
    bp_crossed=[r for r in bp if r["hits"]>3]
    r1=len(crossed)>=2
    r2=(not r1) and len(pc_opt)>=2
    r3=(not r1) and (not r2) and len(pc)>=1 and len(pc_opt)==0 and len(bp_opt)>=1
    r4=len(bp)>=1 and len(bp_crossed)==0
    print(f"  rules: [1 repair works]={r1} (crossed {len(crossed)}/{len(pc)} pc seeds) | "
          f"[2 necessary-not-sufficient]={r2} (optimized {len(pc_opt)}/{len(pc)}) | "
          f"[3 rule-clause 2nd instance]={r3} (bp optimized {len(bp_opt)}/{len(bp)}) | "
          f"[4 repair refuted]={r4} (bp crossed {len(bp_crossed)}/{len(bp)})",flush=True)
    if len(pc)<3 or len(bp)<1:
        print(f"  NOTE: adjudication over PARTIAL records ({len(pc)}/3 pc seeds, {len(bp)}/1 bp). Final verdict needs all runs.",flush=True)
    if r1:   print("VERDICT BRANCH 1: THE REPAIR WORKS. Held-out latent retrieval crosses the bar on >=2 of 3 PC-unif seeds. "
                   "Diagnosis upgrades to diagnosis plus repair; queue the 20k and 2k rungs for the scale story.",flush=True)
    elif r2: print("VERDICT BRANCH 2: NECESSARY BUT NOT SUFFICIENT. PC-unif drives held-out uniformity out of the F-family "
                   "band toward the InfoNCE band, but retrieval stays at chance. The objective clause sharpens; no rescope.",flush=True)
    elif r3: print("VERDICT BRANCH 3: SECOND INDEPENDENT INSTANCE OF THE RULE CLAUSE. The local PC rule cannot consume the "
                   "repulsive term (uniformity stays in the F-family band) while backprop through the same energy can.",flush=True)
    elif r4: print("VERDICT BRANCH 4: THE REPAIR HYPOTHESIS IS REFUTED. Even backprop through the unrolled relaxation on "
                   "F_unif does not transfer; F_unif's optimum lacks coupling. The paper keeps its shape; this closes the "
                   "future-work line.",flush=True)
    else:    print("VERDICT: BETWEEN BRANCHES with the records present (see rule truth values above); rerun once the "
                   "remaining arms land.",flush=True)
adjudicate(existing["records"])
if SMOKE: print("\n[SMOKE] mechanics-only; numbers meaningless. Confirms probes (stop-grad, grad balance, unroll "
                "coverage), both arms end-to-end, uniformity readouts on both surfaces, append-merge json, verdict. "
                "The LAMBDA=0 equivalence check runs separately against run_coupling_scale.py.",flush=True)
