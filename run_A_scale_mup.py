"""CONFERENCE-PATH Run A -- scale sweep + muP (maximal-update parameterization) LR transfer.

GOAL (when run on a GPU): (PART 1) find the model size where training the bidirectional PCN FROM
SCRATCH starts to UNDERTRAIN in a fixed step budget; (PART 2) show muP transfers a small-model LR to a
larger model -- the same LR lands the larger model in the dissociation "useful" weight-movement band
(collapse breaks ~22-48% movement, saturates >100%) WITHOUT divergence, which standard parameterization
(SP) does not.

Built on the validated recipe (midscale_seeds.py / dissociation.py): one scalar energy F, GELU, LARS +
bias trust floor, relax-then-step, dense multi-scale shared-latent anchors (L3), A_GEN>=A_cross (L4),
ALL grads via tf.GradientTape. Data = MNIST, N=64 distinct images each with a distinct random caption
(no class-label shortcut), chance retrieval 1/64 = 0.016 -- the dissociation regime.

=========================== muP UNDER LARS -- the per-layer scaling we apply ===========================
References: arXiv:2411.02001 (Ishikawa et al., abc-parameterization; feature learning needs
DeltaW.h = Theta(1), Cond 3.1) and arXiv:2505.13124 (muPC; depth/width LR transfer for PC; inference
must also be scaled). The abc rules are optimizer-dependent. The validated recipe's optimizer is LARS,
whose update is  Delta_w = -eta * (||w|| / ||g||) * g  (with the +1e-3 bias trust floor), so
||Delta_w|| = eta*||w||  ==>  per-layer RELATIVE movement = eta, width-invariant (this is already why
LARS gives scale-stable *movement %*). But the per-coordinate FEATURE update is NOT automatically
Theta(1) under LARS, and the LARS trust ratio CANCELS the gradient magnitude, so the exponents differ
from SGD-muP and Adam-muP. Deriving DeltaW.h for a linear layer y=w^T x under LARS:
    Delta_y = Delta_w^T x = -eta (||w||/||g||) (x (x) delta)^T x = -eta (||w||*||x||/||delta||) delta
    ==> ||Delta_y|| = eta * ||w|| * ||x||   (the ||delta|| cancels -- LARS normalizes the gradient).
With fan_in init (entries ~ 1/sqrt(fan_in)) and Theta(1) activations:
  - HIDDEN (fan_in ~ n, fan_out ~ n): ||w||_F=Theta(sqrt n), ||x||=Theta(sqrt n), y is n-dim ==>
    Delta_y per-coordinate rms = eta*n/sqrt(n) = eta*sqrt(n).  Theta(1)  ==>  eta_hidden ~ m^(-1/2).
  - READOUT (fan_in ~ n, fan_out fixed): same algebra, fixed-dim output ==>  eta_out ~ m^(-1/2).
  - INPUT (fan_in fixed, fan_out ~ n): ||w||_F=Theta(sqrt n), ||x||=Theta(1), y n-dim ==>
    per-coordinate rms = eta. Theta(1)  ==>  eta_in ~ m^0 = 1.
  - WIDTH-INDEPENDENT (fan_in & fan_out fixed: the decode readouts W_DI/W_DT, all biases): eta ~ 1.
  (m = width / base_width.)  These LARS exponents (in: 1, hidden/out: m^-1/2) are DERIVED here, not
  lifted from the SGD/Adam tables; PART 2 is their empirical test -- if right, the small-model LR
  transfers; if wrong, it will not, and we will see it.

So muP here =
  (a) INIT: fan_in init (1/sqrt(fan_in)) for every width-dependent weight INCLUDING the readout proj
      (so the decode code stays Theta(1) at init across widths); width-independent decode heads keep the
      recipe's tiny DEC_SD init.
  (b) LR: per-layer LR = base_LR * mult, mult = m^(-1/2) for hidden+readout (fan_in ~ width), 1 otherwise.
  (c) INFERENCE: relaxation step betas[k] ~ DIMS[k] (already in the recipe) -- the muPC inference
      scaling that keeps Delta_S = Theta(1) as the latent width grows.
SP (standard parameterization, the comparison): the validated recipe unchanged -- fan_in init for
hidden, fixed DEC_SD for proj/decoders, and a UNIFORM LR (mult=1 for all layers). Under SP the
small-model LR does NOT transfer (feature updates grow ~ m^(1/2) with width -> forward blow-up /
divergence or out-of-band movement at the larger size).

PARAMETERIZED via env so the pod run only sets values (no code edits):
  RUNA_SIZES   width multipliers, comma list (default "1,3.2,7.8" ~ 5M/50M/300M; param count printed)
  RUNA_PARAM   "mup" | "sp"        (PART-1 sweep parameterization; PART 2 runs both)
  RUNA_LR      base learning rate
  RUNA_STEPS   training steps per size (default 1500)
  RUNA_SWEEP_LRS  comma LRs for the PART-2 small-model tune (default "5e-3,1e-2,2e-2,4e-2")
  RUNA_SWEEP_STEPS short budget for the LR tune (default 500)
  RUNA_SEED, RUNA_N, RUNA_CKPT (local dir), RUNA_SMOKE (1 = tiny CPU mechanics check)
"""
import os, sys, time, json, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNA_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE   = os.environ.get("RUNA_SMOKE", "0") == "1"
SEED    = int(os.environ.get("RUNA_SEED", 0))
PARAM   = os.environ.get("RUNA_PARAM", "mup")
LR      = float(os.environ.get("RUNA_LR", 2e-2))
STEPS   = int(os.environ.get("RUNA_STEPS", 15 if SMOKE else 1500))
SWEEP_STEPS = int(os.environ.get("RUNA_SWEEP_STEPS", 10 if SMOKE else 500))
SWEEP_LRS   = [float(x) for x in os.environ.get("RUNA_SWEEP_LRS", "5e-3,1e-2,2e-2,4e-2").split(",")]
SIZES   = [float(x) for x in os.environ.get("RUNA_SIZES", ("0.15,0.3" if SMOKE else "1,3.2,7.8")).split(",")]
BASE_WMUL = SIZES[0]                                            # smallest = muP base (LR tuned here)
N       = int(os.environ.get("RUNA_N", 16 if SMOKE else 64))
CKPT    = os.environ.get("RUNA_CKPT", "/tmp/runA_ckpt")
os.makedirs(CKPT, exist_ok=True)

# fixed structural constants (depth/structure held; only WIDTH scales)
HW, CH = 28, 1
T, V   = 8, 32
HEADS, NBLK = 4, 4
NS = 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C  = 0.05
N_INFER = 4 if SMOKE else 8
GEN_INFER = 5 if SMOKE else 25
PIX = HW * HW
MOVE_COLLAPSE_LO, MOVE_BAND_HI = 0.22, 1.00                    # dissociation: collapse breaks ~22-48%, saturates >100%
DIVERGE_W = 1e3
# base widths at wmul=1 (~5M on MNIST); everything scales linearly in wmul (params ~ wmul^2)
B_DM, B_C1, B_C2, B_C3, B_BN = 128, 16, 32, 64, 256
B_DIMS = [768, 768, 512, 512]
B_FFN  = 256

def cfg(wmul):
    r = lambda x: max(4, int(round(x * wmul)))
    DM = r(B_DM); DM -= DM % HEADS                              # keep d_model divisible by heads
    return dict(DM=max(HEADS, DM), C1=r(B_C1), C2=r(B_C2), C3=r(B_C3), BN=r(B_BN),
                DIMS=[r(d) for d in B_DIMS], FFN=r(B_FFN), HEAD=max(1, (max(HEADS, DM)) // HEADS))

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

# ---- data (MNIST, N distinct images + distinct random captions) ----
(XTR, YTR), _ = tf.keras.datasets.mnist.load_data()
def make_data(seed):
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(XTR))[:N]
    imgs = (XTR[idx].astype("float32") / 255.0)[..., None]
    labels = YTR[idx]
    toks = np.random.RandomState(seed + 1000).randint(0, V, size=(N, T)).astype("int32")
    return imgs, labels, toks, tf.one_hot(toks, V).numpy().astype("float32")

# ---- per-key layer category (for muP init + LR scaling) ----
def category(k):
    if k in ("emb", "pos", "c1", "cb1"):                        return "in"      # fan_in fixed, fan_out ~ width
    if k.startswith("proj"):                                    return "out"     # fan_in ~ width, fan_out fixed (readout)
    if k in ("W_DI", "B_DI", "W_DT", "B_DT"):                   return "fixed"   # fan_in & fan_out fixed
    if k.startswith(("b", "cb", "fb", "bi", "bt", "bbn")):      return "bias"
    return "hidden"                                                              # fan_in ~ width, fan_out ~ width

def build(wmul, mode, seed):
    """Returns (P dict, lr_mult list aligned to list(P.values())). m = wmul/BASE_WMUL."""
    c = cfg(wmul); DM, C1, C2, C3, BN, DIMS, FFN = c["DM"], c["C1"], c["C2"], c["C3"], c["BN"], c["DIMS"], c["FFN"]
    f0d, f1d, f2d = 14*14*C1, 7*7*C2, 4*4*C3
    g = tf.random.Generator.from_seed(seed)
    m = wmul / BASE_WMUL
    def W(shape, key):
        cat = category(key)
        if mode == "sp" and (key.startswith("proj") or key in ("W_DI", "W_DT")):
            sd = DEC_SD                                          # SP: recipe's fixed tiny decoder init
        elif mode == "mup" and key in ("W_DI", "W_DT"):
            sd = DEC_SD                                          # width-independent decode heads keep tiny init in both
        else:
            sd = 1.0 / np.sqrt(np.prod(shape[:-1]))              # fan_in init (muP-correct for in/hidden/out)
        return tf.Variable(g.normal(shape, stddev=sd))
    def Z(shape): return tf.Variable(tf.zeros(shape))
    P = dict(c1=W([3,3,CH,C1],"c1"),cb1=Z([C1]),c2=W([3,3,C1,C1],"c2"),cb2=Z([C1]),c3=W([3,3,C1,C2],"c3"),cb3=Z([C2]),
             c4=W([3,3,C2,C2],"c4"),cb4=Z([C2]),c5=W([3,3,C2,C3],"c5"),cb5=Z([C3]),wbn=W([f2d,BN],"wbn"),bbn=Z([BN]),
             Wi0=W([f0d,DIMS[0]],"Wi0"),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]],"Wi1"),bi1=Z([DIMS[1]]),
             Wi2=W([f2d,DIMS[2]],"Wi2"),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]],"Wi3"),bi3=Z([DIMS[3]]),
             emb=W([V,DM],"emb"),pos=W([T,DM],"pos"))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM],"Wq");P[f"Wk{b}"]=W([DM,DM],"Wk");P[f"Wv{b}"]=W([DM,DM],"Wv");P[f"Wo{b}"]=W([DM,DM],"Wo")
        P[f"f1_{b}"]=W([DM,FFN],"f1_");P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM],"f2_");P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]],"Wt");P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],f"proj{k}")
    P["W_DI"]=W([NS*CODE,PIX],"W_DI");P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,T*V],"W_DT");P["B_DT"]=Z([T*V])
    # per-layer LR multipliers
    mult = []
    for k in P:
        cat = category(k)
        if mode == "mup" and cat in ("hidden", "out"): mult.append(m ** -0.5)
        else: mult.append(1.0)                                  # in / fixed / bias  (and everything under SP)
    return P, c, mult

def make_ops(P, c, lr_mult):
    DM, C1, C2, C3, BN, DIMS, FFN, HEAD = c["DM"], c["C1"], c["C2"], c["C3"], c["BN"], c["DIMS"], c["FFN"], c["HEAD"]
    betas = [REL_C * d for d in DIMS]
    ALL_W = list(P.values()); MULT = list(lr_mult)
    def enc_img(x):
        h=gelu(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]);h=gelu(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f0=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]);h=gelu(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f1=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c5"],1,"SAME")+P["cb5"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f2=tf.reshape(h,[tf.shape(x)[0],-1]); f3=gelu(f2@P["wbn"]+P["bbn"])
        return [gelu(f0@P["Wi0"]+P["bi0"]),gelu(f1@P["Wi1"]+P["bi1"]),gelu(f2@P["Wi2"]+P["bi2"]),gelu(f3@P["Wi3"]+P["bi3"])]
    def enc_txt(tk):
        B=tf.shape(tk)[0]; x=tf.gather(P["emb"],tk)+P["pos"][None]; tt=[]
        for b in range(NBLK):
            q,k_,v=x@P[f"Wq{b}"],x@P[f"Wk{b}"],x@P[f"Wv{b}"]
            sp=lambda t: tf.transpose(tf.reshape(t,[B,T,HEADS,HEAD]),[0,2,1,3])
            a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
            ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,T,DM])
            x=x+ctx@P[f"Wo{b}"]; x=x+(gelu(x@P[f"f1_{b}"]+P[f"fb1_{b}"])@P[f"f2_{b}"]+P[f"fb2_{b}"])
            tt.append(gelu(tf.reduce_mean(x,1)@P[f"Wt{b}"]+P[f"bt{b}"]))
        return tt
    def code_of(S): return tf.concat([gelu(S[k]@P[f"proj{k}"]) for k in range(NS)],axis=1)
    def dec_img(S): return tf.nn.sigmoid(code_of(S)@P["W_DI"]+P["B_DI"])
    def dec_txt(S): return code_of(S)@P["W_DT"]+P["B_DT"]
    def F_full(S,it,tt,igt,tgt):
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        return 0.5*tf.reduce_mean(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
    @tf.function
    def get_taps(x,tk): return enc_img(x),enc_txt(tk)
    def relax_full(S,it,tt,igt,tgt,n):
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tp: tp.watch(Sv); f=F_full(Sv,it,tt,igt,tgt)
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    @tf.function
    def weight_step(x,tk,S,igt,tgt,lr):
        with tf.GradientTape() as t: t.watch(ALL_W); F=F_full(S,enc_img(x),enc_txt(tk),igt,tgt)
        gr=t.gradient(F,ALL_W)
        for v,gg,mu in zip(ALL_W,gr,MULT):                       # LARS + bias trust floor + per-layer muP LR mult
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*mu*tr*gg)
        return F, tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    def relax_mono(S,taps,decfn,tgt,n):
        Sv=[tf.identity(s) for s in S]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_mean(tf.add_n([mse(Sv[k]-taps[k]) for k in range(NS)])+A_GEN*mse(decfn(Sv)-tgt))
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    return dict(get_taps=get_taps,relax_full=relax_full,weight_step=weight_step,relax_mono=relax_mono,
                dec_img=dec_img,dec_txt=dec_txt)

def movement(P, P0):
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

def train_and_eval(wmul, mode, lr, steps, seed, tag):
    imgs, labels, toks, toks_oh = make_data(seed)
    DATA_STD=float(np.std(imgs[...,0]))
    P,c,mult=build(wmul,mode,seed); P0={k:tf.identity(v) for k,v in P.items()}; ops=make_ops(P,c,mult)
    nparams=int(sum(int(np.prod(v.shape)) for v in P.values()))
    order=np.random.RandomState(seed+7).permutation(N)
    img_t=lambda i: tf.constant(imgs[i].reshape(1,-1)); txt_t=lambda i: tf.constant(toks_oh[i].reshape(1,-1))
    F0=Fend=None; diverged=False; lrt=tf.constant(lr,tf.float32); t0=time.time()
    for s in range(steps):
        i=int(order[s%N]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None]); igt=img_t(i); tgt=txt_t(i)
        it,tt=ops["get_taps"](x,tk)
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt); F=float(F); mxw=float(mxw)
        if F0 is None: F0=F
        Fend=F
        if not (np.isfinite(F) and mxw < DIVERGE_W):
            diverged=True; print(f"    !! [{tag}] DIVERGENCE step {s}: F={F:.3e} max|w|={mxw:.2e}",flush=True); break
    move=movement(P,P0)
    # generation read-outs (text->image retrieval + diversity)
    t2i=np.zeros((N,HW,HW))
    if not diverged:
        for j in range(N):
            x=tf.constant(imgs[j][None]); tk=tf.constant(toks[j][None]); it,tt=ops["get_taps"](x,tk)
            St=ops["relax_mono"](tt,tt,ops["dec_txt"],txt_t(j),GEN_INFER)
            t2i[j]=ops["dec_img"](St).numpy().reshape(HW,HW)
        diversity=float(np.mean(np.std(t2i,0))/(DATA_STD+1e-9))
        d=((t2i[:,None]-imgs[...,0][None])**2).reshape(N,N,-1).mean(-1); retr=float(np.mean(np.argmin(d,1)==np.arange(N)))
    else:
        diversity=float("nan"); retr=float("nan")
    # local-disk checkpoint (timed)
    tc=time.time(); ckpt_path=os.path.join(CKPT,f"runA_{tag}.npz")
    try:
        np.savez(ckpt_path, **{k:P[k].numpy() for k in P}); ckdt=time.time()-tc
    except Exception as e:
        ckpt_path=f"FAILED({e})"; ckdt=-1
    return dict(tag=tag,wmul=wmul,mode=mode,lr=lr,steps=steps,params=nparams,DM=c["DM"],DIMS=c["DIMS"],
                m=wmul/BASE_WMUL,move=move,diversity=diversity,retr=retr,diverged=diverged,F0=F0,Fend=Fend,
                secs=time.time()-t0,ckpt=ckpt_path,ckpt_secs=ckdt,chance=1.0/N)

def band(move):                                                 # is movement in the dissociation "useful" band?
    if not np.isfinite(move): return "diverged"
    if move < MOVE_COLLAPSE_LO: return "undertrain"
    if move < 0.48: return "collapse-break"
    return "useful"

print(f"=== Run A: scale sweep + muP transfer === param={PARAM} seed={SEED} N={N} chance={1/N:.3f} "
      f"steps={STEPS} smoke={SMOKE}", flush=True)
print(f"sizes (width muls)={SIZES} base(muP)={BASE_WMUL} | LARS muP exponents: in=1, hidden/readout=m^-0.5, inference betas~DIMS", flush=True)
for w in SIZES:
    c=cfg(w); P,_,_=build(w,"mup",SEED); np_=int(sum(int(np.prod(v.shape)) for v in P.values()))
    print(f"  size wmul={w}: DM={c['DM']} C=({c['C1']},{c['C2']},{c['C3']}) DIMS={c['DIMS']} -> {np_/1e6:.2f}M params", flush=True)
    del P

if os.environ.get("RUNA_VERIFY"):                               # prove the muP init+LR wiring, then exit
    mexp = (SIZES[-1] / BASE_WMUL) ** -0.5
    print(f"\n--- muP WIRING CHECK (large wmul={SIZES[-1]}, m={SIZES[-1]/BASE_WMUL:.2f}; expect hidden/readout LR-mult=m^-0.5={mexp:.3f}, others=1) ---", flush=True)
    for mode in ("mup", "sp"):
        P, c, mult = build(SIZES[-1], mode, SEED); keys = list(P.keys()); mm = {k: mult[i] for i, k in enumerate(keys)}
        print(f"  [{mode}]", flush=True)
        for k in ("emb", "c1", "Wq0", "Wi0", "Wt0", "proj0", "W_DI", "bi0"):
            print(f"    {k:6} cat={category(k):6} lr_mult={mm[k]:.4f}  init_std~{float(tf.math.reduce_std(P[k])):.5f}", flush=True)
    sys.exit(0)

results={"part1":[], "part2":{}}

# ===================== PART 1: from-scratch scale sweep =====================
print(f"\n----- PART 1: from-scratch sweep ({PARAM}), {STEPS} steps, lr={LR} -----", flush=True)
for w in SIZES:
    r=train_and_eval(w,PARAM,LR,STEPS,SEED,f"p1_w{w}_{PARAM}")
    r["band"]=band(r["move"])
    results["part1"].append(r)
    print(f"  wmul={w:<5} {r['params']/1e6:6.1f}M  move={r['move']*100:6.1f}%  retr={r['retr']:.3f}  div={r['diversity']:.3f}  "
          f"{'DIVERGED' if r['diverged'] else r['band']:14}  ({r['secs']:.0f}s, ckpt {r['ckpt_secs']:.1f}s)",flush=True)

# identify the undertraining onset (movement drops below collapse-break OR retrieval toward chance)
under=[r for r in results["part1"] if (not r["diverged"]) and (r["move"]<MOVE_COLLAPSE_LO or (np.isfinite(r["retr"]) and r["retr"]<5/N))]
undersize=min((r["wmul"] for r in under), default=None)

# ===================== PART 2: muP LR transfer =====================
print(f"\n----- PART 2: muP LR transfer (tune LR on small wmul={BASE_WMUL}, transfer to large wmul={SIZES[-1]}) -----", flush=True)
small_sweep=[]
for lr in SWEEP_LRS:
    r=train_and_eval(BASE_WMUL,"mup",lr,SWEEP_STEPS,SEED,f"p2_small_lr{lr}")
    r["band"]=band(r["move"]); small_sweep.append(r)
    print(f"  [small muP] lr={lr:.0e}: move={r['move']*100:6.1f}% retr={r['retr']:.3f} {'DIV' if r['diverged'] else r['band']}",flush=True)
# pick the small-model LR in the useful band, nearest target ~0.8 movement (else largest non-diverging)
cand=[r for r in small_sweep if (not r["diverged"]) and r["move"]>=MOVE_COLLAPSE_LO]
if cand: chosen=min(cand,key=lambda r:abs(r["move"]-0.8))
else:    chosen=min((r for r in small_sweep if not r["diverged"]),key=lambda r:-r["move"],default=small_sweep[0])
tuned_lr=chosen["lr"]
print(f"  -> tuned small-model LR = {tuned_lr:.0e} (small move {chosen['move']*100:.1f}%)",flush=True)

# transfer the SAME tuned LR to the LARGE model under muP and under SP
large=SIZES[-1]
tr_mup=train_and_eval(large,"mup",tuned_lr,STEPS,SEED,f"p2_large_mup_lr{tuned_lr}"); tr_mup["band"]=band(tr_mup["move"])
tr_sp =train_and_eval(large,"sp", tuned_lr,STEPS,SEED,f"p2_large_sp_lr{tuned_lr}");  tr_sp["band"]=band(tr_sp["move"])
results["part2"]=dict(tuned_lr=tuned_lr,small=chosen,small_sweep=small_sweep,large_mup=tr_mup,large_sp=tr_sp,large_wmul=large)
print(f"  [large muP] lr={tuned_lr:.0e}: move={tr_mup['move']*100:6.1f}% retr={tr_mup['retr']:.3f} {'DIV' if tr_mup['diverged'] else tr_mup['band']}",flush=True)
print(f"  [large SP ] lr={tuned_lr:.0e}: move={tr_sp['move']*100:6.1f}% retr={tr_sp['retr']:.3f} {'DIV' if tr_sp['diverged'] else tr_sp['band']}",flush=True)

# ===================== results table + JSON + verdict =====================
def clean(r): return {k:v for k,v in r.items() if k not in ()}
with open(os.path.join(HERE,"runA_results.json"),"w") as fh:
    json.dump({"config":dict(param=PARAM,seed=SEED,N=N,steps=STEPS,sizes=SIZES,base_wmul=BASE_WMUL,lr=LR,smoke=SMOKE),
               "part1":results["part1"],
               "part2":{k:(clean(v) if isinstance(v,dict) else ([clean(x) for x in v] if isinstance(v,list) else v))
                        for k,v in results["part2"].items()}}, fh, indent=2)

print("\n==================== RUN A SUMMARY ====================",flush=True)
print(f"PART 1 from-scratch sweep ({PARAM}, {STEPS} steps, lr={LR}):",flush=True)
print(f"  {'wmul':>6} {'params':>9} {'move%':>7} {'retr':>7} {'div':>6}  band",flush=True)
for r in results["part1"]:
    print(f"  {r['wmul']:>6} {r['params']/1e6:>7.1f}M {r['move']*100:>7.1f} {r['retr']:>7.3f} {r['diversity']:>6.3f}  "
          f"{'DIVERGED' if r['diverged'] else r['band']}",flush=True)
print(f"  undertraining onset: {'wmul='+str(undersize) if undersize else 'none in swept range'} "
      f"(movement<{MOVE_COLLAPSE_LO*100:.0f}% or retrieval<{5/N:.3f})",flush=True)
print(f"\nPART 2 muP transfer (tuned LR {tuned_lr:.0e} on small wmul={BASE_WMUL} -> large wmul={large}, {tr_mup['params']/1e6:.0f}M):",flush=True)
print(f"  small muP move {chosen['move']*100:.1f}% | large muP move {tr_mup['move']*100:.1f}% ({tr_mup['band']}) | large SP move {tr_sp['move']*100:.1f}% ({tr_sp['band']})",flush=True)

# blunt verdict
mup_transfers = (not tr_mup["diverged"]) and (MOVE_COLLAPSE_LO <= tr_mup["move"])           # in/above band, finite
sp_fails      = tr_sp["diverged"] or tr_sp["move"] < MOVE_COLLAPSE_LO or tr_sp["move"] > 5.0  # diverge or out-of-band
print("\n==================== VERDICT ====================",flush=True)
print(f"muP LR transfer: {'WORKS' if mup_transfers else 'FAILS'} -- tuned small LR lands the {tr_mup['params']/1e6:.0f}M model "
      f"at {tr_mup['move']*100:.0f}% movement ({tr_mup['band']}), {'finite' if not tr_mup['diverged'] else 'DIVERGED'}.",flush=True)
print(f"muP vs SP at the same LR: SP {'FAILS to transfer' if sp_fails else 'also transfers'} "
      f"(SP large move {tr_sp['move']*100:.0f}%, {'DIVERGED' if tr_sp['diverged'] else tr_sp['band']}) "
      f"vs muP {tr_mup['move']*100:.0f}% -- {'muP is needed' if (mup_transfers and sp_fails) else 'no clear muP advantage here'}.",flush=True)
under_str = (f"wmul={undersize}" if undersize else "not reached in swept range -- go larger")
print(f"undertraining size (from-scratch, fixed budget): {under_str}",flush=True)
print(f"saved: runA_results.json  | checkpoints in {CKPT}",flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only run (tiny widths/steps); numbers are not meaningful -- this only confirms the code, muP wiring, shapes, ckpt, and verdict logic run end-to-end.",flush=True)
