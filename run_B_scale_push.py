"""CONFERENCE-PATH Run B-prime -- plain-LARS scale push: how big can the model get and still GENERATE
from scratch, and WHERE (if anywhere) does from-scratch start to fail?

Run A redirected the plan: no size up to 330M cleanly undertrained, and PLAIN LARS + the transferred
LR (2e-2, tuned on 5.5M) got 330M to "useful" generation from scratch (retrieval 0.391, 24x chance) in
1500 steps. muP UNDER-moves at scale, so it is NOT the lever -- plain LARS is. Before building any
InfoNCE warm-up, test the simplest hypothesis: does plain LARS + the transferred LR let LARGER models
(past 330M) generate from scratch, and is there a clear failure scale? That failure scale is where a
warm-up would matter; if from-scratch never fails in the affordable range, a warm-up may be unnecessary.

Recipe (validated; run_A_scale_mup.py / dissociation.py): single energy F, GELU, LARS + bias trust
floor, relax-then-step, dense multi-scale shared-latent anchors (L3), A_GEN>=A_cross (L4), all grads
via tf.GradientTape. **PLAIN LARS / standard parameterization** -- uniform LR, fan_in init, tiny DEC_SD
decoder init (NO muP per-layer LR scaling: Run A showed muP under-moves at scale). Data MNIST, N=64
distinct images each with a distinct random caption, chance retrieval 1/64 = 0.016.

What it does on a GPU: trains the SAME architecture from scratch at INCREASING sizes past 330M
(~330M / 700M / 1.5B / 3B; scale width/channels/d_model; whatever fits 80GB), each at lr=2e-2 plain
LARS for a FIXED 1500-step budget (comparable to Run A Part 1). Per size it reports weight-movement %,
text->image retrieval, diversity, max|w|, diverged?, band, params, PEAK GPU MEM, wall-time. It then
identifies the scale (if any) where plain-LARS from-scratch starts to FAIL (movement below the
~22-48% collapse-break band and/or retrieval toward chance, or divergence/OOM).

Robustness: each size is wrapped so a divergence OR an OOM stops THAT size and the sweep CONTINUES
(never kills the whole run); local-disk checkpoints (/root); clean table + JSON + blunt verdict.

PARAMETERIZED via env (pod run is set-and-go, no code edits):
  RUNB_SIZES  width multipliers, comma list (default "7.8,11.4,16.6,23.5" ~ 330M/700M/1.5B/3B; printed)
  RUNB_LR     learning rate (default 2e-2, the Run-A transferred LR)
  RUNB_STEPS  steps per size (default 1500, == Run A Part 1)
  RUNB_N, RUNB_CKPT (local dir, default /root), RUNB_SEED, RUNB_SMOKE (1 = tiny CPU mechanics check)
"""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNB_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE  = os.environ.get("RUNB_SMOKE", "0") == "1"
SEED   = int(os.environ.get("RUNB_SEED", 0))
LR     = float(os.environ.get("RUNB_LR", 2e-2))
STEPS  = int(os.environ.get("RUNB_STEPS", 15 if SMOKE else 1500))
SIZES  = [float(x) for x in os.environ.get("RUNB_SIZES", ("0.15,0.3" if SMOKE else "7.8,11.4,16.6,23.5")).split(",")]
N      = int(os.environ.get("RUNB_N", 16 if SMOKE else 64))
CKPT   = os.environ.get("RUNB_CKPT", "/tmp/runB_ckpt" if SMOKE else "/root")
os.makedirs(CKPT, exist_ok=True)

# structural constants (depth/structure fixed; only WIDTH scales) -- identical to Run A
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
MOVE_COLLAPSE_LO, MOVE_USEFUL = 0.22, 0.48           # dissociation bands: collapse breaks ~22-48%, useful >48%
DIVERGE_W = 1e3
RETR_FAIL_MULT = 5.0                                 # retrieval "toward chance" if < RETR_FAIL_MULT/N (Run A threshold)
B_DM, B_C1, B_C2, B_C3, B_BN = 128, 16, 32, 64, 256
B_DIMS = [768, 768, 512, 512]
B_FFN  = 256

def cfg(wmul):
    r = lambda x: max(4, int(round(x * wmul)))
    DM = r(B_DM); DM -= DM % HEADS
    return dict(DM=max(HEADS, DM), C1=r(B_C1), C2=r(B_C2), C3=r(B_C3), BN=r(B_BN),
                DIMS=[r(d) for d in B_DIMS], FFN=r(B_FFN), HEAD=max(1, (max(HEADS, DM)) // HEADS))

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

(XTR, YTR), _ = tf.keras.datasets.mnist.load_data()
def make_data(seed):
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(XTR))[:N]
    imgs = (XTR[idx].astype("float32") / 255.0)[..., None]
    toks = np.random.RandomState(seed + 1000).randint(0, V, size=(N, T)).astype("int32")
    return imgs, YTR[idx], toks, tf.one_hot(toks, V).numpy().astype("float32")

def build(wmul, seed):
    """Standard parameterization (plain LARS): fan_in init everywhere, tiny DEC_SD for proj/decoders."""
    c = cfg(wmul); DM, C1, C2, C3, BN, DIMS, FFN = c["DM"], c["C1"], c["C2"], c["C3"], c["BN"], c["DIMS"], c["FFN"]
    f0d, f1d, f2d = 14*14*C1, 7*7*C2, 4*4*C3
    g = tf.random.Generator.from_seed(seed)
    def W(shape, key=""):
        sd = DEC_SD if (key.startswith("proj") or key in ("W_DI", "W_DT")) else 1.0 / np.sqrt(np.prod(shape[:-1]))
        return tf.Variable(g.normal(shape, stddev=sd))
    def Z(shape): return tf.Variable(tf.zeros(shape))
    P = dict(c1=W([3,3,CH,C1]),cb1=Z([C1]),c2=W([3,3,C1,C1]),cb2=Z([C1]),c3=W([3,3,C1,C2]),cb3=Z([C2]),
             c4=W([3,3,C2,C2]),cb4=Z([C2]),c5=W([3,3,C2,C3]),cb5=Z([C3]),wbn=W([f2d,BN]),bbn=Z([BN]),
             Wi0=W([f0d,DIMS[0]]),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]]),bi1=Z([DIMS[1]]),
             Wi2=W([f2d,DIMS[2]]),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]]),bi3=Z([DIMS[3]]),emb=W([V,DM]),pos=W([T,DM]))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM]);P[f"Wk{b}"]=W([DM,DM]);P[f"Wv{b}"]=W([DM,DM]);P[f"Wo{b}"]=W([DM,DM])
        P[f"f1_{b}"]=W([DM,FFN]);P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM]);P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]]);P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],f"proj{k}")
    P["W_DI"]=W([NS*CODE,PIX],"W_DI");P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,T*V],"W_DT");P["B_DT"]=Z([T*V])
    return P, c

def make_ops(P, c):
    DM, C1, C2, C3, BN, DIMS, FFN, HEAD = c["DM"], c["C1"], c["C2"], c["C3"], c["BN"], c["DIMS"], c["FFN"], c["HEAD"]
    betas = [REL_C * d for d in DIMS]; ALL_W = list(P.values())
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
    def weight_step(x,tk,S,igt,tgt,lr):                      # PLAIN LARS + bias trust floor (uniform LR)
        with tf.GradientTape() as t: t.watch(ALL_W); F=F_full(S,enc_img(x),enc_txt(tk),igt,tgt)
        gr=t.gradient(F,ALL_W)
        for v,gg in zip(ALL_W,gr):
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
        return F, tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    def relax_mono(S,taps,decfn,tgt,n):
        Sv=[tf.identity(s) for s in S]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_mean(tf.add_n([mse(Sv[k]-taps[k]) for k in range(NS)])+A_GEN*mse(decfn(Sv)-tgt))
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    return dict(get_taps=get_taps,relax_full=relax_full,weight_step=weight_step,relax_mono=relax_mono,dec_img=dec_img,dec_txt=dec_txt)

def movement(P, P0):
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

def peak_gpu_gb():
    try: return tf.config.experimental.get_memory_info("GPU:0")["peak"]/1e9
    except Exception: return float("nan")
def reset_gpu_peak():
    try: tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception: pass

def band(move, retr, diverged):
    if diverged: return "diverged"
    if not np.isfinite(move): return "diverged"
    if move < MOVE_COLLAPSE_LO: return "undertrain"
    if move < MOVE_USEFUL: return "collapse-break"
    return "useful"

def train_and_eval(wmul, lr, steps, seed):
    imgs, labels, toks, toks_oh = make_data(seed)
    DATA_STD=float(np.std(imgs[...,0]))
    reset_gpu_peak()
    P,c=build(wmul,seed); P0={k:tf.identity(v) for k,v in P.items()}; ops=make_ops(P,c)
    nparams=int(sum(int(np.prod(v.shape)) for v in P.values()))
    order=np.random.RandomState(seed+7).permutation(N)
    img_t=lambda i: tf.constant(imgs[i].reshape(1,-1)); txt_t=lambda i: tf.constant(toks_oh[i].reshape(1,-1))
    F0=Fend=None; diverged=False; mxw_last=0.0; lrt=tf.constant(lr,tf.float32); t0=time.time()
    for s in range(steps):
        i=int(order[s%N]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None]); igt=img_t(i); tgt=txt_t(i)
        it,tt=ops["get_taps"](x,tk)
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt); F=float(F); mxw=float(mxw); mxw_last=mxw
        if F0 is None: F0=F
        Fend=F
        if not (np.isfinite(F) and mxw < DIVERGE_W):
            diverged=True; print(f"    !! [wmul={wmul}] DIVERGENCE step {s}: F={F:.3e} max|w|={mxw:.2e} -> stop this size, continue sweep",flush=True); break
    move=movement(P,P0)
    t2i=np.zeros((N,HW,HW)); retr=diversity=float("nan")
    if not diverged:
        for j in range(N):
            x=tf.constant(imgs[j][None]); tk=tf.constant(toks[j][None]); it,tt=ops["get_taps"](x,tk)
            St=ops["relax_mono"](tt,tt,ops["dec_txt"],txt_t(j),GEN_INFER)
            t2i[j]=ops["dec_img"](St).numpy().reshape(HW,HW)
        diversity=float(np.mean(np.std(t2i,0))/(DATA_STD+1e-9))
        d=((t2i[:,None]-imgs[...,0][None])**2).reshape(N,N,-1).mean(-1); retr=float(np.mean(np.argmin(d,1)==np.arange(N)))
    peak=peak_gpu_gb()
    tc=time.time(); ckpt_path=os.path.join(CKPT,f"runB_w{wmul}.npz")
    try: np.savez(ckpt_path, **{k:P[k].numpy() for k in P}); ckdt=time.time()-tc
    except Exception as e: ckpt_path=f"FAILED({str(e)[:40]})"; ckdt=-1
    return dict(wmul=wmul,params=nparams,DM=c["DM"],DIMS=c["DIMS"],lr=lr,steps=steps,move=move,retr=retr,diversity=diversity,
                max_w=mxw_last,diverged=diverged,band=band(move,retr,diverged),peak_gb=peak,secs=time.time()-t0,ckpt=ckpt_path,ckpt_secs=ckdt,chance=1.0/N)

# ---- run the scale push ----
print(f"=== Run B-prime: plain-LARS scale push === seed={SEED} N={N} chance={1/N:.3f} lr={LR} steps={STEPS} smoke={SMOKE}",flush=True)
print(f"recipe: GELU, PLAIN LARS + bias floor, relax({N_INFER})-then-step, dense anchors, A_GEN={A_GEN}>=A_CROSS={A_CROSS}, standard param (NO muP)",flush=True)
print(f"sizes (width muls) = {SIZES}  [params ~ 5.54M * wmul^2]",flush=True)
for w in SIZES:
    P,_=build(w,SEED); np_=int(sum(int(np.prod(v.shape)) for v in P.values())); c=cfg(w)
    print(f"  wmul={w}: DM={c['DM']} C=({c['C1']},{c['C2']},{c['C3']}) DIMS={c['DIMS']} -> {np_/1e6:.1f}M params",flush=True); del P

results=[]
for w in SIZES:
    print(f"\n----- size wmul={w} (lr={LR}, {STEPS} steps, plain LARS) -----",flush=True)
    try:
        r=train_and_eval(w,LR,STEPS,SEED)
    except tf.errors.ResourceExhaustedError as e:
        r=dict(wmul=w,params=int(5.54e6*w*w),band="OOM",diverged=False,oom=True,move=float("nan"),retr=float("nan"),
                diversity=float("nan"),max_w=float("nan"),peak_gb=peak_gpu_gb(),secs=float("nan"),error=str(e)[:120])
        print(f"    !! [wmul={w}] OOM -> does NOT fit 80GB; stop this size, continue sweep",flush=True)
    except Exception as e:
        r=dict(wmul=w,params=int(5.54e6*w*w),band="ERROR",diverged=False,move=float("nan"),retr=float("nan"),
                diversity=float("nan"),max_w=float("nan"),peak_gb=float("nan"),secs=float("nan"),error=str(e)[:160])
        print(f"    !! [wmul={w}] ERROR: {str(e)[:120]} -> continue sweep",flush=True)
    results.append(r)
    pk = (f"{r['peak_gb']:.1f}GB" if np.isfinite(r.get('peak_gb',float('nan'))) else "n/a")
    sc = (f"{r['secs']:.0f}s" if np.isfinite(r.get('secs',float('nan'))) else "n/a")
    print(f"  => {r['params']/1e6:.0f}M  move={r['move']*100 if np.isfinite(r['move']) else float('nan'):.1f}%  retr={r['retr']:.3f}  div={r['diversity']:.3f}  max|w|={r.get('max_w',float('nan')):.2e}  band={r['band']}  peak={pk}  ({sc})",flush=True)

# ---- failure scale: first size where plain-LARS from-scratch fails ----
def failed(r):
    if r["band"] in ("diverged","OOM","ERROR"): return True
    if not np.isfinite(r["move"]) or not np.isfinite(r["retr"]): return True
    return (r["move"] < MOVE_COLLAPSE_LO) or (r["retr"] < RETR_FAIL_MULT/N)     # below collapse-break band OR retrieval toward chance
fail_sizes=[r for r in results if failed(r)]
fail_scale=min((r["params"] for r in fail_sizes), default=None)

with open(os.path.join(HERE,"runB_results.json"),"w") as fh:
    json.dump({"config":dict(seed=SEED,N=N,lr=LR,steps=STEPS,sizes=SIZES,smoke=SMOKE,chance=1/N),
               "sizes":[{k:(v if not isinstance(v,float) or np.isfinite(v) else None) for k,v in r.items()} for r in results],
               "failure_scale_params":fail_scale}, fh, indent=2)

print(f"\n==================== RUN B-prime SUMMARY (plain LARS, lr={LR}, {STEPS} steps) ====================",flush=True)
print(f"  chance retrieval = {1/N:.3f} | useful band move>={MOVE_USEFUL*100:.0f}%, collapse-break {MOVE_COLLAPSE_LO*100:.0f}-{MOVE_USEFUL*100:.0f}%, fail<{MOVE_COLLAPSE_LO*100:.0f}% or retr<{RETR_FAIL_MULT/N:.3f}",flush=True)
print(f"  {'params':>9} {'move%':>7} {'retr':>7} {'div':>6} {'max|w|':>9} {'peak':>8} {'band':>14}",flush=True)
for r in results:
    mv = f"{r['move']*100:.1f}" if np.isfinite(r['move']) else "  -"
    rt = f"{r['retr']:.3f}" if np.isfinite(r['retr']) else "  -"
    dv = f"{r['diversity']:.3f}" if np.isfinite(r['diversity']) else "  -"
    mw = f"{r.get('max_w',float('nan')):.2e}" if np.isfinite(r.get('max_w',float('nan'))) else "  -"
    pk = f"{r['peak_gb']:.1f}GB" if np.isfinite(r.get('peak_gb',float('nan'))) else "n/a"
    print(f"  {r['params']/1e6:>7.0f}M {mv:>7} {rt:>7} {dv:>6} {mw:>9} {pk:>8} {r['band']:>14}",flush=True)

fits=[r for r in results if r["band"] not in ("OOM","ERROR")]
largest_fit=max((r["params"] for r in fits), default=None)
print("\n==================== VERDICT ====================",flush=True)
if largest_fit: print(f"largest size that FIT + ran in 80GB: {largest_fit/1e6:.0f}M params",flush=True)
if fail_scale is None:
    print(f"plain LARS KEEPS GENERATING across the whole swept range -- NO from-scratch failure scale found up to {largest_fit/1e6 if largest_fit else 0:.0f}M. "
          f"A warm-up may be unnecessary in this range; push larger to find the limit.",flush=True)
else:
    fr=next(r for r in results if r["params"]==fail_scale)
    print(f"plain-LARS from-scratch FAILS starting at ~{fail_scale/1e6:.0f}M params (band={fr['band']}, move={fr['move']*100 if np.isfinite(fr['move']) else float('nan'):.1f}%, retr={fr['retr']:.3f}). "
          f"THAT is the warm-up-relevant scale.",flush=True)
print(f"saved: runB_results.json | checkpoints in {CKPT}",flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only (tiny widths/steps); numbers are not meaningful -- only confirms the code, shapes, OOM/divergence guards, checkpoint, table, and verdict logic run end-to-end.",flush=True)
