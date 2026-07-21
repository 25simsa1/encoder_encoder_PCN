"""Confirmatory multi-seed run for the headline GELU result before it goes in the paper.

The activation grid showed GELU @ aggressive lr (2e-2) generates recognizably (retrieval ~0.70 at 2500
steps, ONE seed). This replicates it: GELU and leaky-ReLU, lr=2e-2, full 4000 steps, 3 INDEPENDENT seeds
each (seed drives weight init, the MNIST subset, the captions, and the train order -- fully independent
replicates). Reports per-seed and aggregate (mean +/- std): weight-movement %, text-encoder tap std
(alive?), text->image diversity + retrieval (vs chance 0.016), image->image recon, image->text. Saves a
sample grid for the best GELU seed. Matched 4000-step budget answers: is GELU genuinely more
sample-efficient, or does leaky catch up given the steps?
"""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
N, HW, T, V = 64, 28, 8, 32
DM, HEADS, NBLK = 256, 4, 4; HEAD = DM // HEADS; FFN = 512
DIMS = [4096, 4096, 2048, 2048]; NS = len(DIMS); CODE = 16; DEC_SD = 1e-3
A_CROSS, A_GEN = 1.0, 2.0; REL_C, N_INFER = 0.05, 8; betas = [REL_C * d for d in DIMS]; PIX = HW * HW
STEPS = int(os.environ.get("SD_STEPS", 4000)); GEN_INFER = 25; LR = 2e-2
SEEDS = [int(s) for s in os.environ.get("SD_SEEDS", "0,1,2").split(",")]
ACTS = os.environ.get("SD_ACTS", "gelu,leaky").split(",")
C1, C2, C3 = 32, 64, 128; f0d, f1d, f2d, BN = 14*14*C1, 7*7*C2, 4*4*C3, 512
(XTR, YTR), _ = tf.keras.datasets.mnist.load_data()

def make_act(name):
    return {"relu": lambda z: tf.nn.relu(z), "leaky": lambda z: tf.nn.leaky_relu(z, 0.01),
            "gelu": lambda z: tf.nn.gelu(z)}[name]
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

def make_data(seed):
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(XTR))[:N]
    imgs = (XTR[idx].astype("float32") / 255.0)[..., None]
    toks = np.random.RandomState(seed + 1000).randint(0, V, size=(N, T)).astype("int32")
    return imgs, toks, tf.one_hot(toks, V).numpy().astype("float32")

def build(seed):
    g = tf.random.Generator.from_seed(seed)
    def W(s, sd=None): return tf.Variable(g.normal(s, stddev=(1.0/np.sqrt(np.prod(s[:-1])) if sd is None else sd)))
    def Z(s): return tf.Variable(tf.zeros(s))
    P = dict(c1=W([3,3,1,C1]),cb1=Z([C1]),c2=W([3,3,C1,C1]),cb2=Z([C1]),c3=W([3,3,C1,C2]),cb3=Z([C2]),
             c4=W([3,3,C2,C2]),cb4=Z([C2]),c5=W([3,3,C2,C3]),cb5=Z([C3]),wbn=W([f2d,BN]),bbn=Z([BN]),
             Wi0=W([f0d,DIMS[0]]),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]]),bi1=Z([DIMS[1]]),
             Wi2=W([f2d,DIMS[2]]),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]]),bi3=Z([DIMS[3]]),emb=W([V,DM]),pos=W([T,DM]))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM]);P[f"Wk{b}"]=W([DM,DM]);P[f"Wv{b}"]=W([DM,DM]);P[f"Wo{b}"]=W([DM,DM])
        P[f"f1_{b}"]=W([DM,FFN]);P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM]);P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]]);P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],DEC_SD)
    P["W_DI"]=W([NS*CODE,PIX],DEC_SD);P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,T*V],DEC_SD);P["B_DT"]=Z([T*V])
    return P

def run(act_name, seed):
    imgs, toks, toks_oh = make_data(seed)
    DATA_STD = float(np.std(imgs[..., 0]))
    act = make_act(act_name); P = build(seed); P0 = {k: tf.identity(v) for k, v in P.items()}; ALL_W = list(P.values())
    def enc_img(x):
        h=act(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]);h=act(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f0=tf.reshape(h,[tf.shape(x)[0],-1])
        h=act(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]);h=act(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f1=tf.reshape(h,[tf.shape(x)[0],-1])
        h=act(tf.nn.conv2d(h,P["c5"],1,"SAME")+P["cb5"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        f2=tf.reshape(h,[tf.shape(x)[0],-1]); f3=act(f2@P["wbn"]+P["bbn"])
        return [act(f0@P["Wi0"]+P["bi0"]),act(f1@P["Wi1"]+P["bi1"]),act(f2@P["Wi2"]+P["bi2"]),act(f3@P["Wi3"]+P["bi3"])]
    def enc_txt(tk):
        B=tf.shape(tk)[0]; x=tf.gather(P["emb"],tk)+P["pos"][None]; tt=[]
        for b in range(NBLK):
            q,k_,v=x@P[f"Wq{b}"],x@P[f"Wk{b}"],x@P[f"Wv{b}"]
            sp=lambda t: tf.transpose(tf.reshape(t,[B,T,HEADS,HEAD]),[0,2,1,3])
            a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
            ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,T,DM])
            x=x+ctx@P[f"Wo{b}"]; x=x+(act(x@P[f"f1_{b}"]+P[f"fb1_{b}"])@P[f"f2_{b}"]+P[f"fb2_{b}"])
            tt.append(act(tf.reduce_mean(x,1)@P[f"Wt{b}"]+P[f"bt{b}"]))
        return tt
    def code_of(S): return tf.concat([act(S[k]@P[f"proj{k}"]) for k in range(NS)],axis=1)
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
    def weight_step(x,tk,S,igt,tgt,lr_):
        with tf.GradientTape() as t: t.watch(ALL_W); it,tt=enc_img(x),enc_txt(tk); F=F_full(S,it,tt,igt,tgt)
        gr=t.gradient(F,ALL_W)
        for v,gg in zip(ALL_W,gr):
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr_*tr*gg)
        mxw=tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W])); return F,mxw
    def img_tgt(x): return tf.reshape(x,[tf.shape(x)[0],-1])
    def txt_tgt(o): return tf.reshape(o,[tf.shape(o)[0],-1])
    def movement():
        num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
        den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)
    t0=time.time(); order=np.random.RandomState(seed+7).permutation(N); F0=Fend=None; diverged=False
    for step in range(STEPS):
        i=int(order[step%N]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None])
        igt=img_tgt(x); tgt=txt_tgt(tf.constant(toks_oh[i][None])); it,tt=get_taps(x,tk)
        Sv=relax_full([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=weight_step(x,tk,tuple(tf.constant(s) for s in Sv),igt,tgt,tf.constant(LR,tf.float32)); F=float(F)
        if F0 is None: F0=F
        Fend=F
        if not (np.isfinite(F) and float(mxw)<1e3): diverged=True; break
    move=movement()
    def relax_text(tt,tgt,n):
        Sv=[tf.identity(tt[k]) for k in range(NS)]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_mean(tf.add_n([mse(Sv[k]-tt[k]) for k in range(NS)])+A_GEN*mse(dec_txt(Sv)-tgt))
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    def relax_image(it,igt,n):
        Sv=[tf.identity(it[k]) for k in range(NS)]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_mean(tf.add_n([mse(Sv[k]-it[k]) for k in range(NS)])+A_GEN*mse(dec_img(Sv)-igt))
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    t2i=np.zeros((N,HW,HW)); i2i=np.zeros((N,HW,HW)); i2tacc=[]; ttall=[[] for _ in range(NS)]
    for j in range(N):
        x=tf.constant(imgs[j][None]); tk=tf.constant(toks[j][None]); it,tt=get_taps(x,tk)
        for k in range(NS): ttall[k].append(tt[k].numpy().reshape(-1))
        St=relax_text(tt,txt_tgt(tf.constant(toks_oh[j][None])),GEN_INFER); t2i[j]=dec_img(St).numpy().reshape(HW,HW)
        Si=relax_image(it,img_tgt(x),GEN_INFER); i2i[j]=dec_img(Si).numpy().reshape(HW,HW)
        i2tacc.append(float(np.mean(dec_txt(Si).numpy().reshape(T,V).argmax(-1)==toks[j])))
    tap_std=float(np.mean([np.mean(np.std(np.array(ttall[k]),0)) for k in range(NS)]))
    diversity=float(np.mean(np.std(t2i,0))/(DATA_STD+1e-9))
    d=((t2i[:,None]-imgs[...,0][None])**2).reshape(N,N,-1).mean(-1); retr=float(np.mean(np.argmin(d,1)==np.arange(N)))
    recon=float(np.mean((i2i-imgs[...,0])**2)); i2t=float(np.mean(i2tacc))
    print(f"  [{act_name:5s} seed={seed}] move={move*100:6.1f}% tap_std={tap_std:.3e} {'ALIVE' if tap_std>1e-2 else 'DEAD'} | "
          f"div={diversity:.3f} retr={retr:.3f} | recon={recon:.4f} i2t={i2t:.3f} | F {F0:.2e}->{Fend:.2e} {'DIV' if diverged else ''}", flush=True)
    return dict(act=act_name,seed=seed,move=move,tap_std=tap_std,diversity=diversity,retr=retr,recon=recon,
                i2t=i2t,F0=F0,Fend=Fend,diverged=bool(diverged),t2i=t2i[:12],targets=imgs[:12,:,:,0],labels=YTR[np.random.RandomState(seed).permutation(len(XTR))[:12]])

print(f"SEED REPLICATION: lr={LR} steps={STEPS} acts={ACTS} seeds={SEEDS} | 50.5M, N={N}, chance retr={1/N:.3f}", flush=True)
res=[]; tg=time.time()
for act_name in ACTS:
    for s in SEEDS: res.append(run(act_name, s))
print(f"\nTOTAL {(time.time()-tg)/60:.1f} min", flush=True)

def agg(act, key):
    xs=[r[key] for r in res if r["act"]==act]; return float(np.mean(xs)), float(np.std(xs))
summary={}
print("\n==================== AGGREGATE (mean +/- std over seeds) ====================", flush=True)
for act in ACTS:
    a={k: agg(act,k) for k in ("move","tap_std","diversity","retr","recon","i2t")}
    summary[act]=a
    print(f"  {act:5s}: retr={a['retr'][0]:.3f}+/-{a['retr'][1]:.3f}  div={a['diversity'][0]:.3f}+/-{a['diversity'][1]:.3f}  "
          f"move={a['move'][0]*100:.0f}+/-{a['move'][1]*100:.0f}%  tap_std={a['tap_std'][0]:.2e}  recon={a['recon'][0]:.4f}  i2t={a['i2t'][0]:.3f}", flush=True)

# best GELU seed grid
gelu=[r for r in res if r["act"]=="gelu"]
if gelu:
    best=max(gelu,key=lambda r:r["retr"]); nc=12
    fig,axes=plt.subplots(2,nc,figsize=(1.1*nc,2.5))
    for jj in range(nc):
        axes[0,jj].imshow(best["targets"][jj],cmap="gray",vmin=0,vmax=1); axes[0,jj].axis("off")
        axes[1,jj].imshow(np.clip(best["t2i"][jj],0,1),cmap="gray",vmin=0,vmax=1); axes[1,jj].axis("off")
    axes[0,0].set_title(f"target",fontsize=7,loc="left"); axes[1,0].set_title(f"GELU text->image (seed {best['seed']}, retr={best['retr']:.2f})",fontsize=7,loc="left")
    plt.suptitle(f"Best GELU seed: text->image generation (retr {best['retr']:.2f} vs chance {1/N:.3f})",fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(HERE,"seeds_gelu_grid.png"),dpi=130); plt.close()

with open(os.path.join(HERE,"seeds_results.json"),"w") as fh:
    json.dump(dict(per_run=[{k:v for k,v in r.items() if k not in ("t2i","targets","labels")} for r in res], summary=summary),fh,indent=2)

# verdict
gr=summary.get("gelu",{}).get("retr",(0,0)); lr_=summary.get("leaky",{}).get("retr",(0,0))
chance=1/N
print("\n==================== VERDICT ====================", flush=True)
gretr=[r["retr"] for r in res if r["act"]=="gelu"]
print(f"GELU retrieval per seed: {[round(x,3) for x in gretr]}  -> mean {gr[0]:.3f} +/- {gr[1]:.3f} (chance {chance:.3f})", flush=True)
if gretr and min(gretr) > 5*chance:
    print(f"=> GELU REPLICATES: retrieval consistently well above chance across all {len(gretr)} seeds (min {min(gretr):.3f} = {min(gretr)/chance:.0f}x chance). Headline number: {gr[0]:.2f} +/- {gr[1]:.2f}.", flush=True)
elif gr[0] > 5*chance and gr[1] > 0.5*gr[0]:
    print(f"=> GELU generates ON AVERAGE ({gr[0]:.2f}) but is HIGH-VARIANCE across seeds (+/-{gr[1]:.2f}); report with the spread, not as a single number.", flush=True)
else:
    print(f"=> GELU does NOT robustly replicate (mean {gr[0]:.3f}); the single-seed 0.70 was not representative. Report honestly.", flush=True)
print(f"GELU vs leaky @ 4000 steps: GELU retr {gr[0]:.3f}+/-{gr[1]:.3f} vs leaky {lr_[0]:.3f}+/-{lr_[1]:.3f}  -> "
      + ("GELU more sample-efficient" if gr[0] > lr_[0]+max(gr[1],lr_[1]) else "leaky catches up / comparable"), flush=True)
print("saved: seeds_gelu_grid.png, seeds_results.json", flush=True)
