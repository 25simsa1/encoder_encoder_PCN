"""Resolve the dying-ReLU asterisk from the mid-scale run: is it an LR artifact or architectural?

Grid (everything else IDENTICAL to midscale.py's 50.5M Option-C bidirectional PCN; fresh identical init
per cell): activation {relu, leaky-relu(0.01), gelu} x LR {aggressive 2e-2, gentle 6.7e-3 (1/3), gentle
2e-3 (1/10)}. Per cell report: text-encoder tap std across captions (ALIVE vs DEAD), weight-movement %,
text->image diversity ratio + retrieval, image-side metrics (recon, image->text token acc).

VERDICTS:
  - plain ReLU survives + generates at gentler LR  -> dying-ReLU was an LR artifact, not architecture.
  - plain ReLU dies even at gentle LR              -> architectural fragility; leaky/gelu a real fix.
  - GELU (smooth, no hard zero) stays alive like leaky, unlike dead ReLU -> the collapse is a HARD-ZERO
    activation problem (ReLU dies, smooth activations survive). Clean general statement for the paper.
"""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
N, HW, T, V = 64, 28, 8, 32
DM, HEADS, NBLK = 256, 4, 4; HEAD = DM // HEADS; FFN = 512
DIMS = [4096, 4096, 2048, 2048]; NS = len(DIMS); CODE = 16; DEC_SD = 1e-3
A_CROSS, A_GEN = 1.0, 2.0; REL_C, N_INFER = 0.05, 8; betas = [REL_C * d for d in DIMS]; PIX = HW * HW
STEPS = int(os.environ.get("AG_STEPS", 2200)); GEN_INFER = 25
LR_AGG = 2e-2; LRS = [LR_AGG, LR_AGG / 3.0, LR_AGG / 10.0]
ACTS = ["relu", "leaky", "gelu"]
C1, C2, C3 = 32, 64, 128; f0d, f1d, f2d, BN = 14*14*C1, 7*7*C2, 4*4*C3, 512

(xtr, ytr), _ = tf.keras.datasets.mnist.load_data()
idx = np.random.permutation(len(xtr))[:N]
imgs = (xtr[idx].astype("float32") / 255.0)[..., None]; labels = ytr[idx]
DATA_STD = float(np.std(imgs[..., 0])); MEAN_IMG = imgs[..., 0].mean(0)
toks = np.random.RandomState(1).randint(0, V, size=(N, T)).astype("int32")
toks_oh = tf.one_hot(toks, V).numpy().astype("float32")

def make_act(name):
    if name == "relu":  return lambda z: tf.nn.relu(z)
    if name == "leaky": return lambda z: tf.nn.leaky_relu(z, 0.01)
    if name == "gelu":  return lambda z: tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

def build():
    g = tf.random.Generator.from_seed(42)                                  # identical init every cell
    def W(s, sd=None): return tf.Variable(g.normal(s, stddev=(1.0/np.sqrt(np.prod(s[:-1])) if sd is None else sd)))
    def Z(s): return tf.Variable(tf.zeros(s))
    P = dict(c1=W([3,3,1,C1]),cb1=Z([C1]),c2=W([3,3,C1,C1]),cb2=Z([C1]),c3=W([3,3,C1,C2]),cb3=Z([C2]),
             c4=W([3,3,C2,C2]),cb4=Z([C2]),c5=W([3,3,C2,C3]),cb5=Z([C3]),wbn=W([f2d,BN]),bbn=Z([BN]),
             Wi0=W([f0d,DIMS[0]]),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]]),bi1=Z([DIMS[1]]),
             Wi2=W([f2d,DIMS[2]]),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]]),bi3=Z([DIMS[3]]),
             emb=W([V,DM]),pos=W([T,DM]))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM]);P[f"Wk{b}"]=W([DM,DM]);P[f"Wv{b}"]=W([DM,DM]);P[f"Wo{b}"]=W([DM,DM])
        P[f"f1_{b}"]=W([DM,FFN]);P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM]);P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]]);P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],DEC_SD)
    P["W_DI"]=W([NS*CODE,PIX],DEC_SD);P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,T*V],DEC_SD);P["B_DT"]=Z([T*V])
    return P

def run(act_name, lr):
    act = make_act(act_name); P = build(); P0 = {k: tf.identity(v) for k, v in P.items()}
    ALL_W = list(P.values())
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
        mxw=tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
        return F,mxw
    def img_tgt(x): return tf.reshape(x,[tf.shape(x)[0],-1])
    def txt_tgt(o): return tf.reshape(o,[tf.shape(o)[0],-1])
    def movement():
        num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
        den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)
    # train
    t0=time.time(); order=np.random.permutation(N); F0=Fend=None; diverged=False
    for step in range(STEPS):
        i=int(order[step%N]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None])
        igt=img_tgt(x); tgt=txt_tgt(tf.constant(toks_oh[i][None])); it,tt=get_taps(x,tk)
        Sv=relax_full([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=weight_step(x,tk,tuple(tf.constant(s) for s in Sv),igt,tgt,tf.constant(lr,tf.float32))
        F=float(F);
        if F0 is None: F0=F
        Fend=F
        if not (np.isfinite(F) and float(mxw)<1e3): diverged=True; break
    move=movement()
    # eval
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
    tap_std=float(np.mean([np.mean(np.std(np.array(ttall[k]),0)) for k in range(NS)]))   # std across captions, mean over units+scales
    diversity=float(np.mean(np.std(t2i,0))/(DATA_STD+1e-9))
    d=((t2i[:,None]-imgs[...,0][None])**2).reshape(N,N,-1).mean(-1); retr=float(np.mean(np.argmin(d,1)==np.arange(N)))
    recon=float(np.mean((i2i-imgs[...,0])**2)); i2t=float(np.mean(i2tacc))
    mode_tok=np.bincount(toks.reshape(-1),minlength=V).argmax(); base=float(np.mean(toks==mode_tok))
    alive = tap_std > 1e-2
    print(f"  [{act_name:5s} lr={lr:.1e}] move={move*100:6.1f}% tap_std={tap_std:.3e} {'ALIVE' if alive else 'DEAD '} | "
          f"t2i div={diversity:.3f} retr={retr:.3f} | recon={recon:.4f} i2t={i2t:.3f}(base {base:.3f}) | F {F0:.2e}->{Fend:.2e} {'DIVERGED' if diverged else ''}", flush=True)
    return dict(act=act_name,lr=lr,move=move,tap_std=tap_std,alive=bool(alive),diversity=diversity,
                retr=retr,recon=recon,i2t=i2t,base=base,F0=F0,Fend=Fend,diverged=bool(diverged),t2i=t2i[:10])

print(f"ACT GRID: 50.5M, N={N}, STEPS={STEPS} | acts={ACTS} lrs={[f'{l:.1e}' for l in LRS]} | chance retr={1/N:.3f}", flush=True)
res=[]; tg=time.time()
for act_name in ACTS:
    for lr in LRS:
        res.append(run(act_name, lr))
print(f"\nTOTAL {(time.time()-tg)/60:.1f} min", flush=True)

# heatmaps: diversity + tap_std(alive) across act x lr
def cell(a,l): return next(r for r in res if r["act"]==a and abs(r["lr"]-l)<1e-12)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,(key,title,fmt) in zip(axs,[("diversity","text->image diversity",".2f"),("move","weight movement",".0%")]):
    M=np.array([[cell(a,l)[key] for l in LRS] for a in ACTS])
    im=ax.imshow(M,cmap="viridis",vmin=0); ax.set_xticks(range(3),[f"{l:.1e}" for l in LRS]); ax.set_yticks(range(3),ACTS)
    ax.set_xlabel("learning rate"); ax.set_title(title)
    for i,a in enumerate(ACTS):
        for jj,l in enumerate(LRS):
            c=cell(a,l); ax.text(jj,i,f"{c[key]:{fmt}}\n{'ALIVE' if c['alive'] else 'DEAD'}\nretr={c['retr']:.2f}",
                                 ha="center",va="center",color="white" if M[i,jj]<M.max()*0.6 else "black",fontsize=7)
    fig.colorbar(im,ax=ax,fraction=0.046)
plt.suptitle("Activation x LR: does the text encoder survive (ALIVE) and does text->image generate?",fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"actgrid_heatmap.png"),dpi=130); plt.close()

# montage: text->image strips, rows = act x {aggressive, gentle 1/10}
rows=[("relu",LR_AGG),("leaky",LR_AGG),("gelu",LR_AGG),("relu",LRS[2]),("leaky",LRS[2]),("gelu",LRS[2])]
nc=10; fig,axes=plt.subplots(len(rows)+1,nc,figsize=(1.0*nc,1.1*(len(rows)+1)))
for jj in range(nc): axes[0,jj].imshow(imgs[jj,:,:,0],cmap="gray",vmin=0,vmax=1); axes[0,jj].axis("off")
axes[0,0].set_title("target",fontsize=7,loc="left")
for ri,(a,l) in enumerate(rows):
    c=cell(a,l)
    for jj in range(nc): axes[ri+1,jj].imshow(np.clip(c["t2i"][jj],0,1),cmap="gray",vmin=0,vmax=1); axes[ri+1,jj].axis("off")
    axes[ri+1,0].set_title(f"{a} lr={l:.0e} div={c['diversity']:.2f} {'ALIVE' if c['alive'] else 'DEAD'}",fontsize=6,loc="left")
plt.suptitle("text->image by activation x LR (top=target). DEAD text encoder => identical mush.",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"actgrid_samples.png"),dpi=130); plt.close()

with open(os.path.join(HERE,"actgrid_results.json"),"w") as fh:
    json.dump([{k:v for k,v in r.items() if k!="t2i"} for r in res],fh,indent=2)

# verdict
relu_agg=cell("relu",LR_AGG); relu_g1=cell("relu",LRS[1]); relu_g2=cell("relu",LRS[2])
print("\n==================== VERDICT ====================",flush=True)
print(f"relu aggressive: {'ALIVE' if relu_agg['alive'] else 'DEAD'} (tap_std={relu_agg['tap_std']:.2e}, div={relu_agg['diversity']:.2f})",flush=True)
print(f"relu gentle 1/3: {'ALIVE' if relu_g1['alive'] else 'DEAD'} (tap_std={relu_g1['tap_std']:.2e}, div={relu_g1['diversity']:.2f}, move={relu_g1['move']*100:.0f}%)",flush=True)
print(f"relu gentle 1/10:{'ALIVE' if relu_g2['alive'] else 'DEAD'} (tap_std={relu_g2['tap_std']:.2e}, div={relu_g2['diversity']:.2f}, move={relu_g2['move']*100:.0f}%)",flush=True)
smooth_alive=all(cell(a,l)["alive"] for a in ("leaky","gelu") for l in LRS)
relu_survives_gentle = relu_g1["alive"] or relu_g2["alive"]
if relu_survives_gentle:
    print("=> dying-ReLU is an LR ARTIFACT: plain ReLU survives at gentler LR (not architectural).",flush=True)
else:
    print("=> plain ReLU DIES even at gentle LR: architectural fragility of hard-zero activation.",flush=True)
print(f"=> smooth activations (leaky+gelu) alive across ALL LRs: {smooth_alive}  (=> collapse is a HARD-ZERO-activation problem)" if smooth_alive
      else "=> smooth activations NOT uniformly alive (see grid)",flush=True)
print("saved: actgrid_heatmap.png, actgrid_samples.png, actgrid_results.json",flush=True)
