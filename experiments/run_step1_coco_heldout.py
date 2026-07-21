"""Step 1 GATE -- HELD-OUT variant. Same recipe/model/optimizer/budget as run_step1_coco_gate.py
(committed 00304a9); the ONLY change is the data split + the scope of the generation readouts.

WHY: the gate's retrieval/recon/diversity were computed on the SAME 400 pairs the model trained on
(in-sample). That is fine for the gate's yes/no question ("does the 7c mode-collapse break when weights
move?") but says NOTHING about generalization, and it is the wrong basis the moment the goal shifts from
"responds to text" to "renders the (unseen) scene". This run trains on N_TRAIN pairs and evaluates
text->image retrieval / image->image recon / diversity / i2t on a DISJOINT N_EVAL set the model never saw.

We report every metric on BOTH train and held-out so the train<->eval GAP is explicit -- that gap is the
generalization story, and it quantifies exactly how much the gate's in-sample numbers were inflated.

Honesty fixes vs the gate:
  - vocab is built from TRAIN captions only; unseen eval chars map to the null token (no eval leakage
    into the embedding table).
  - recon now has a baseline: MSE of predicting the TRAIN-MEAN image against the eval images
    (recon below this baseline = genuine reconstruction; at/above it = trivial).

Everything else (GELU, plain LARS + bias floor, lr=2e-2, single energy F, dense multi-scale anchors,
A_GEN>=A_cross, relax-then-step, N_INFER/GEN_INFER, model widths, init) is byte-for-byte the gate recipe.

ENV: RUNS1_NTRAIN (default 400), RUNS1_NEVAL (default 100), plus all the gate's knobs
(RUNS1_RES/CAPLEN/WMUL/LR/STEPS/CKPT/DATA/SEED/SMOKE). chance retrieval = 1/N_EVAL (held), 1/N_TRAIN (train).
"""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNS1_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE  = os.environ.get("RUNS1_SMOKE", "0") == "1"
SEED   = int(os.environ.get("RUNS1_SEED", 0))
RES    = int(os.environ.get("RUNS1_RES", 16 if SMOKE else 64))
CAPLEN = int(os.environ.get("RUNS1_CAPLEN", 16 if SMOKE else 64))
N_TRAIN= int(os.environ.get("RUNS1_NTRAIN", 6 if SMOKE else 400))
N_EVAL = int(os.environ.get("RUNS1_NEVAL", 2 if SMOKE else 100))
N_WANT = N_TRAIN + N_EVAL
PAIRS  = int(os.environ.get("RUNS1_PAIRS", N_WANT + 300))                 # extra: some image fetches fail
WMUL   = float(os.environ.get("RUNS1_WMUL", 0.1 if SMOKE else 1.5))
LR     = float(os.environ.get("RUNS1_LR", 2e-2))
STEPS  = int(os.environ.get("RUNS1_STEPS", 12 if SMOKE else 5000))
CKPT   = os.environ.get("RUNS1_CKPT", "/tmp/s1ho_ckpt" if SMOKE else "/root")
DATA   = os.environ.get("RUNS1_DATA", "/tmp/s1ho_data" if SMOKE else "/root/coco_s1ho")
os.makedirs(CKPT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"

# recipe constants (identical to run_step1_coco_gate.py)
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER  = 2 if SMOKE else 8
GEN_INFER = 3 if SMOKE else 25
DIVERGE_W = 1e3
MOVE_MIN = 0.40
LOG_EVERY = 5 if SMOKE else 200
CKPT_EVERY = 9999 if SMOKE else 1500
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

# ============================ DATA ============================
SMOKE_CAPS = ["a dog runs across a grassy field","a red bus on a city street","two people sit on a wooden bench",
    "a plate of food on a table","a cat sleeping on a couch","a man riding a surfboard on a wave",
    "a clock tower against a blue sky","a bowl of fruit near a window"]

def build_vocab(caps):
    chars = sorted(set("".join(caps)) | {"\0"}); return chars, {c:i for i,c in enumerate(chars)}
def encode_caps(caps, c2i, caplen):
    nul = c2i["\0"]; toks = np.full((len(caps), caplen), nul, "int32")
    for n,cp in enumerate(caps):
        for t in range(caplen):
            if t < len(cp): toks[n,t] = c2i.get(cp[t], nul)        # unseen char -> null (no leakage)
    return toks

def load_synthetic():
    rs = np.random.RandomState(SEED)
    caps = [c.lower() for c in (SMOKE_CAPS * 3)[:N_WANT]]
    imgs = np.zeros((len(caps), RES, RES, 3), "float32")
    for i in range(len(caps)):
        imgs[i] = rs.rand(RES,RES,3).astype("float32")*0.3
        c = rs.rand(3); y,x = rs.randint(0,RES,2); r = RES//4
        yy,xx = np.ogrid[:RES,:RES]; m = (yy-y)**2+(xx-x)**2 <= r*r
        imgs[i][m] = c
    return imgs, caps

def load_coco():
    f_img,f_cap = os.path.join(DATA,"imgs_ho.npy"), os.path.join(DATA,"caps_ho.txt")
    if os.path.exists(f_img) and os.path.exists(f_cap):
        imgs=np.load(f_img); caps=open(f_cap).read().split("\n")[:len(imgs)]
        print(f"[data] reused cache {imgs.shape}",flush=True); return imgs, caps
    import urllib.request, zipfile
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    ANN="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"; IMG="http://images.cocodataset.org/val2017/{}"
    imgdir=os.path.join(DATA,"img"); os.makedirs(imgdir,exist_ok=True); capj=os.path.join(DATA,"cap.json"); t0=time.time()
    if not os.path.exists(capj):
        z=os.path.join(DATA,"ann.zip")
        if not os.path.exists(z): print("[data] downloading COCO annotations (~241MB)...",flush=True); urllib.request.urlretrieve(ANN,z)
        with zipfile.ZipFile(z) as zf, zf.open("annotations/captions_val2017.json") as s, open(capj,"wb") as d: d.write(s.read())
    cap=json.load(open(capj)); id2cap={}
    for a in cap["annotations"]: id2cap.setdefault(a["image_id"], a["caption"])
    id2file={im["id"]:im["file_name"] for im in cap["images"]}
    ids=[i for i in id2cap if i in id2file][:PAIRS]
    def dl(iid):
        p=os.path.join(imgdir,id2file[iid])
        if not os.path.exists(p):
            try: urllib.request.urlretrieve(IMG.format(id2file[iid]),p)
            except Exception: pass
    with ThreadPoolExecutor(max_workers=32) as ex: list(ex.map(dl,ids))
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
# deterministic disjoint split
perm = np.random.RandomState(SEED+1).permutation(N_HAVE)
tr_idx = perm[:N_TRAIN]
ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]                                # may be < N_EVAL if data short
imgs = imgs_all; caps = caps_all                                    # index by tr_idx / ev_idx everywhere
NTR, NEV = len(tr_idx), len(ev_idx)
PIX = RES*RES*3; CH = 3

# vocab from TRAIN captions only
chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)
toks = encode_caps(caps, c2i, CAPLEN)                               # encode all; eval uses train vocab
toks_oh = tf.one_hot(toks, V).numpy().astype("float32")
train_mean_img = imgs[tr_idx].mean(0)                               # recon baseline predictor
DATA_STD_TR = float(np.std(imgs[tr_idx])); DATA_STD_EV = float(np.std(imgs[ev_idx])) if NEV else float("nan")
print(f"=== Step 1 HELD-OUT GATE === smoke={SMOKE} N_have={N_HAVE} -> train={NTR} eval={NEV} (disjoint) | img {imgs.shape[1:]} [0,1] | CAPLEN={CAPLEN} V={V}(train-only) | chance retr train={1/NTR:.4f} eval={1/max(NEV,1):.4f}",flush=True)
print(f"  train cap[0]={caps[tr_idx[0]]!r}",flush=True)
if NEV: print(f"  eval  cap[0]={caps[ev_idx[0]]!r}",flush=True)

# ============================ MODEL (identical to gate) ============================
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

def movement(P,P0):
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

# ============================ TRAIN (train indices only) ============================
P,c=build(WMUL,SEED); P0={k:tf.identity(v) for k,v in P.items()}; ops=make_ops(P,c)
NP=int(sum(int(np.prod(v.shape)) for v in P.values()))
print(f"model: {NP/1e6:.1f}M params | DM={c['DM']} C=({c['C1']},{c['C2']},{c['C3']},{c['C4']}) DIMS={c['DIMS']} | plain LARS lr={LR} | budget {STEPS} steps | trains on {NTR} pairs",flush=True)
IMG_T=imgs.reshape(N_HAVE,-1).astype("float32"); TXT_T=toks_oh.reshape(N_HAVE,-1).astype("float32")
img_t=lambda i: tf.constant(IMG_T[i][None]); txt_t=lambda i: tf.constant(TXT_T[i][None])
order=np.random.RandomState(SEED+7).permutation(NTR); Fhist=[]; diverged=False; t0=time.time(); lrt=tf.constant(LR,tf.float32)
for s in range(STEPS):
    i=int(tr_idx[order[s%NTR]]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None]); igt=img_t(i); tgt=txt_t(i)
    it,tt=ops["get_taps"](x,tk)
    Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
    F,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt); F=float(F); mxw=float(mxw); Fhist.append(F)
    if not (np.isfinite(F) and mxw<DIVERGE_W):
        diverged=True; print(f"  !! DIVERGENCE step {s}: F={F:.3e} max|w|={mxw:.2e} -> STOP",flush=True); break
    if (s+1)%LOG_EVERY==0:
        mv=movement(P,P0); print(f"  step {s+1:5d} t={(time.time()-t0)/60:.1f}m F={F:.4e} move={mv*100:.1f}% max|w|={mxw:.2e}",flush=True)
    if (s+1)%CKPT_EVERY==0:
        try: np.savez(os.path.join(CKPT,"s1ho_ckpt.npz"), **{k:P[k].numpy() for k in P})
        except Exception as e: print(f"  ckpt failed: {e}",flush=True)
move=movement(P,P0); elapsed=(time.time()-t0)/60
print(f"\n[train done] steps={len(Fhist)} diverged={diverged} t={elapsed:.1f}m F {Fhist[0]:.3e}->{Fhist[-1]:.3e} | WEIGHT-MOVEMENT={move*100:.1f}%",flush=True)
try: np.savez(os.path.join(CKPT,"s1ho_ckpt.npz"), **{k:P[k].numpy() for k in P})
except Exception: pass

# ============================ READOUTS (run on a given index set) ============================
def readouts(idx):
    M=len(idx); t2i=np.zeros((M,RES,RES,CH)); i2i=np.zeros((M,RES,RES,CH)); i2t_acc=[]
    for jj in range(M):
        j=int(idx[jj]); x=tf.constant(imgs[j][None]); tk=tf.constant(toks[j][None]); it,tt=ops["get_taps"](x,tk)
        St=ops["relax_mono"]([tf.identity(tt[k]) for k in range(NS)], tt, ops["dec_txt"], txt_t(j), GEN_INFER)
        t2i[jj]=ops["dec_img"](St).numpy().reshape(RES,RES,CH)
        Si=ops["relax_mono"]([tf.identity(it[k]) for k in range(NS)], it, ops["dec_img"], img_t(j), GEN_INFER)
        i2i[jj]=ops["dec_img"](Si).numpy().reshape(RES,RES,CH)
        i2t_acc.append(float(np.mean(ops["dec_txt"](Si).numpy().reshape(CAPLEN,V).argmax(-1)==toks[j])))
    real=imgs[idx]
    diversity=float(np.mean(np.std(t2i,0))/(np.std(real)+1e-9))
    out_range=float((t2i.max(0)-t2i.min(0)).mean())
    dd=((t2i.reshape(M,1,-1)-real.reshape(1,M,-1))**2).mean(-1)
    retr=float(np.mean(np.argmin(dd,1)==np.arange(M)))
    recon=float(np.mean((i2i-real)**2))
    recon_base=float(np.mean((train_mean_img[None]-real)**2))      # trivial predictor (train mean image)
    i2t=float(np.mean(i2t_acc))
    return dict(M=M,diversity=diversity,out_range=out_range,retr=retr,chance=1.0/M,
                recon=recon,recon_base=recon_base,i2t=i2t), t2i, i2i

mode_char=int(np.bincount(toks[tr_idx].reshape(-1),minlength=V).argmax())
def i2t_base_on(idx): return float(np.mean(toks[idx]==mode_char))

if not diverged:
    m_tr,t2i_tr,i2i_tr = readouts(tr_idx)
    m_ev,t2i_ev,i2i_ev = (readouts(ev_idx) if NEV else (None,None,None))
else:
    m_tr=m_ev=None; t2i_ev=i2i_ev=None

def fmt(m,base_i2t):
    if m is None: return "  (skipped/diverged)"
    return (f"    retrieval = {m['retr']:.4f}  (chance {m['chance']:.4f}, = {m['retr']*m['M']:.0f}x)\n"
            f"    diversity = {m['diversity']:.3f}   out-range = {m['out_range']:.3e}\n"
            f"    recon MSE = {m['recon']:.4f}  (train-mean baseline {m['recon_base']:.4f}; lower=better)\n"
            f"    i2t acc   = {m['i2t']:.3f}  (mode-char baseline {base_i2t:.3f})")

print(f"\n==================== STEP 1 HELD-OUT METRICS (F is NOT the signal) ====================",flush=True)
print(f"  weight-movement = {move*100:.1f}%  (>= {MOVE_MIN*100:.0f}% required, else undertrained=VOID)",flush=True)
print(f"  --- TRAIN (in-sample, N={NTR}) ---\n{fmt(m_tr,i2t_base_on(tr_idx))}",flush=True)
print(f"  --- HELD-OUT (N={NEV}) ---\n{fmt(m_ev,i2t_base_on(ev_idx) if NEV else 0.0)}",flush=True)
if m_tr and m_ev:
    print(f"  --- GENERALIZATION GAP (train - eval) ---",flush=True)
    print(f"    retrieval {m_tr['retr']-m_ev['retr']:+.4f} | diversity {m_tr['diversity']-m_ev['diversity']:+.3f} | recon {m_tr['recon']-m_ev['recon']:+.4f}",flush=True)

# ============================ GRID (held-out examples) ============================
if not diverged and NEV:
    nc=min(8,NEV)
    fig,axes=plt.subplots(3,nc,figsize=(1.5*nc,4.8))
    for jj in range(nc):
        j=int(ev_idx[jj])
        axes[0,jj].imshow(np.clip(imgs[j],0,1)); axes[0,jj].axis("off"); axes[0,jj].set_title(caps[j][:22],fontsize=5)
        axes[1,jj].imshow(np.clip(t2i_ev[jj],0,1)); axes[1,jj].axis("off")
        axes[2,jj].imshow(np.clip(i2i_ev[jj],0,1)); axes[2,jj].axis("off")
    for r,l in [(0,"target"),(1,"text->img"),(2,"img->img")]: axes[r,0].set_ylabel(l,fontsize=8)
    plt.suptitle(f"Step1 HELD-OUT ({NP/1e6:.0f}M, move {move*100:.0f}%): UNSEEN captions. eval retr {m_ev['retr']:.3f} vs chance {m_ev['chance']:.4f}",fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(HERE,"step1_heldout_grid.png"),dpi=120); plt.close()

dump=dict(config=dict(smoke=SMOKE,N_have=N_HAVE,N_train=NTR,N_eval=NEV,RES=RES,CAPLEN=CAPLEN,V=V,wmul=WMUL,params=NP,lr=LR,steps=len(Fhist),seed=SEED),
          diverged=diverged,move=move,
          train=m_tr,heldout=m_ev,
          i2t_base_train=i2t_base_on(tr_idx),i2t_base_eval=(i2t_base_on(ev_idx) if NEV else None),
          F0=Fhist[0],Fend=Fhist[-1],elapsed_min=elapsed)
with open(os.path.join(HERE,"step1_heldout_results.json"),"w") as fh: json.dump(dump,fh,indent=2)

# ============================ VERDICT (on HELD-OUT) ============================
print(f"\n==================== STEP 1 HELD-OUT VERDICT ====================",flush=True)
if diverged:
    print("VERDICT: DIVERGED -- non-finite/blowup. Lower LR or check setup.",flush=True)
elif move < MOVE_MIN:
    print(f"VERDICT: VOID (UNDERTRAINED) -- weights moved only {move*100:.1f}% (<{MOVE_MIN*100:.0f}%).",flush=True)
elif not NEV:
    print("VERDICT: NO EVAL SET -- not enough pairs for a held-out split; rerun with more data.",flush=True)
else:
    varies = (m_ev['diversity'] >= 0.20) and (m_ev['out_range'] > 1e-2)
    above_chance = m_ev['retr'] > 3.0/NEV
    recon_beats = m_ev['recon'] < m_ev['recon_base']
    if varies and above_chance:
        print(f"VERDICT: HELD-OUT PASS -- on UNSEEN captions, text->image VARIES (diversity {m_ev['diversity']:.2f}, out-range {m_ev['out_range']:.1e}) and is ABOVE CHANCE (retr {m_ev['retr']:.3f} = {m_ev['retr']*NEV:.0f}x). recon {'beats' if recon_beats else 'does NOT beat'} the train-mean baseline. The text->image map GENERALIZES, not just memorizes.",flush=True)
    elif varies and not above_chance:
        print(f"VERDICT: HELD-OUT PARTIAL -- unseen output VARIES (diversity {m_ev['diversity']:.2f}) but retrieval not above chance ({m_ev['retr']:.3f} vs {1/NEV:.4f}): the caption->image map did NOT generalize, even though it fit the training pairs. In-sample gate was inflated by memorization.",flush=True)
    else:
        print(f"VERDICT: HELD-OUT COLLAPSE -- unseen text->image is ~identical across captions (diversity {m_ev['diversity']:.2f}). No generalization.",flush=True)
print(f"saved: step1_heldout_grid.png, step1_heldout_results.json | ckpt {CKPT}",flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only (synthetic data, tiny config); numbers meaningless -- only confirms the split, train-on-train, eval-on-disjoint, dual readouts, baseline, grid, verdict run end-to-end.",flush=True)
