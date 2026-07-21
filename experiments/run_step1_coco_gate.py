"""CONFERENCE Step 1 GATE -- does the validated recipe generate from REAL TEXT + REAL IMAGES at a real
scale and budget? Re-attacks exactly what mushed in 7c, but with the Run-A/B fixes.

Everything clean so far is MNIST + fake random-token captions (an optimization proxy). The two real-data
attempts (CIFAR-100 staged, COCO 7c) were near-chance BUT undertrained/small (<=62M, 3k steps, CPU) --
never scale-matched to the 3B-clean optimization result. THIS run closes the gap: real COCO sentence
captions + real images, 50-300M params, GPU, a real step budget, GELU + PLAIN LARS + the weight-moving
LR (2e-2) that moved weights 48-72% at scale in Run B.

Recipe (run_B_scale_push.py / midscale_seeds.py): single energy F, GELU, PLAIN LARS + bias trust floor
(NO muP -- Run A showed it under-moves), relax-then-step, dense multi-scale shared-latent anchors (L3),
A_GEN>=A_cross (L4), all grads via tf.GradientTape, standard parameterization.

DATA: small COCO val2017 subset (~300-1000 pairs), REAL sentence captions (lowercased, char-level
one-hot), images 64x64x3 in [0,1] RGB (sigmoid decode target; NOT 572 -- that made 7c impossibly heavy,
64x64 is enough to judge recognizability). prep adapted from prep_coco.py; cached + resumable. SMOKE uses
synthetic images + real-word caption strings so the CPU mechanics test needs NO download.

METRICS (judge on these, NOT F):
  - weight-movement % (if <40% the run UNDERTRAINED -> verdict VOID, train longer).
  - text->image: diversity ratio (varies by caption?) + retrieval top-1 vs chance (=1/N) + output range
    across captions (0 => identical for every caption = the 7c mode-collapse failure).
  - image->image recon MSE, image->text token acc (vs most-common-char baseline).
  - SAVE grids: text->image for several real captions + image->image recon.
CALIBRATION: real images + real sentences at this scale is HARD -- expect blobby, not crisp. The bar is
"output VARIES by caption AND retrieval above chance" (image responds to text), NOT prettiness. 7c failed
by producing IDENTICAL output for every caption; the gate is outputs that DIFFER by caption.

PARAMETERIZED via env (pod run is set-and-go): RUNS1_N (pairs to use), RUNS1_PAIRS (download count),
RUNS1_RES (image size, default 64), RUNS1_CAPLEN (caption chars, default 64), RUNS1_WMUL (size knob),
RUNS1_LR (default 2e-2), RUNS1_STEPS, RUNS1_CKPT (local dir), RUNS1_DATA (cache dir), RUNS1_SEED,
RUNS1_SMOKE (1 = tiny synthetic CPU mechanics check).
"""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNS1_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE  = os.environ.get("RUNS1_SMOKE", "0") == "1"
SEED   = int(os.environ.get("RUNS1_SEED", 0))
RES    = int(os.environ.get("RUNS1_RES", 16 if SMOKE else 64))           # must be divisible by 16 (4 conv-pool stages)
CAPLEN = int(os.environ.get("RUNS1_CAPLEN", 16 if SMOKE else 64))
N_WANT = int(os.environ.get("RUNS1_N", 8 if SMOKE else 400))
PAIRS  = int(os.environ.get("RUNS1_PAIRS", N_WANT + 200))                 # download a few extra (some images fail)
WMUL   = float(os.environ.get("RUNS1_WMUL", 0.1 if SMOKE else 1.5))       # ~150M at wmul=1.5
LR     = float(os.environ.get("RUNS1_LR", 2e-2))
STEPS  = int(os.environ.get("RUNS1_STEPS", 12 if SMOKE else 5000))
CKPT   = os.environ.get("RUNS1_CKPT", "/tmp/s1_ckpt" if SMOKE else "/root")
DATA   = os.environ.get("RUNS1_DATA", "/tmp/s1_data" if SMOKE else "/root/coco_s1")
os.makedirs(CKPT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"

# recipe constants
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER  = 2 if SMOKE else 8
GEN_INFER = 3 if SMOKE else 25
DIVERGE_W = 1e3
MOVE_MIN = 0.40                       # below this => undertrained => verdict void
LOG_EVERY = 5 if SMOKE else 200
CKPT_EVERY = 9999 if SMOKE else 1500
# base widths at wmul=1 (image conv channels, d_model, shared-latent dims, ffn)
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
            if t < len(cp): toks[n,t] = c2i.get(cp[t], nul)
    return toks

def load_synthetic():
    rs = np.random.RandomState(SEED)
    caps = [c.lower() for c in SMOKE_CAPS[:N_WANT]]
    # distinct-ish structured fake images (a colored blob per caption) so recon/retrieval are well-defined
    imgs = np.zeros((len(caps), RES, RES, 3), "float32")
    for i in range(len(caps)):
        imgs[i] = rs.rand(RES,RES,3).astype("float32")*0.3
        c = rs.rand(3); y,x = rs.randint(0,RES,2); r = RES//4
        yy,xx = np.ogrid[:RES,:RES]; m = (yy-y)**2+(xx-x)**2 <= r*r
        imgs[i][m] = c
    chars,c2i = build_vocab(caps); V=len(chars); toks = encode_caps(caps,c2i,CAPLEN)
    return imgs, toks, tf.one_hot(toks,V).numpy().astype("float32"), caps, V, chars

def load_coco():
    f_img,f_tok,f_voc,f_cap = (os.path.join(DATA,x) for x in ("imgs.npy","toks.npy","vocab.npy","caps.txt"))
    if all(os.path.exists(p) for p in (f_img,f_tok,f_voc,f_cap)):
        imgs=np.load(f_img); toks=np.load(f_tok); chars=list(np.load(f_voc)); caps=open(f_cap).read().split("\n")[:len(imgs)]
        V=len(chars); print(f"[data] reused cache {imgs.shape} V={V}",flush=True)
        return imgs, toks, tf.one_hot(toks,V).numpy().astype("float32"), caps, V, chars
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
    imgs=np.asarray(imgs,"float32"); chars,c2i=build_vocab(caps); V=len(chars); toks=encode_caps(caps,c2i,CAPLEN)
    np.save(f_img,imgs); np.save(f_tok,toks); np.save(f_voc,np.array(chars,dtype=object).astype(str)); open(f_cap,"w").write("\n".join(caps))
    print(f"[data] COCO ready {imgs.shape} V={V} ({time.time()-t0:.0f}s)",flush=True)
    return imgs, toks, tf.one_hot(toks,V).numpy().astype("float32"), caps, V, chars

imgs, toks, toks_oh, caps, V, chars = (load_synthetic() if SMOKE else load_coco())
N = len(imgs); PIX = RES*RES*3; DATA_STD = float(np.std(imgs)); CH = 3
print(f"=== Step 1 COCO GATE === smoke={SMOKE} N={N} pairs | image {imgs.shape} [0,1] RGB | caption char-1hot CAPLEN={CAPLEN} V={V} | chance retr={1/N:.4f}",flush=True)
print(f"  sample cap[0]={caps[0]!r}  img range[{imgs.min():.2f},{imgs.max():.2f}] shapes img{imgs[0].shape} cap{toks_oh[0].shape}",flush=True)

# ============================ MODEL (standard param, plain LARS) ============================
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
        h=gelu(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]); h=tf.nn.max_pool2d(h,2,2,"SAME")          # RES/2
        h=gelu(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]); h=tf.nn.max_pool2d(h,2,2,"SAME")          # RES/4
        f0=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]); h=tf.nn.max_pool2d(h,2,2,"SAME")          # RES/8
        f1=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]); h=tf.nn.max_pool2d(h,2,2,"SAME")          # RES/16
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
    def weight_step(x,tk,S,igt,tgt,lr):                      # PLAIN LARS + bias trust floor
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

# ============================ TRAIN ============================
P,c=build(WMUL,SEED); P0={k:tf.identity(v) for k,v in P.items()}; ops=make_ops(P,c)
NP=int(sum(int(np.prod(v.shape)) for v in P.values()))
print(f"model: {NP/1e6:.1f}M params | DM={c['DM']} C=({c['C1']},{c['C2']},{c['C3']},{c['C4']}) DIMS={c['DIMS']} | plain LARS lr={LR} | budget {STEPS} steps",flush=True)
IMG_T=imgs.reshape(N,-1).astype("float32"); TXT_T=toks_oh.reshape(N,-1).astype("float32")
img_t=lambda i: tf.constant(IMG_T[i][None]); txt_t=lambda i: tf.constant(TXT_T[i][None])
order=np.random.RandomState(SEED+7).permutation(N); Fhist=[]; diverged=False; t0=time.time(); lrt=tf.constant(LR,tf.float32)
for s in range(STEPS):
    i=int(order[s%N]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None]); igt=img_t(i); tgt=txt_t(i)
    it,tt=ops["get_taps"](x,tk)
    Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
    F,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt); F=float(F); mxw=float(mxw); Fhist.append(F)
    if not (np.isfinite(F) and mxw<DIVERGE_W):
        diverged=True; print(f"  !! DIVERGENCE step {s}: F={F:.3e} max|w|={mxw:.2e} -> STOP",flush=True); break
    if (s+1)%LOG_EVERY==0:
        mv=movement(P,P0); print(f"  step {s+1:5d} t={(time.time()-t0)/60:.1f}m F={F:.4e} move={mv*100:.1f}% max|w|={mxw:.2e}",flush=True)
    if (s+1)%CKPT_EVERY==0:
        try: np.savez(os.path.join(CKPT,"s1_ckpt.npz"), **{k:P[k].numpy() for k in P})
        except Exception as e: print(f"  ckpt failed: {e}",flush=True)
move=movement(P,P0); elapsed=(time.time()-t0)/60
print(f"\n[train done] steps={len(Fhist)} diverged={diverged} t={elapsed:.1f}m F {Fhist[0]:.3e}->{Fhist[-1]:.3e} | WEIGHT-MOVEMENT={move*100:.1f}%",flush=True)
try: np.savez(os.path.join(CKPT,"s1_ckpt.npz"), **{k:P[k].numpy() for k in P})
except Exception: pass

# ============================ GENERATION READOUTS ============================
def F_img(S,taps,tgt): return 0.5*tf.reduce_mean(tf.add_n([mse(S[k]-taps[k]) for k in range(NS)])+A_GEN*mse(ops["dec_img"](S)-tgt))
def F_txt(S,taps,tgt): return 0.5*tf.reduce_mean(tf.add_n([mse(S[k]-taps[k]) for k in range(NS)])+A_GEN*mse(ops["dec_txt"](S)-tgt))
t2i=np.zeros((N,RES,RES,CH)); i2i=np.zeros((N,RES,RES,CH)); i2t_acc=[]
if not diverged:
    for j in range(N):
        x=tf.constant(imgs[j][None]); tk=tf.constant(toks[j][None]); it,tt=ops["get_taps"](x,tk)
        St=ops["relax_mono"]([tf.identity(tt[k]) for k in range(NS)], tt, ops["dec_txt"], txt_t(j), GEN_INFER)
        t2i[j]=ops["dec_img"](St).numpy().reshape(RES,RES,CH)
        Si=ops["relax_mono"]([tf.identity(it[k]) for k in range(NS)], it, ops["dec_img"], img_t(j), GEN_INFER)
        i2i[j]=ops["dec_img"](Si).numpy().reshape(RES,RES,CH)
        i2t_acc.append(float(np.mean(ops["dec_txt"](Si).numpy().reshape(CAPLEN,V).argmax(-1)==toks[j])))
diversity=float(np.mean(np.std(t2i,0))/(DATA_STD+1e-9)) if not diverged else float("nan")
out_range=float((t2i.max(0)-t2i.min(0)).mean()) if not diverged else float("nan")     # 0 => identical for every caption (7c failure)
d=((t2i.reshape(N,1,-1)-imgs.reshape(1,N,-1))**2).mean(-1) if not diverged else None
retr=float(np.mean(np.argmin(d,1)==np.arange(N))) if not diverged else float("nan")
recon=float(np.mean((i2i-imgs)**2)) if not diverged else float("nan")
i2t=float(np.mean(i2t_acc)) if i2t_acc else float("nan")
mode_char=np.bincount(toks.reshape(-1),minlength=V).argmax(); i2t_base=float(np.mean(toks==mode_char))

print(f"\n==================== STEP 1 GATE METRICS (F is NOT the signal) ====================",flush=True)
print(f"  weight-movement      = {move*100:.1f}%  (>= {MOVE_MIN*100:.0f}% required, else undertrained=VOID)",flush=True)
print(f"  text->image diversity= {diversity:.3f}  (0 => collapse; >0 => varies by caption)",flush=True)
print(f"  text->image retrieval= {retr:.4f}  (chance {1/N:.4f})",flush=True)
print(f"  text->image out-range= {out_range:.3e}  (0 => IDENTICAL image for every caption = the 7c failure)",flush=True)
print(f"  image->image recon   = {recon:.4f}",flush=True)
print(f"  image->text token acc= {i2t:.3f}  (baseline {i2t_base:.3f})",flush=True)

# ============================ GRIDS ============================
if not diverged:
    nc=min(8,N)
    fig,axes=plt.subplots(3,nc,figsize=(1.5*nc,4.8))
    for j in range(nc):
        axes[0,j].imshow(np.clip(imgs[j],0,1)); axes[0,j].axis("off"); axes[0,j].set_title(caps[j][:22],fontsize=5)
        axes[1,j].imshow(np.clip(t2i[j],0,1)); axes[1,j].axis("off")
        axes[2,j].imshow(np.clip(i2i[j],0,1)); axes[2,j].axis("off")
    for r,l in [(0,"target"),(1,"text->img"),(2,"img->img")]: axes[r,0].set_ylabel(l,fontsize=8)
    plt.suptitle(f"Step1 COCO gate ({NP/1e6:.0f}M, move {move*100:.0f}%): text->image must VARY by caption (retr {retr:.3f} vs chance {1/N:.4f})",fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(HERE,"step1_coco_grid.png"),dpi=120); plt.close()

dump=dict(config=dict(smoke=SMOKE,N=N,RES=RES,CAPLEN=CAPLEN,V=V,wmul=WMUL,params=NP,lr=LR,steps=len(Fhist),seed=SEED,chance=1/N),
          diverged=diverged,move=move,diversity=diversity,retr=retr,out_range=out_range,recon=recon,i2t=i2t,i2t_base=i2t_base,
          F0=Fhist[0],Fend=Fhist[-1],elapsed_min=elapsed)
with open(os.path.join(HERE,"step1_coco_results.json"),"w") as fh: json.dump(dump,fh,indent=2)

# ============================ VERDICT ============================
print(f"\n==================== STEP 1 GATE VERDICT ====================",flush=True)
if diverged:
    print("VERDICT: DIVERGED -- non-finite/blowup. Lower LR or check setup.",flush=True)
elif move < MOVE_MIN:
    print(f"VERDICT: VOID (UNDERTRAINED) -- weights moved only {move*100:.1f}% (<{MOVE_MIN*100:.0f}%). Train longer/raise LR before judging generation.",flush=True)
else:
    varies = (diversity >= 0.20) and (out_range > 1e-2)
    above_chance = retr > 3.0/N
    if varies and above_chance:
        print(f"VERDICT: GATE PASS -- weights moved {move*100:.0f}%, and text->image VARIES by caption (diversity {diversity:.2f}, out-range {out_range:.1e}) and is ABOVE CHANCE (retr {retr:.3f} = {retr*N:.0f}x chance). Real text->image at scale generates -- unlike the 7c mode-collapse.",flush=True)
    elif varies and not above_chance:
        print(f"VERDICT: PARTIAL -- output VARIES by caption (diversity {diversity:.2f}) but retrieval not above chance ({retr:.3f} vs {1/N:.4f}): responds to text but not yet matching the right image. Better than 7c (which was identical-for-all), short of recognizable.",flush=True)
    else:
        print(f"VERDICT: MODE-COLLAPSE (7c-style) -- weights moved {move*100:.0f}% but text->image is ~IDENTICAL for every caption (diversity {diversity:.2f}, out-range {out_range:.1e}). Weight movement did NOT fix the real-data collapse; the bottleneck is elsewhere (objective/decoder/data).",flush=True)
print(f"saved: step1_coco_grid.png, step1_coco_results.json | ckpt {CKPT}",flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only (synthetic data, tiny config); numbers are meaningless -- only confirms the pipeline, shapes, train loop, metrics, grids, and verdict logic run end-to-end.",flush=True)
