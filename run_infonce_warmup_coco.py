"""CONFERENCE experiment -- does an InfoNCE coupling WARM-UP make text->image MATCH on HELD-OUT captions?

CONTEXT. The Step 1 held-out re-test (commit d36b6ae) showed real-COCO text->image BREAKS mode-collapse
(output varies by caption) but does NOT match unseen captions above chance -- caption-varying blobs, not
matching scenes. Diagnosed cause: the shared latent is shaped mostly by the IMAGE side during training,
so a text-only latent lands OFF-distribution where the decoder is weak, hence blobs. This run tests the
lever aimed at that exact weakness: a contrastive (InfoNCE) warm-up that ALIGNS the text-latent and
image-latent so the text-latent lands on-distribution before joint training.

RECIPE FROZEN. Built on run_step1_coco_heldout.py / run_step1_coco_gate.py (d36b6ae): real COCO val2017,
real sentence captions char-level, 64x64x3 [0,1] images, ~156M params, GELU, plain LARS + bias floor,
relax-then-step, dense multi-scale anchors, A_GEN>=A_cross, lr=2e-2. NOTHING in the recipe changes except
the optional warm-up. Train on N_TRAIN, evaluate on a DISJOINT held-out set, SAME split as d36b6ae
(perm seed+1, train-only vocab, no leakage). Both arms start from the IDENTICAL random init and see the
IDENTICAL data/order -- the ONLY difference is the warm-up. Judge on HELD-OUT only.

ARMS (run in one process, matched init/data):
  A      BASELINE   recipe as-is, JOINT steps, no warm-up. Control; should reproduce held-out NOT matching.
  B      WARM-UP    InfoNCE warm-up (align image-latent <-> text-latent on paired data) THEN the same
                    JOINT training on the aligned init.
  A_long CONTROL    [optional, RUNS1_CONTROL=1] no warm-up, JOINT+WARMUP joint steps. Isolates the
                    warm-up from "B simply had more total gradient steps". TURN THIS ON for the real run.

VALIDITY NOTES baked in (see the run prompt / chat):
  - A_long defuses the compute confound. Enable it on the pod (RUNS1_CONTROL=1).
  - RUNS1_JOINTW>0 keeps the InfoNCE term ON during joint training, in case the warm-up alignment washes
    out under the image-dominated joint phase (the post-joint alignment metric reveals if it did).
  - Held-out N=100 is underpowered; on the pod use RUNS1_NEVAL 300-500 and sweep >=3 seeds.

MEASURE on HELD-OUT (the only real bar) for every arm:
  - weight-movement % from the original init (>40% or VOID/extend).
  - text->image HELD-OUT retrieval top-1 vs chance (=1/N_eval). HEADLINE: does B beat chance where A does
    not? Pre-registered bar = retr > 3/N_eval (~2-3 sigma), NOT "Nx chance" on a big pool.
  - text->image held-out diversity + out-range (still varies / not collapsed).
  - cross-modal latent ALIGNMENT on held-out: matched-pair cosine + latent-space retrieval top-1
    (B should align MORE than A -- that is the mechanism).
  - image->image recon (vs train-mean baseline) + image->text acc (sanity; should hold in both).
  - SAVE side-by-side held-out text->image grids (A vs B) on UNSEEN captions.

VERDICT (held-out only): B beats chance where A does not => the warm-up is the lever, scale it up. B does
NOT beat chance => warm-up does not crack the hard direction at this scale, report honestly (bounds the
limitation -> workshop-with-scope). Report alignment delta regardless (mechanism worked but insufficient
is itself informative). NEVER report in-sample gains as success.

ENV: RUNS1_NTRAIN(400) RUNS1_NEVAL(100) RUNS1_RES(64) RUNS1_CAPLEN(64) RUNS1_WMUL(1.5) RUNS1_LR(2e-2)
RUNS1_WARMUP(1500) RUNS1_JOINT(5000) RUNS1_BATCH(64) RUNS1_TEMP(0.07) RUNS1_CONTROL(0) RUNS1_JOINTW(0.0)
RUNS1_SEED(0) RUNS1_CKPT RUNS1_DATA RUNS1_SMOKE(1=tiny CPU mechanics check, numbers meaningless).
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
PAIRS  = int(os.environ.get("RUNS1_PAIRS", N_WANT + 300))
WMUL   = float(os.environ.get("RUNS1_WMUL", 0.1 if SMOKE else 1.5))
LR     = float(os.environ.get("RUNS1_LR", 2e-2))
WARMUP = int(os.environ.get("RUNS1_WARMUP", 6 if SMOKE else 1500))
JOINT  = int(os.environ.get("RUNS1_JOINT", 10 if SMOKE else 5000))
BATCH  = int(os.environ.get("RUNS1_BATCH", 4 if SMOKE else 64))
TEMP   = float(os.environ.get("RUNS1_TEMP", 0.07))
CONTROL= os.environ.get("RUNS1_CONTROL", "1" if SMOKE else "0") == "1"   # A_long compute-matched control
JOINTW = float(os.environ.get("RUNS1_JOINTW", 0.1 if SMOKE else 0.0))    # persistent InfoNCE during joint
CKPT   = os.environ.get("RUNS1_CKPT", "/tmp/coup_ckpt" if SMOKE else "/root")
DATA   = os.environ.get("RUNS1_DATA", "/tmp/s1ho_data" if SMOKE else "/root/coco_s1ho")
os.makedirs(CKPT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0, "RES must be divisible by 16"

# recipe constants (identical to the gate / held-out)
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER  = 2 if SMOKE else 8
GEN_INFER = 3 if SMOKE else 25
DIVERGE_W = 1e3
MOVE_MIN = 0.40
LOG_EVERY = 5 if SMOKE else 200
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

# ============================ DATA (identical loader/split to run_step1_coco_heldout.py) ============================
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
perm = np.random.RandomState(SEED+1).permutation(N_HAVE)              # SAME split rule as d36b6ae
tr_idx = perm[:N_TRAIN]; ev_idx = perm[N_TRAIN:N_TRAIN+N_EVAL]
imgs = imgs_all; caps = caps_all
NTR, NEV = len(tr_idx), len(ev_idx)
PIX = RES*RES*3; CH = 3
chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)   # train-only vocab
toks = encode_caps(caps, c2i, CAPLEN); toks_oh = tf.one_hot(toks, V).numpy().astype("float32")
train_mean_img = imgs[tr_idx].mean(0)
print(f"=== InfoNCE WARM-UP experiment === smoke={SMOKE} N_have={N_HAVE} -> train={NTR} eval={NEV} (disjoint) | img {imgs.shape[1:]} | CAPLEN={CAPLEN} V={V} | warmup={WARMUP} joint={JOINT} batch={BATCH} temp={TEMP} | control={CONTROL} jointw={JOINTW} | chance retr eval={1/max(NEV,1):.4f}",flush=True)

# ============================ MODEL (identical to gate/held-out) ============================
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
    def weight_step(x,tk,S,igt,tgt,lr):                                 # PLAIN LARS + bias trust floor
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
    # ---- coupling (InfoNCE) additions ----
    def l2n(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))   # zi, zt
    def infonce(zi,zt,temp):
        logits=tf.matmul(zi,zt,transpose_b=True)/temp; B=tf.shape(zi)[0]; lab=tf.range(B)
        return 0.5*(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=logits))
                   +tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=tf.transpose(logits))))
    @tf.function
    def warmup_step(xb,tkb,lr,temp):                                    # LARS on InfoNCE (encoder grads only)
        with tf.GradientTape() as t: t.watch(ALL_W); zi,zt=latents(xb,tkb); L=infonce(zi,zt,temp)
        gr=t.gradient(L,ALL_W)
        for v,gg in zip(ALL_W,gr):
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
        return L
    return dict(get_taps=get_taps,relax_full=relax_full,weight_step=weight_step,relax_mono=relax_mono,
                dec_img=dec_img,dec_txt=dec_txt,latents=latents,warmup_step=warmup_step,infonce=infonce)

def movement(P,P0):
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P)))
    den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

# ============================ BUILD ONCE, snapshot init (arms share it) ============================
P,c=build(WMUL,SEED); P_init={k:v.numpy().copy() for k,v in P.items()}; ops=make_ops(P,c)
NP=int(sum(int(np.prod(v.shape)) for v in P.values()))
print(f"model: {NP/1e6:.1f}M params | DM={c['DM']} DIMS={c['DIMS']} | plain LARS lr={LR}",flush=True)
IMG_T=imgs.reshape(N_HAVE,-1).astype("float32"); TXT_T=toks_oh.reshape(N_HAVE,-1).astype("float32")
img_t=lambda i: tf.constant(IMG_T[i][None]); txt_t=lambda i: tf.constant(TXT_T[i][None])
def reset(): [P[k].assign(P_init[k]) for k in P]

# ============================ READOUTS (a given index set) ============================
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
    recon=float(np.mean((i2i-real)**2)); recon_base=float(np.mean((train_mean_img[None]-real)**2))
    i2t=float(np.mean(i2t_acc))
    # cross-modal latent alignment (the mechanism)
    ZI,ZT=ops["latents"](tf.constant(imgs[idx]), tf.constant(toks[idx])); ZI=ZI.numpy(); ZT=ZT.numpy()
    align_cos=float(np.mean(np.sum(ZI*ZT,1)))                       # matched-pair cosine (normed)
    sim=ZT@ZI.T; lat_retr=float(np.mean(np.argmax(sim,1)==np.arange(M)))   # text-latent -> image-latent top-1
    hits=int(round(retr*M))
    return dict(M=M,diversity=diversity,out_range=out_range,retr=retr,hits=hits,chance=1.0/M,
                recon=recon,recon_base=recon_base,i2t=i2t,align_cos=align_cos,lat_retr=lat_retr), t2i

mode_char=int(np.bincount(toks[tr_idx].reshape(-1),minlength=V).argmax())
def i2t_base_on(idx): return float(np.mean(toks[idx]==mode_char))

# ============================ ARM RUNNER ============================
warm_rs=np.random.RandomState(SEED+11); joint_order=np.random.RandomState(SEED+7).permutation(NTR)
def warmup_phase(steps):
    if steps<=0: return
    lrt=tf.constant(LR,tf.float32); tmp=tf.constant(TEMP,tf.float32); t0=time.time()
    for s in range(steps):
        b=warm_rs.choice(NTR, size=min(BATCH,NTR), replace=False); bi=tr_idx[b]
        L=float(ops["warmup_step"](tf.constant(imgs[bi]), tf.constant(toks[bi]), lrt, tmp))
        if (s+1)%LOG_EVERY==0: print(f"    [warmup] {s+1:5d} infonce={L:.4f} t={(time.time()-t0)/60:.1f}m",flush=True)

def joint_phase(steps, jointw):
    lrt=tf.constant(LR,tf.float32); tmp=tf.constant(TEMP,tf.float32); Fhist=[]; diverged=False; t0=time.time()
    for s in range(steps):
        i=int(tr_idx[joint_order[s%NTR]]); x=tf.constant(imgs[i][None]); tk=tf.constant(toks[i][None]); igt=img_t(i); tgt=txt_t(i)
        it,tt=ops["get_taps"](x,tk)
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt); F=float(F); mxw=float(mxw); Fhist.append(F)
        if jointw>0:                                               # optional persistent coupling
            b=warm_rs.choice(NTR, size=min(BATCH,NTR), replace=False); bi=tr_idx[b]
            ops["warmup_step"](tf.constant(imgs[bi]), tf.constant(toks[bi]), tf.constant(LR*jointw,tf.float32), tmp)
        if not (np.isfinite(F) and mxw<DIVERGE_W):
            diverged=True; print(f"    !! DIVERGENCE joint step {s}: F={F:.3e} max|w|={mxw:.2e}",flush=True); break
        if (s+1)%LOG_EVERY==0: print(f"    [joint] {s+1:5d} F={F:.4e} move={movement(P,P_init)*100:.1f}% t={(time.time()-t0)/60:.1f}m",flush=True)
    return Fhist, diverged

def run_arm(name, do_warmup, joint_steps, jointw):
    print(f"\n----- ARM {name} (warmup={'yes' if do_warmup else 'no'}, joint={joint_steps}, jointw={jointw}) -----",flush=True)
    reset(); t0=time.time()
    if do_warmup: warmup_phase(WARMUP)
    Fhist,diverged=joint_phase(joint_steps, jointw)
    move=movement(P,P_init); elapsed=(time.time()-t0)/60
    try: np.savez(os.path.join(CKPT,f"coup_{name}.npz"), **{k:P[k].numpy() for k in P})
    except Exception: pass
    if diverged: return dict(name=name,diverged=True,move=move,elapsed=elapsed), None
    m_tr,_=readouts(tr_idx); m_ev,t2i_ev=(readouts(ev_idx) if NEV else (None,None))
    print(f"  ARM {name}: move={move*100:.1f}% | HELD-OUT retr={m_ev['retr']:.4f} ({m_ev['hits']}/{NEV}, chance {m_ev['chance']:.4f}) "
          f"align_cos={m_ev['align_cos']:.3f} lat_retr={m_ev['lat_retr']:.3f} diversity={m_ev['diversity']:.3f} recon={m_ev['recon']:.4f}",flush=True)
    return dict(name=name,diverged=False,move=move,elapsed=elapsed,train=m_tr,heldout=m_ev), t2i_ev

# ============================ RUN ARMS ============================
resA,t2iA = run_arm("A", False, JOINT, 0.0)
resB,t2iB = run_arm("B", True,  JOINT, JOINTW)
resL,_    = run_arm("A_long", False, JOINT+WARMUP, 0.0) if CONTROL else (None,None)

# ============================ GRID (held-out, A vs B side by side) ============================
if NEV and (t2iA is not None) and (t2iB is not None):
    nc=min(8,NEV); fig,axes=plt.subplots(3,nc,figsize=(1.5*nc,4.8))
    for jj in range(nc):
        j=int(ev_idx[jj])
        axes[0,jj].imshow(np.clip(imgs[j],0,1)); axes[0,jj].axis("off"); axes[0,jj].set_title(caps[j][:22],fontsize=5)
        axes[1,jj].imshow(np.clip(t2iA[jj],0,1)); axes[1,jj].axis("off")
        axes[2,jj].imshow(np.clip(t2iB[jj],0,1)); axes[2,jj].axis("off")
    for r,l in [(0,"target"),(1,"A text->img"),(2,"B text->img")]: axes[r,0].set_ylabel(l,fontsize=8)
    ra=resA['heldout']['retr'] if resA and not resA['diverged'] else float('nan')
    rb=resB['heldout']['retr'] if resB and not resB['diverged'] else float('nan')
    plt.suptitle(f"InfoNCE warm-up, HELD-OUT text->image (UNSEEN). A retr {ra:.3f} vs B retr {rb:.3f} (chance {1/NEV:.4f})",fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(HERE,"infonce_warmup_grid.png"),dpi=120); plt.close()

dump=dict(config=dict(smoke=SMOKE,N_have=N_HAVE,N_train=NTR,N_eval=NEV,RES=RES,CAPLEN=CAPLEN,V=V,wmul=WMUL,params=NP,
                      lr=LR,warmup=WARMUP,joint=JOINT,batch=BATCH,temp=TEMP,control=CONTROL,jointw=JOINTW,seed=SEED),
          arm_A=resA,arm_B=resB,arm_A_long=resL,
          i2t_base_train=i2t_base_on(tr_idx),i2t_base_eval=(i2t_base_on(ev_idx) if NEV else None))
with open(os.path.join(HERE,"infonce_warmup_results.json"),"w") as fh: json.dump(dump,fh,indent=2)

# ============================ VERDICT (HELD-OUT only) ============================
print(f"\n==================== InfoNCE WARM-UP VERDICT (held-out only) ====================",flush=True)
def valid(r): return r is not None and not r["diverged"] and r["move"]>=MOVE_MIN
def above_chance(r): return valid(r) and NEV and r["heldout"]["retr"] > 3.0/NEV
if not (valid(resA) and valid(resB)):
    bad=[r['name'] for r in (resA,resB) if not valid(r)]
    print(f"VERDICT: VOID -- arm(s) {bad} diverged or moved <{MOVE_MIN*100:.0f}%. Extend steps / check setup before judging.",flush=True)
elif not NEV:
    print("VERDICT: NO EVAL SET -- not enough pairs for a held-out split.",flush=True)
else:
    A_ac, B_ac = above_chance(resA), above_chance(resB)
    da=resB["heldout"]["align_cos"]-resA["heldout"]["align_cos"]
    ctl = f" (A_long control retr {resL['heldout']['retr']:.3f}, {resL['heldout']['hits']}/{NEV})" if valid(resL) else " (NO compute-matched control run -- enable RUNS1_CONTROL=1 on the pod to defuse the budget confound)"
    if B_ac and not A_ac:
        print(f"VERDICT: WARM-UP IS THE LEVER -- held-out text->image beats chance with warm-up (B retr {resB['heldout']['retr']:.3f}={resB['heldout']['hits']}/{NEV}) but NOT without (A retr {resA['heldout']['retr']:.3f}={resA['heldout']['hits']}/{NEV}). Latent alignment {'rose' if da>0 else 'did not rise'} (delta cos {da:+.3f}).{ctl} Conference result if it survives the control + multi-seed -> scale up.",flush=True)
    elif B_ac and A_ac:
        print(f"VERDICT: BOTH ABOVE CHANCE -- A unexpectedly matched too (A {resA['heldout']['hits']}/{NEV}, B {resB['heldout']['hits']}/{NEV}). Re-examine the baseline; warm-up effect not isolated.{ctl}",flush=True)
    else:
        print(f"VERDICT: WARM-UP DOES NOT CRACK IT -- held-out text->image still not above chance (A {resA['heldout']['hits']}/{NEV}, B {resB['heldout']['hits']}/{NEV}, bar >{int(3)} hits). At this scale/budget the coupling warm-up is insufficient. Honest scope -> workshop. Mechanism check: alignment delta cos {da:+.3f} ({'B aligned more (mechanism worked but not sufficient)' if da>0 else 'B did NOT align more -- warm-up washed out in joint; try RUNS1_JOINTW>0'}).",flush=True)
print(f"saved: infonce_warmup_grid.png, infonce_warmup_results.json | ckpt {CKPT}",flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only (synthetic data, tiny config); numbers meaningless -- confirms warm-up phase, joint phase, persistent-coupling path, A_long control, held-out split, InfoNCE + alignment metric, A-vs-B grid, and verdict logic run end-to-end.",flush=True)
