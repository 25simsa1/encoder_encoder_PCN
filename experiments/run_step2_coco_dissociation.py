"""CONFERENCE Step 2 -- the DISSOCIATION MATRIX on REAL COCO (the novel-core experiment).

Re-runs the 2x2x2 dissociation (validated on MNIST in dissociation.py / DISSOCIATION.md) on the
real-COCO text->image task that PASSED the Step 1 gate (run_step1_coco_gate.py, commit 00304a9: 156M,
weights moved 108%, retrieval 10x chance, out-range 0.54 -- collapse broke when weights moved). Goal:
show "ONLY weight-movement flips mode-collapse" holds on REAL sentence captions + real images, not just
the MNIST proxy. That is what makes the dissociation conference-grade.

ASSEMBLY (combine two validated things; recipe/architecture UNCHANGED):
  - Step 1 COCO pipeline + recipe: real val2017 sentence captions (char one-hot), images 64x64x3 [0,1];
    single energy F, GELU, plain LARS + bias floor, relax-then-step, multi-scale shared-latent anchors
    (L3), A_GEN>=A_cross (L4); ~156M (the size that passed the gate).
  - dissociation.py 2x2x2 factorial: A1 latent width (narrow vs wide shared-latent DIMS), A2 anti-mean
    (InfoNCE contrastive on the latents, off vs on), A3 weight movement (LOW tiny-LR coast vs HIGH
    lr=2e-2 -- the value that moved weights 108% on COCO in Step 1).
  ONE necessary assembly choice, stated honestly: the anti-mean axis (InfoNCE) needs a BATCH of
  negatives, so this matrix trains MINI-BATCH (in-batch InfoNCE) like dissociation.py's full-batch,
  NOT Step 1's batch-1. Every other recipe element is identical.

8 cells = move(LOW/HIGH) x latent(narrow/wide) x anti(off/on). Fresh model per cell, SAME seed/init/
data/budget across cells; ONLY the swept axis differs. Per cell report weight-movement %, text->image
diversity, retrieval vs chance (=1/N), out-range (0 = identical image every caption = collapse), recon,
and F (ONLY to show it drops in every cell -- F is NOT the signal).

CLAIM: LOW-movement cells COLLAPSE regardless of width/anti; HIGH-movement cells BREAK collapse
regardless of width/anti. So ONLY weight-movement flips it, on real text+images. If wide latent OR
anti-mean ALSO flips collapse at LOW movement -> dissociation FAILS on real data -- reported honestly.
Per-cell VOID-if-HIGH-and-move<40% guard so an undertrained HIGH cell can't masquerade (LOW cells are
intentionally low-movement).

PARAMETERIZED via env (pod set-and-go): ST2_N, ST2_PAIRS, ST2_RES, ST2_CAPLEN, ST2_B (mini-batch),
ST2_WMUL (encoder size; wide DIMS), ST2_NARROW_DIV (narrow = wide/div), ST2_LR_LOW, ST2_LR_HIGH,
ST2_STEPS, ST2_SEEDS (comma), ST2_CKPT, ST2_DATA, ST2_SMOKE.
"""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("ST2_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE  = os.environ.get("ST2_SMOKE", "0") == "1"
SEEDS  = [int(s) for s in os.environ.get("ST2_SEEDS", "0").split(",")]
RES    = int(os.environ.get("ST2_RES", 16 if SMOKE else 64))
CAPLEN = int(os.environ.get("ST2_CAPLEN", 16 if SMOKE else 64))
N_WANT = int(os.environ.get("ST2_N", 8 if SMOKE else 256))
PAIRS  = int(os.environ.get("ST2_PAIRS", N_WANT + 150))
B      = int(os.environ.get("ST2_B", 4 if SMOKE else 48))                 # mini-batch (in-batch InfoNCE)
WMUL   = float(os.environ.get("ST2_WMUL", 0.1 if SMOKE else 1.5))         # encoder size; wide DIMS (~156M at 1.5)
NARROW_DIV = int(os.environ.get("ST2_NARROW_DIV", 16))                    # narrow DIMS = wide // div
LR_LOW  = float(os.environ.get("ST2_LR_LOW", 1e-5))                       # A3 LOW: weights coast (reproduce collapse)
LR_HIGH = float(os.environ.get("ST2_LR_HIGH", 2e-2))                      # A3 HIGH: the Step-1 weight-moving LR
STEPS  = int(os.environ.get("ST2_STEPS", 12 if SMOKE else 1500))
CKPT   = os.environ.get("ST2_CKPT", "/tmp/s2_ckpt" if SMOKE else "/root")
DATA   = os.environ.get("ST2_DATA", "/tmp/s2_data" if SMOKE else "/root/coco_s2")
os.makedirs(CKPT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
assert RES % 16 == 0

HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER  = 2 if SMOKE else 8
GEN_INFER = 3 if SMOKE else 25
DIVERGE_W = 1e3
MOVE_MIN = 0.40                  # HIGH cell with move<this => VOID (undertrained); LOW cells intentionally low
TAU = 0.30                       # collapse iff diversity ratio < TAU (from the MNIST dissociation)
ANTI = 1.0
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

# ============================ DATA (Step 1 pipeline) ============================
SMOKE_CAPS=["a dog runs across a grassy field","a red bus on a city street","two people sit on a bench",
    "a plate of food on a table","a cat sleeping on a couch","a man riding a surfboard","a clock tower in blue sky",
    "a bowl of fruit by a window","a train on the tracks","a child flying a kite"]
def build_vocab(caps): chars=sorted(set("".join(caps))|{"\0"}); return chars,{c:i for i,c in enumerate(chars)}
def encode_caps(caps,c2i,L):
    nul=c2i["\0"]; t=np.full((len(caps),L),nul,"int32")
    for n,cp in enumerate(caps):
        for i in range(L):
            if i<len(cp): t[n,i]=c2i.get(cp[i],nul)
    return t
def load_synthetic():
    rs=np.random.RandomState(0); caps=[c.lower() for c in SMOKE_CAPS[:N_WANT]]
    imgs=np.zeros((len(caps),RES,RES,3),"float32")
    for i in range(len(caps)):
        imgs[i]=rs.rand(RES,RES,3).astype("float32")*0.3; col=rs.rand(3); y,x=rs.randint(0,RES,2); r=RES//4
        yy,xx=np.ogrid[:RES,:RES]; imgs[i][(yy-y)**2+(xx-x)**2<=r*r]=col
    chars,c2i=build_vocab(caps); V=len(chars); toks=encode_caps(caps,c2i,CAPLEN)
    return imgs,toks,tf.one_hot(toks,V).numpy().astype("float32"),caps,V
def load_coco():
    fi,ft,fv,fc=(os.path.join(DATA,x) for x in ("imgs.npy","toks.npy","vocab.npy","caps.txt"))
    if all(os.path.exists(p) for p in (fi,ft,fv,fc)):
        imgs=np.load(fi);toks=np.load(ft);chars=list(np.load(fv));caps=open(fc).read().split("\n")[:len(imgs)]
        print(f"[data] reused cache {imgs.shape} V={len(chars)}",flush=True); return imgs,toks,tf.one_hot(toks,len(chars)).numpy().astype("float32"),caps,len(chars)
    import urllib.request,zipfile
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    ANN="http://images.cocodataset.org/annotations/annotations_trainval2017.zip";IMG="http://images.cocodataset.org/val2017/{}"
    imgdir=os.path.join(DATA,"img");os.makedirs(imgdir,exist_ok=True);capj=os.path.join(DATA,"cap.json");t0=time.time()
    if not os.path.exists(capj):
        z=os.path.join(DATA,"ann.zip")
        if not os.path.exists(z): print("[data] downloading COCO annotations...",flush=True);urllib.request.urlretrieve(ANN,z)
        with zipfile.ZipFile(z) as zf,zf.open("annotations/captions_val2017.json") as s,open(capj,"wb") as d: d.write(s.read())
    cap=json.load(open(capj));id2cap={}
    for a in cap["annotations"]: id2cap.setdefault(a["image_id"],a["caption"])
    id2file={im["id"]:im["file_name"] for im in cap["images"]};ids=[i for i in id2cap if i in id2file][:PAIRS]
    def dl(iid):
        p=os.path.join(imgdir,id2file[iid])
        if not os.path.exists(p):
            try: urllib.request.urlretrieve(IMG.format(id2file[iid]),p)
            except Exception: pass
    with ThreadPoolExecutor(max_workers=32) as ex: list(ex.map(dl,ids))
    imgs,caps=[],[]
    for iid in ids:
        p=os.path.join(imgdir,id2file[iid])
        if not os.path.exists(p): continue
        try:
            im=Image.open(p).convert("RGB").resize((RES,RES)); imgs.append(np.asarray(im,"float32")/255.0); caps.append(id2cap[iid].strip().lower())
            if len(imgs)>=N_WANT: break
        except Exception: pass
    imgs=np.asarray(imgs,"float32");chars,c2i=build_vocab(caps);V=len(chars);toks=encode_caps(caps,c2i,CAPLEN)
    np.save(fi,imgs);np.save(ft,toks);np.save(fv,np.array(chars,dtype=object).astype(str));open(fc,"w").write("\n".join(caps))
    print(f"[data] COCO ready {imgs.shape} V={V} ({time.time()-t0:.0f}s)",flush=True)
    return imgs,toks,tf.one_hot(toks,V).numpy().astype("float32"),caps,V

imgs,toks,toks_oh,caps,V = (load_synthetic() if SMOKE else load_coco())
N=len(imgs); PIX=RES*RES*3; CH=3; DATA_STD=float(np.std(imgs)); B=min(B,N)
print(f"=== Step 2 COCO DISSOCIATION === smoke={SMOKE} N={N} | img {imgs.shape} | cap CAPLEN={CAPLEN} V={V} | mini-batch B={B} | chance retr={1/N:.4f}",flush=True)

# ============================ MODEL (Step 1 recipe; DIMS is the swept latent-width axis) ============================
def enc_cfg(wmul):
    r=lambda x:max(4,int(round(x*wmul)));DM=r(B_DM);DM-=DM%HEADS
    return dict(DM=max(HEADS,DM),C1=r(B_C1),C2=r(B_C2),C3=r(B_C3),C4=r(B_C4),BN=r(B_BN),FFN=r(B_FFN),HEAD=max(1,(max(HEADS,DM))//HEADS))
def wide_dims(wmul): return [max(4,int(round(d*wmul))) for d in B_DIMS]

def build(c, DIMS, seed):
    DM,C1,C2,C3,C4,BN,FFN=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["FFN"]
    s2,s3,s4=RES//4,RES//8,RES//16; f0d,f1d,f2d=s2*s2*C2,s3*s3*C3,s4*s4*C4
    g=tf.random.Generator.from_seed(seed)
    def W(shape,key=""):
        sd=DEC_SD if (key.startswith("proj") or key in ("W_DI","W_DT")) else 1.0/np.sqrt(np.prod(shape[:-1]))
        return tf.Variable(g.normal(shape,stddev=sd))
    def Z(shape): return tf.Variable(tf.zeros(shape))
    P=dict(c1=W([3,3,CH,C1]),cb1=Z([C1]),c2=W([3,3,C1,C2]),cb2=Z([C2]),c3=W([3,3,C2,C3]),cb3=Z([C3]),
           c4=W([3,3,C3,C4]),cb4=Z([C4]),wbn=W([f2d,BN]),bbn=Z([BN]),
           Wi0=W([f0d,DIMS[0]]),bi0=Z([DIMS[0]]),Wi1=W([f1d,DIMS[1]]),bi1=Z([DIMS[1]]),
           Wi2=W([f2d,DIMS[2]]),bi2=Z([DIMS[2]]),Wi3=W([BN,DIMS[3]]),bi3=Z([DIMS[3]]),emb=W([V,DM]),pos=W([CAPLEN,DM]))
    for b in range(NBLK):
        P[f"Wq{b}"]=W([DM,DM]);P[f"Wk{b}"]=W([DM,DM]);P[f"Wv{b}"]=W([DM,DM]);P[f"Wo{b}"]=W([DM,DM])
        P[f"f1_{b}"]=W([DM,FFN]);P[f"fb1_{b}"]=Z([FFN]);P[f"f2_{b}"]=W([FFN,DM]);P[f"fb2_{b}"]=Z([DM])
        P[f"Wt{b}"]=W([DM,DIMS[b]]);P[f"bt{b}"]=Z([DIMS[b]])
    for k in range(NS): P[f"proj{k}"]=W([DIMS[k],CODE],f"proj{k}")
    P["W_DI"]=W([NS*CODE,PIX],"W_DI");P["B_DI"]=Z([PIX]);P["W_DT"]=W([NS*CODE,CAPLEN*V],"W_DT");P["B_DT"]=Z([CAPLEN*V])
    return P

def make_ops(P,c,DIMS,anti):
    DM,C1,C2,C3,C4,BN,FFN,HEAD=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["FFN"],c["HEAD"]
    betas=[REL_C*d for d in DIMS]; ALL_W=list(P.values())
    def enc_img(x):
        h=gelu(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
        h=gelu(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]);h=tf.nn.max_pool2d(h,2,2,"SAME"); f0=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]);h=tf.nn.max_pool2d(h,2,2,"SAME"); f1=tf.reshape(h,[tf.shape(x)[0],-1])
        h=gelu(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]);h=tf.nn.max_pool2d(h,2,2,"SAME"); f2=tf.reshape(h,[tf.shape(x)[0],-1]); f3=gelu(f2@P["wbn"]+P["bbn"])
        return [gelu(f0@P["Wi0"]+P["bi0"]),gelu(f1@P["Wi1"]+P["bi1"]),gelu(f2@P["Wi2"]+P["bi2"]),gelu(f3@P["Wi3"]+P["bi3"])]
    def enc_txt(tk):
        Bt=tf.shape(tk)[0]; x=tf.gather(P["emb"],tk)+P["pos"][None]; tt=[]
        for b in range(NBLK):
            q,k_,v=x@P[f"Wq{b}"],x@P[f"Wk{b}"],x@P[f"Wv{b}"]
            sp=lambda t: tf.transpose(tf.reshape(t,[Bt,CAPLEN,HEADS,HEAD]),[0,2,1,3])
            a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
            ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[Bt,CAPLEN,DM])
            x=x+ctx@P[f"Wo{b}"]; x=x+(gelu(x@P[f"f1_{b}"]+P[f"fb1_{b}"])@P[f"f2_{b}"]+P[f"fb2_{b}"])
            tt.append(gelu(tf.reduce_mean(x,1)@P[f"Wt{b}"]+P[f"bt{b}"]))
        return tt
    def code_of(S): return tf.concat([gelu(S[k]@P[f"proj{k}"]) for k in range(NS)],axis=1)
    def dec_img(S): return tf.nn.sigmoid(code_of(S)@P["W_DI"]+P["B_DI"])
    def dec_txt(S): return code_of(S)@P["W_DT"]+P["B_DT"]
    def F_full(S,it,tt,igt,tgt):
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        return 0.5*tf.reduce_mean(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
    def infonce(it,tt,temp=0.2):                                  # in-batch contrastive on the (concatenated) latents
        zi=tf.math.l2_normalize(tf.concat(it,1),1); zt=tf.math.l2_normalize(tf.concat(tt,1),1)
        logits=zt@tf.transpose(zi)/temp; lab=tf.range(tf.shape(zi)[0])
        return 0.5*(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lab,logits))+
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lab,tf.transpose(logits))))
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
        with tf.GradientTape() as t:
            t.watch(ALL_W); it,tt=enc_img(x),enc_txt(tk); F=F_full(S,it,tt,igt,tgt)
            if anti>0: F=F+anti*infonce(it,tt)
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
    num=float(tf.sqrt(sum(tf.reduce_sum((P[k]-P0[k])**2) for k in P))); den=float(tf.sqrt(sum(tf.reduce_sum(P0[k]**2) for k in P))); return num/(den+1e-9)

IMG_T=imgs.reshape(N,-1).astype("float32"); TXT_T=toks_oh.reshape(N,-1).astype("float32")

def run_cell(move_lvl, wname, DIMS, anti, LR, seed):
    c=enc_cfg(WMUL); P=build(c,DIMS,seed); P0={k:tf.identity(v) for k,v in P.items()}; ops=make_ops(P,c,DIMS,anti)
    NPc=int(sum(int(np.prod(v.shape)) for v in P.values()))
    rs=np.random.RandomState(seed+7); Fhist=[]; diverged=False; lrt=tf.constant(LR,tf.float32); t0=time.time()
    for s in range(STEPS):
        bi=rs.randint(0,N,size=B)
        x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); igt=tf.constant(IMG_T[bi]); tgt=tf.constant(TXT_T[bi])
        it,tt=ops["get_taps"](x,tk)
        Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
        F,mxw=ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,lrt); F=float(F); mxw=float(mxw); Fhist.append(F)
        if not (np.isfinite(F) and mxw<DIVERGE_W): diverged=True; break
    move=movement(P,P0)
    # ---- batched generation read-outs over all N ----
    if not diverged:
        it_all,tt_all=ops["get_taps"](tf.constant(imgs),tf.constant(toks))
        St=ops["relax_mono"]([tf.identity(tt_all[k]) for k in range(NS)], tt_all, ops["dec_txt"], tf.constant(TXT_T), GEN_INFER)
        t2i=ops["dec_img"](St).numpy().reshape(N,RES,RES,CH)
        Si=ops["relax_mono"]([tf.identity(it_all[k]) for k in range(NS)], it_all, ops["dec_img"], tf.constant(IMG_T), GEN_INFER)
        i2i=ops["dec_img"](Si).numpy().reshape(N,RES,RES,CH)
        diversity=float(np.mean(np.std(t2i,0))/(DATA_STD+1e-9))
        out_range=float((t2i.max(0)-t2i.min(0)).mean())
        d=((t2i.reshape(N,1,-1)-imgs.reshape(1,N,-1))**2).mean(-1); retr=float(np.mean(np.argmin(d,1)==np.arange(N)))
        recon=float(np.mean((i2i-imgs)**2))
    else:
        t2i=np.zeros((N,RES,RES,CH)); diversity=out_range=retr=recon=float("nan")
    collapsed = (not np.isfinite(diversity)) or (diversity < TAU)
    void = (move_lvl=="HIGH") and (not diverged) and (move < MOVE_MIN)
    print(f"  [{move_lvl}/{wname}/{'anti' if anti>0 else 'noanti'}] {NPc/1e6:.0f}M LR={LR:.0e}: F {Fhist[0]:.2e}->{Fhist[-1]:.2e} | "
          f"move={move*100:5.1f}% div={diversity:.3f} retr={retr:.4f}(ch {1/N:.4f}) out-rng={out_range:.2e} recon={recon:.4f} "
          f"| {'DIVERGED' if diverged else ('VOID' if void else ('COLLAPSE' if collapsed else 'VARIES'))} ({time.time()-t0:.0f}s)",flush=True)
    return dict(move_lvl=move_lvl,wname=wname,anti=float(anti),LR=float(LR),DIMS=list(DIMS),params=NPc,seed=seed,
                move=move,diversity=diversity,retr=retr,out_range=out_range,recon=recon,collapsed=bool(collapsed),
                diverged=bool(diverged),void=bool(void),F0=Fhist[0],Fend=Fhist[-1],Fhist=Fhist,t2i=t2i[:10])

print(f"recipe: GELU, plain LARS + bias floor, relax({N_INFER})-then-step, dense anchors, A_GEN={A_GEN}>=A_CROSS={A_CROSS}, mini-batch in-batch InfoNCE",flush=True)
print(f"axes: latent wide={wide_dims(WMUL)} narrow={[max(4,d//NARROW_DIV) for d in wide_dims(WMUL)]} | anti off/on | move LOW(lr={LR_LOW:.0e})/HIGH(lr={LR_HIGH:.0e}) | STEPS={STEPS} seeds={SEEDS} TAU={TAU} MOVE_MIN={MOVE_MIN}",flush=True)

WIDE=wide_dims(WMUL); NARROW=[max(4,d//NARROW_DIV) for d in WIDE]
allcells=[]
for seed in SEEDS:
    print(f"\n################## SEED {seed} ##################",flush=True)
    for move_lvl,LR in [("LOW",LR_LOW),("HIGH",LR_HIGH)]:
        for wname,DIMS in [("narrow",NARROW),("wide",WIDE)]:
            for anti in [0.0,ANTI]:
                allcells.append(run_cell(move_lvl,wname,DIMS,anti,LR,seed))

# aggregate over seeds (mean) keyed by (move_lvl,wname,anti)
def agg(ml,wn,an):
    cs=[r for r in allcells if r["move_lvl"]==ml and r["wname"]==wn and r["anti"]==an and not r["diverged"]]
    if not cs: return dict(move=float("nan"),diversity=float("nan"),retr=float("nan"),out_range=float("nan"),recon=float("nan"),void=True,collapsed=True,n=0,t2i=allcells[0]["t2i"])
    m=lambda key: float(np.mean([c[key] for c in cs]))
    return dict(move=m("move"),diversity=m("diversity"),retr=m("retr"),out_range=m("out_range"),recon=m("recon"),
                void=any(c["void"] for c in cs),collapsed=(m("diversity")<TAU),n=len(cs),t2i=cs[0]["t2i"])
cells={(ml,wn,an):agg(ml,wn,an) for ml in ("LOW","HIGH") for wn in ("narrow","wide") for an in (0.0,ANTI)}

# ============================ FIGURES ============================
# (1) matrix heatmap: diversity, LOW vs HIGH panels (latent x anti), annotated
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for ax,lvl in zip(axs,["LOW","HIGH"]):
    M=np.array([[cells[(lvl,w,a)]["diversity"] for a in (0.0,ANTI)] for w in ("narrow","wide")])
    im=ax.imshow(np.nan_to_num(M),vmin=0,vmax=max(0.4,np.nanmax(list(c["diversity"] for c in cells.values() if np.isfinite(c["diversity"]))+[0.4])),cmap="viridis")
    ax.set_xticks([0,1],["anti-off","anti-on"]); ax.set_yticks([0,1],["narrow","wide"]); ax.set_title(f"{lvl} weight movement")
    for i,w in enumerate(("narrow","wide")):
        for j,a in enumerate((0.0,ANTI)):
            c=cells[(lvl,w,a)]
            ax.text(j,i,f"div={c['diversity']:.2f}\nretr={c['retr']:.3f}\nmove={c['move']*100:.0f}%\nout={c['out_range']:.1e}\n{'VOID' if c['void'] else ('COLLAPSE' if c['collapsed'] else 'VARIES')}",
                    ha="center",va="center",color="white" if (not np.isfinite(c['diversity']) or c['diversity']<0.25) else "black",fontsize=7)
    fig.colorbar(im,ax=ax,fraction=0.046)
plt.suptitle(f"Step2 COCO dissociation: text->image diversity (collapse iff <{TAU}; chance retr {1/N:.4f}). Only LOW->HIGH should flip it.",fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"step2_dissoc_matrix.png"),dpi=130); plt.close()

# (2) sample grid: targets, then collapsed-LOW vs responsive-HIGH at wide+anti-on (the two that "should" rescue but don't without movement)
nc=min(10,N)
key=[("targets",None),
     ("LOW/wide/anti-on", cells[("LOW","wide",ANTI)]),
     ("HIGH/wide/anti-on", cells[("HIGH","wide",ANTI)]),
     ("LOW/narrow/anti-off", cells[("LOW","narrow",0.0)]),
     ("HIGH/narrow/anti-off", cells[("HIGH","narrow",0.0)])]
fig,axes=plt.subplots(len(key),nc,figsize=(1.3*nc,1.5*len(key)))
for ri,(name,c) in enumerate(key):
    row=imgs[:nc] if c is None else np.clip(c["t2i"][:nc],0,1)
    for j in range(nc):
        axes[ri,j].imshow(np.clip(row[j],0,1)); axes[ri,j].axis("off")
    tag=name if c is None else f"{name} div={c['diversity']:.2f} retr={c['retr']:.3f} {'VOID' if c['void'] else ('COLLAPSE' if c['collapsed'] else 'VARIES')}"
    axes[ri,0].set_title(tag,fontsize=6,loc="left")
plt.suptitle("text->image (top=targets). wide+anti-on does NOT rescue at LOW movement; HIGH movement responds even narrow+no-anti.",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"step2_dissoc_samples.png"),dpi=120); plt.close()

# (3) F trajectories: LOW (collapsed) vs HIGH (responsive) at narrow/anti-off -- F drops in BOTH
loc=[r for r in allcells if r["move_lvl"]=="LOW" and r["wname"]=="narrow" and r["anti"]==0.0][0]
hic=[r for r in allcells if r["move_lvl"]=="HIGH" and r["wname"]=="narrow" and r["anti"]==0.0][0]
fig,ax=plt.subplots(figsize=(6.5,4))
ax.plot(loc["Fhist"],label=f"LOW move ({loc['move']*100:.0f}%) -> {'COLLAPSE' if loc['collapsed'] else 'VARIES'} (div {loc['diversity']:.2f})",color="C3")
ax.plot(hic["Fhist"],label=f"HIGH move ({hic['move']*100:.0f}%) -> {'VARIES' if not hic['collapsed'] else 'COLLAPSE'} (div {hic['diversity']:.2f})",color="C2")
ax.set_xlabel("step"); ax.set_ylabel("energy F"); ax.set_yscale("log"); ax.set_title("F drops in BOTH cells -> F-descent does NOT track collapse"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE,"step2_dissoc_Ftraj.png"),dpi=130); plt.close()

# ============================ VERDICT ============================
def broke(c): return (np.isfinite(c["diversity"])) and (not c["collapsed"]) and (c["retr"]>3.0/N) and (not c["void"])
low=[(k,v) for k,v in cells.items() if k[0]=="LOW"]; high=[(k,v) for k,v in cells.items() if k[0]=="HIGH"]
low_broke=[k for k,v in low if broke(v)]
high_valid=[(k,v) for k,v in high if not v["void"]]
high_broke=[k for k,v in high_valid if broke(v)]
any_void=[k for k,v in high if v["void"]]
claim_holds=(len(low_broke)==0) and (len(high_valid)>0) and (len(high_broke)==len(high_valid))

dump=dict(config=dict(smoke=SMOKE,N=N,RES=RES,CAPLEN=CAPLEN,V=V,B=B,WMUL=WMUL,wide=WIDE,narrow=NARROW,
                      LR_LOW=LR_LOW,LR_HIGH=LR_HIGH,STEPS=STEPS,seeds=SEEDS,TAU=TAU,MOVE_MIN=MOVE_MIN,chance=1/N),
          cells={f"{k[0]}/{k[1]}/{'anti' if k[2]>0 else 'noanti'}":{kk:vv for kk,vv in v.items() if kk!="t2i"} for k,v in cells.items()},
          per_run=[{kk:vv for kk,vv in r.items() if kk not in ("Fhist","t2i")} for r in allcells],
          verdict=dict(low_broke=[f"{k[0]}/{k[1]}/{k[2]}" for k in low_broke],high_broke=[f"{k[0]}/{k[1]}/{k[2]}" for k in high_broke],
                       high_void=[f"{k[0]}/{k[1]}/{k[2]}" for k in any_void],claim_holds=bool(claim_holds)))
with open(os.path.join(HERE,"step2_dissoc_results.json"),"w") as fh: json.dump(dump,fh,indent=2)

print("\n==================== STEP 2 DISSOCIATION VERDICT (real COCO) ====================",flush=True)
print(f"  chance retr={1/N:.4f}; collapse iff diversity<{TAU}; HIGH cell VOID if move<{MOVE_MIN*100:.0f}%",flush=True)
if any_void: print(f"  WARNING: VOID HIGH cells (undertrained, move<40%): {[f'{k[0]}/{k[1]}/{k[2]}' for k in any_void]} -- raise STEPS and re-run",flush=True)
print(f"  LOW cells that BROKE collapse (should be NONE): {[f'{k[1]}/{k[2]}' for k in low_broke] or 'NONE'}",flush=True)
print(f"  HIGH (valid) cells that BROKE collapse (should be ALL {len(high_valid)}): {len(high_broke)}/{len(high_valid)}",flush=True)
if claim_holds:
    print("  VERDICT: DISSOCIATION HOLDS ON REAL DATA -- only weight-movement flips mode-collapse; latent width and anti-mean do NOT (at LOW movement they still collapse; at HIGH movement it responds regardless).",flush=True)
elif any_void:
    print("  VERDICT: INCONCLUSIVE -- a HIGH cell was VOID (undertrained). Raise STEPS so all HIGH cells clear 40% movement, then re-judge.",flush=True)
elif low_broke:
    print(f"  VERDICT: DISSOCIATION FAILS/PARTIAL ON REAL DATA -- a LOW-movement cell broke collapse ({[f'{k[1]}/{k[2]}' for k in low_broke]}), so width/anti-mean ALSO flips it. Claim does NOT generalize beyond MNIST. Reported honestly.",flush=True)
else:
    print("  VERDICT: PARTIAL -- not all HIGH cells responded; see table.",flush=True)
print("saved: step2_dissoc_matrix.png, step2_dissoc_samples.png, step2_dissoc_Ftraj.png, step2_dissoc_results.json",flush=True)
if SMOKE: print("\n[SMOKE] mechanics-only (synthetic, tiny config, ~12 steps); numbers meaningless -- only confirms the 8-cell loop, COCO pipeline, per-cell VOID logic, metrics, heatmap, sample grid, F-traj, and verdict run end-to-end.",flush=True)
