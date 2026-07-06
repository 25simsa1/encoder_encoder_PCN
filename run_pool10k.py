"""A1 -- enlarged held-out gallery readout. One fixed 10k gallery of never-trained train2017 images
(extended-cache indices [108000:118000], disjoint from every training set in the paper: all banked runs
permuted only [0:22000], the extended BP ladder permutes only [0:108000]). For each key checkpoint we run
the SAME forward-only latent-retrieval readout on this one gallery, so PC-at-chance and BP-transfers are
both measured at 5x the resolution of the 2000-pool and are exactly comparable across systems.

Each checkpoint is scored with ITS OWN train vocabulary, reconstructed from its training split over the
shared 22k prefix (N_HAVE=22000, perm=RandomState(seed+1)[:N_TRAIN]); gallery captions use that vocab
(unseen chars -> null), exactly as the checkpoint's own readout did. Handles both checkpoint layouts:
.npz (run_coupling_scale / E1) and the streamed .dir (LOWHOST capacity runs). Frees each model before the
next so the 3B checkpoint and the 156M ones can share one job.

ENV: RUNS1_DATA(~/coco_scale) RUNS1_COCO(train2017) POOL_START(108000) POOL_N(10000) READB(64)
POOL_SPECS(semicolon name=path:seed:ntrain list; a default covers the standard locations, missing files
skipped loudly). OUT: pool10k_results.json (per system hits/gallery + exact-binomial upper tail).
"""
import os, json, time, math
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import numpy as np, tensorflow as tf
HOME=os.path.expanduser("~"); HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("RUNS1_DATA",os.path.join(HOME,"coco_scale")); COCO=os.environ.get("RUNS1_COCO","train2017")
POOL_START=int(os.environ.get("POOL_START",108000)); POOL_N=int(os.environ.get("POOL_N",10000))
READB=int(os.environ.get("READB",64)); PREFIX=22000
RES,CAPLEN,NS,HEADS,NBLK=64,64,4,4,4
DEFAULT=";".join([
  f"PC8k_s0={HOME}/runs/8k_150ep/cs_A_seed0.npz:0:8000",
  f"PC8k_s1={HOME}/runs/8k_150ep_s1/cs_A_seed1.npz:1:8000",
  f"PC8k_s2={HOME}/runs/8k_150ep_s2/cs_A_seed2.npz:2:8000",
  f"PC20k_s0={HOME}/runs/20k_150ep/cs_A_seed0.npz:0:20000",
  f"PC20k_s1={HOME}/runs/20k_150ep_s1/cs_A_seed1.npz:1:20000",
  f"PC20k_s2={HOME}/runs/20k_150ep_s2/cs_A_seed2.npz:2:20000",
  f"BP_E1_8k={HOME}/runs/geom_ckpts/e1_seed0.npz:0:8000",
  f"E1L_8k={HOME}/runs/geom_ckpts/e1l_seed0.npz:0:8000",
  f"BPonF_8k={HOME}/runs/geom_ckpts/bpf_seed0.npz:0:8000",
  f"PC_3B={HOME}/runs/cap3be/cap_A_w6.59_seed0.dir:0:8000",
  # 20k BP/E1L on the gallery come from item D's gallery-eval runs (no old 20k checkpoint survived); A1
  # covers the surviving checkpoints: PC 8k/20k all seeds, BP E1 8k, E1L 8k, BPonF 8k, and 3B.
])
SPECS=[]
for e in os.environ.get("POOL_SPECS",DEFAULT).split(";"):
    nm,rest=e.split("="); path,seed,nt=rest.rsplit(":",2); SPECS.append((nm,path,int(seed),int(nt)))

def gelu(z): return tf.nn.gelu(z)
def build_vocab(caps): ch=sorted(set("".join(caps))|{"\0"}); return ch,{c:i for i,c in enumerate(ch)}
def encode_caps(caps,c2i,cl):
    nul=c2i["\0"]; t=np.full((len(caps),cl),nul,"int32")
    for n,cp in enumerate(caps):
        for j in range(cl):
            if j<len(cp): t[n,j]=c2i.get(cp[j],nul)
    return t
def load_ckpt(path):
    if path.endswith(".dir"):
        return {os.path.splitext(f)[0]:tf.constant(np.load(os.path.join(path,f))) for f in os.listdir(path) if f.endswith(".npy")}
    z=np.load(path); return {k:tf.constant(z[k]) for k in z.files}
def make(P):
    DM=int(P["emb"].shape[1]); HEAD=DM//HEADS
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
    l2n=lambda z: z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))
    return latents

def binom_sf(hits,n,p):
    # P(X >= hits), exact where scipy is present, else Poisson(np) upper tail (np=1 here, excellent)
    try:
        from scipy.stats import binom; return float(binom.sf(hits-1,n,p))
    except Exception:
        lam=n*p; s=0.0; term=math.exp(-lam)
        for i in range(0,hits): s+=term; term*=lam/(i+1)
        return float(max(0.0,1.0-s))

# gallery images/captions come from the EXTENDED cache (positions >=108000, disjoint from all training
# by id-position separation). Old-checkpoint vocabularies come from the ORIGINAL 22k cache directly, so
# they are exact regardless of any ext-prefix drift.
imgs=np.load(os.path.join(DATA,f"imgs_sc_{COCO}_ext.npy"),mmap_mode="r")
ext_caps=open(os.path.join(DATA,f"caps_sc_{COCO}_ext.txt")).read().split("\n")[:imgs.shape[0]]
NHAVE_EXT=imgs.shape[0]
caps_orig=open(os.path.join(DATA,f"caps_sc_{COCO}.txt")).read().split("\n")   # original 22k captions for vocab
NHAVE_ORIG=len(caps_orig)
g_end=min(POOL_START+POOL_N,NHAVE_EXT); g_idx=np.arange(POOL_START,g_end)
assert POOL_START>=PREFIX and POOL_START>=NHAVE_ORIG, "gallery must start beyond every training id-position"
gallery_imgs=np.asarray(imgs[POOL_START:g_end],dtype="float32")             # materialize the 10k gallery once
gallery_caps=[ext_caps[i] for i in range(POOL_START,g_end)]
print(f"[data] ext cache {NHAVE_EXT} | orig cache {NHAVE_ORIG} | gallery [{POOL_START}:{g_end}] = {len(g_idx)} imgs (chance {1/len(g_idx):.2e})",flush=True)

results={}
for nm,path,seed,ntrain in SPECS:
    if not os.path.exists(path): print(f"!! SKIP {nm}: missing {path}",flush=True); continue
    t0=time.time(); P=load_ckpt(path); latents=make(P)
    perm=np.random.RandomState(seed+1).permutation(NHAVE_ORIG); tr=perm[:ntrain]  # checkpoint's own train split over the ORIGINAL cache
    chars,c2i=build_vocab([caps_orig[i] for i in tr]); toks=encode_caps(gallery_caps,c2i,CAPLEN)
    ZIl=[];ZTl=[]
    for st in range(0,len(g_idx),READB):
        xb=tf.constant(gallery_imgs[st:st+READB]); tb=tf.constant(toks[st:st+READB])
        ZI,ZT=latents(xb,tb); ZIl.append(ZI.numpy()); ZTl.append(ZT.numpy())
    ZI=np.concatenate(ZIl); ZT=np.concatenate(ZTl); M=len(g_idx)
    # text->image top-1 over the full gallery, chunked argmax to avoid an MxM matrix
    hits=0
    for st in range(0,M,512):
        blk=ZT[st:st+512]@ZI.T; hits+=int(np.sum(np.argmax(blk,1)==np.arange(st,st+len(blk))))
    p=1.0/M; sf=binom_sf(hits,M,p); sigma=(hits-M*p)/math.sqrt(M*p*(1-p))
    results[nm]=dict(path=path,seed=seed,n_train=ntrain,gallery=M,hits=hits,chance=p,sigma=sigma,binom_sf=sf,V=len(chars))
    print(f"[{nm}] ({time.time()-t0:.0f}s) hits {hits}/{M} (chance {M*p:.1f}, {sigma:+.1f} sigma, P(>= )={sf:.2e}) V={len(chars)}",flush=True)
    del P; tf.keras.backend.clear_session()

with open(os.path.join(HERE,"pool10k_results.json"),"w") as fh:
    json.dump(dict(config=dict(gallery_start=POOL_START,gallery_n=len(g_idx),coco=COCO),systems=results),fh,indent=2)
print(f"saved: pool10k_results.json ({len(results)} systems)",flush=True); print("JOB_OK_pool10k",flush=True)
