"""A2 -- SETTLING CURVE: the last un-run inference control. The primary endpoint (latent retrieval) is
forward-only and therefore immune to the relaxation depth; this covers the SECONDARY generation-side
metrics, which DO relax the shared latent (relax_mono, GEN_INFER=25 at the operating point). We (i) trace
the per-step relative update norm of the readout relaxation to depth 100, showing it has converged well
before 25, and (ii) recompute the generation metrics at T in {5,10,25,50,100} so any residual T-sensitivity
is visible. Cites Pinchetti/Frieder 2601.20895 (initialization/inference-budget analysis for PCNs).

Byte-matched to run_coupling_scale.py readouts except that GEN_INFER is swept and the relaxation is
instrumented to return its per-step update norm. Reconstructs each checkpoint's own held-out split over
the shared 22k cache (N_HAVE=22000, perm=RandomState(seed+1)), so it reads the exact eval pool the
checkpoint was scored on.

ENV: RUNS1_DATA(~/coco_scale) RUNS1_COCO(train2017) SETTLE_CKPTS(semicolon name=path:seed list;
default = the three 8k PC arm-A seeds) SETTLE_T(5,10,25,50,100) SETTLE_DEPTH(100) RUNS1_READB(128)
RUNS1_READTRAIN(1500). OUT: settling_results.json.
"""
import os, json, time, math
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HOME=os.path.expanduser("~"); HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("RUNS1_DATA", os.path.join(HOME,"coco_scale"))
COCO=os.environ.get("RUNS1_COCO","train2017")
READB=int(os.environ.get("RUNS1_READB",128)); READTRAIN=int(os.environ.get("RUNS1_READTRAIN",1500))
T_LIST=[int(t) for t in os.environ.get("SETTLE_T","5,10,25,50,100").split(",")]
DEPTH=int(os.environ.get("SETTLE_DEPTH",100))
RES,CAPLEN,NS,HEADS,NBLK,CODE=64,64,4,4,4,16
A_CROSS,A_GEN,REL_C=1.0,2.0,0.05
N_TRAIN,N_EVAL,N_HAVE=8000,2000,22000                                       # the 8k runs' split geometry over the shared 22k cache
DEFAULT=";".join([f"PC8k_seed0={HOME}/runs/8k_150ep/cs_A_seed0.npz:0",
                  f"PC8k_seed1={HOME}/runs/8k_150ep_s1/cs_A_seed1.npz:1",
                  f"PC8k_seed2={HOME}/runs/8k_150ep_s2/cs_A_seed2.npz:2"])
SPECS=[]
for e in os.environ.get("SETTLE_CKPTS",DEFAULT).split(";"):
    nm,rest=e.split("="); path,seed=rest.rsplit(":",1); SPECS.append((nm,path,int(seed)))

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps,[tf.shape(eps)[0],-1])**2,axis=1)
def build_vocab(caps): ch=sorted(set("".join(caps))|{"\0"}); return ch,{c:i for i,c in enumerate(ch)}
def encode_caps(caps,c2i,cl):
    nul=c2i["\0"]; t=np.full((len(caps),cl),nul,"int32")
    for n,cp in enumerate(caps):
        for j in range(cl):
            if j<len(cp): t[n,j]=c2i.get(cp[j],nul)
    return t

imgs=np.load(os.path.join(DATA,f"imgs_sc_{COCO}.npy")); caps=open(os.path.join(DATA,f"caps_sc_{COCO}.txt")).read().split("\n")[:len(imgs)]
assert len(imgs)>=N_HAVE, f"need the 22k shared cache, have {len(imgs)}"
print(f"[data] {imgs.shape}",flush=True)

def make(P):
    DM=int(P["emb"].shape[1]); HEAD=DM//HEADS; DIMS=[int(P[f"Wt{b}"].shape[1]) for b in range(NBLK)]
    betas=[REL_C*d for d in DIMS]
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
    def relax_trace(S,taps,decfn,tgt,n):
        # returns relaxed S plus the per-step relative update norm ||beta*grad|| / (||S||+eps) (global Frobenius)
        Sv=[tf.identity(s) for s in S]; rel=[]
        for _ in range(n):
            with tf.GradientTape() as tp:
                tp.watch(Sv); f=0.5*tf.reduce_sum(tf.add_n([mse(Sv[k]-taps[k]) for k in range(NS)])+A_GEN*mse(decfn(Sv)-tgt))
            gr=tp.gradient(f,Sv)
            upd=[betas[k]*gr[k] for k in range(NS)]
            un=float(tf.sqrt(tf.add_n([tf.reduce_sum(u**2) for u in upd])))
            sn=float(tf.sqrt(tf.add_n([tf.reduce_sum(s**2) for s in Sv])))
            rel.append(un/(sn+1e-9)); Sv=[Sv[k]-upd[k] for k in range(NS)]
        return Sv,rel
    def l2n(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))
    return dict(enc_img=enc_img,enc_txt=enc_txt,dec_img=dec_img,dec_txt=dec_txt,relax_trace=relax_trace,latents=latents,betas=betas)

def gen_metrics(ops,idx,toks,toks_oh,T):
    M=len(idx); t2i=np.zeros((M,RES,RES,3)); i2i=np.zeros((M,RES,RES,3)); i2t_h=0; i2t_n=0; relstack=[]
    for st in range(0,M,READB):
        bi=[int(idx[j]) for j in range(st,min(st+READB,M))]
        x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); it=ops["enc_img"](x); tt=ops["enc_txt"](tk)
        igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
        St,rel=ops["relax_trace"]([tf.identity(tt[k]) for k in range(NS)],tt,ops["dec_txt"],tgt,T)
        if st==0: relstack=rel                                              # first batch's per-step trace (T==DEPTH call)
        t2i[st:st+len(bi)]=ops["dec_img"](St).numpy().reshape(len(bi),RES,RES,3)
        Si,_=ops["relax_trace"]([tf.identity(it[k]) for k in range(NS)],it,ops["dec_img"],igt,T)
        i2i[st:st+len(bi)]=ops["dec_img"](Si).numpy().reshape(len(bi),RES,RES,3)
        pred=ops["dec_txt"](Si).numpy().reshape(len(bi),CAPLEN,-1).argmax(-1)
        i2t_h+=int((pred==toks[bi]).sum()); i2t_n+=pred.size
    real=imgs[idx]
    A=t2i.reshape(M,-1).astype("float32"); Bm=real.reshape(M,-1).astype("float32"); Bn=(Bm**2).sum(1)
    nn=np.empty(M,"int64")
    for st in range(0,M,256): nn[st:st+256]=np.argmin(Bn[None,:]-2.0*(A[st:st+256]@Bm.T),1)
    retr=float(np.mean(nn==np.arange(M)))
    recon=float(np.mean((i2i-real)**2)); base=float(np.mean((imgs[idx].mean(0)[None]-real)**2))
    return dict(T=T,retr=retr,hits=int(round(retr*M)),recon=recon,recon_base=base,
                diversity=float(np.mean(np.std(t2i,0))/(np.std(real)+1e-9)),i2t=i2t_h/max(1,i2t_n)), relstack

results={}
for nm,path,seed in SPECS:
    if not os.path.exists(path): print(f"!! SKIP {nm}: missing {path}",flush=True); continue
    t0=time.time(); z=np.load(path); P={k:tf.constant(z[k]) for k in z.files}; ops=make(P)
    perm=np.random.RandomState(seed+1).permutation(N_HAVE); ev=perm[N_TRAIN:N_TRAIN+N_EVAL]; tr=perm[:N_TRAIN]
    chars,c2i=build_vocab([caps[i] for i in tr]); toks=encode_caps(caps,c2i,CAPLEN)
    toks_oh=tf.one_hot(toks,len(chars)).numpy().astype("float32")
    # forward-only lat_retr (T-invariant positive control)
    ZIl=[];ZTl=[]
    for st in range(0,len(ev),READB):
        bi=ev[st:st+READB]; ZI,ZT=ops["latents"](tf.constant(imgs[bi]),tf.constant(toks[bi])); ZIl.append(ZI.numpy());ZTl.append(ZT.numpy())
    ZI=np.concatenate(ZIl);ZT=np.concatenate(ZTl); lat=int(np.sum(np.argmax(ZT@ZI.T,1)==np.arange(len(ev))))
    per_T={}; trace=None
    for T in sorted(set(T_LIST+[DEPTH])):
        m,rel=gen_metrics(ops,ev,toks,toks_oh,T)
        if T==DEPTH: trace=rel                                              # full-depth per-step update-norm trace
        if T in T_LIST: per_T[T]=m
    results[nm]=dict(seed=seed,n_eval=len(ev),lat_retr_forward_hits=lat,chance=1.0/len(ev),
                     update_norm_trace_depth=DEPTH,update_norm_trace=trace,gen_by_T=per_T)
    knee=next((i+1 for i,r in enumerate(trace) if r<1e-2),None)
    print(f"[{nm}] ({time.time()-t0:.0f}s) fwd lat_retr {lat}/{len(ev)} | update-norm to <1e-2 by step {knee} | "
          f"retr@T: {{{', '.join(f'{T}:{per_T[T][\"hits\"]}' for T in T_LIST)}}} recon@25={per_T.get(25,{}).get('recon')}",flush=True)

with open(os.path.join(HERE,"settling_results.json"),"w") as fh: json.dump(dict(config=dict(T_list=T_LIST,depth=DEPTH,n_eval=N_EVAL,cite="2601.20895"),systems=results),fh,indent=2)
print(f"saved: settling_results.json ({len(results)} systems)",flush=True); print("JOB_OK_settling",flush=True)
