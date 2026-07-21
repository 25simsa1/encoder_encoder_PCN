"""B equivalence gate, the rigorous version: prove tf.recompute_grad gives the SAME weight gradients as
the plain eager weight step, at real scale, on identical inputs. The full-run gate (gate_eager vs
gate_recomp) compares scientific endpoints but their weight-movement differs (121 vs 159 percent) because
the recomputed encoder forward has GPU-reduction-order fp noise that compounds chaotically over 9450
steps. That is trajectory chaos, not a math difference. This probe removes the chaos: one build, one batch,
one relaxed (detached) S, then the weight gradient computed BOTH ways from the identical state, compared
per tensor and globally (cosine + relative L2). Equivalence up to fp reduction order is the claim; the
digit-identical smoke already showed it when fully deterministic, this shows it at 156M and (if it fits) 3B.

ENV: GATE_WMUL(1.5) RUNS1_DATA RUNS1_COCO(train2017) RUNS1_NTRAIN(8000) RUNS1_BATCHJ(128) RUNS1_NINFER(8).
OUT: recompute_gate_w{WMUL}.json + printed per-tensor worst offenders. PASS: global cosine > 0.9999 and
global relative L2 < 1e-4.
"""
import os, json, time, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNS1_GPU","0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import numpy as np, tensorflow as tf
HOME=os.path.expanduser("~"); HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("RUNS1_DATA",os.path.join(HOME,"coco_scale")); COCO=os.environ.get("RUNS1_COCO","train2017")
WMUL=float(os.environ.get("GATE_WMUL",1.5)); N_TRAIN=int(os.environ.get("RUNS1_NTRAIN",8000))
BATCHJ=int(os.environ.get("RUNS1_BATCHJ",128)); N_INFER=int(os.environ.get("RUNS1_NINFER",8)); SEED=0
RES,CAPLEN,NS,HEADS,NBLK,CODE=64,64,4,4,4,16
A_CROSS,A_GEN,REL_C,DEC_SD=1.0,2.0,0.05,1e-3
B_C1,B_C2,B_C3,B_C4,B_BN=32,64,128,256,512; B_DM,B_FFN=512,1024; B_DIMS=[2048,2048,1024,1024]
CH=3; PIX=RES*RES*CH
def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps,[tf.shape(eps)[0],-1])**2,axis=1)
def build_vocab(caps): ch=sorted(set("".join(caps))|{"\0"}); return ch,{c:i for i,c in enumerate(ch)}
def encode_caps(caps,c2i,cl):
    nul=c2i["\0"]; t=np.full((len(caps),cl),nul,"int32")
    for n,cp in enumerate(caps):
        for j in range(cl):
            if j<len(cp): t[n,j]=c2i.get(cp[j],nul)
    return t
imgs=np.load(os.path.join(DATA,f"imgs_sc_{COCO}.npy"),mmap_mode="r"); caps=open(os.path.join(DATA,f"caps_sc_{COCO}.txt")).read().split("\n")[:imgs.shape[0]]
N_HAVE=imgs.shape[0]; perm=np.random.RandomState(SEED+1).permutation(N_HAVE); tr=perm[:N_TRAIN]
chars,c2i=build_vocab([caps[i] for i in tr]); V=len(chars); toks=encode_caps(caps,c2i,CAPLEN); toks_oh=tf.one_hot(toks,V).numpy().astype("float32")
def cfg(w):
    r=lambda x:max(4,int(round(x*w))); DM=r(B_DM); DM-=DM%HEADS
    return dict(DM=max(HEADS,DM),C1=r(B_C1),C2=r(B_C2),C3=r(B_C3),C4=r(B_C4),BN=r(B_BN),DIMS=[r(d) for d in B_DIMS],FFN=r(B_FFN),HEAD=max(1,(max(HEADS,DM))//HEADS))
def build(w,seed):
    c=cfg(w); DM,C1,C2,C3,C4,BN,DIMS,FFN=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"]
    s2=RES//4;s3=RES//8;s4=RES//16;f0d,f1d,f2d=s2*s2*C2,s3*s3*C3,s4*s4*C4; g=tf.random.Generator.from_seed(seed)
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
    return P,c
P,c=build(WMUL,SEED); DM,DIMS,HEAD=c["DM"],c["DIMS"],c["HEAD"]; betas=[REL_C*d for d in DIMS]; NAMES=list(P.keys()); ALL_W=list(P.values())
NP=int(sum(int(np.prod(v.shape)) for v in P.values())); print(f"=== RECOMPUTE GATE === wmul={WMUL} params={NP/1e9:.3f}B batch={BATCHJ}",flush=True)
def enc_img(x):
    h=gelu(tf.nn.conv2d(x,P["c1"],1,"SAME")+P["cb1"]);h=tf.nn.max_pool2d(h,2,2,"SAME")
    h=gelu(tf.nn.conv2d(h,P["c2"],1,"SAME")+P["cb2"]);h=tf.nn.max_pool2d(h,2,2,"SAME");f0=tf.reshape(h,[tf.shape(x)[0],-1])
    h=gelu(tf.nn.conv2d(h,P["c3"],1,"SAME")+P["cb3"]);h=tf.nn.max_pool2d(h,2,2,"SAME");f1=tf.reshape(h,[tf.shape(x)[0],-1])
    h=gelu(tf.nn.conv2d(h,P["c4"],1,"SAME")+P["cb4"]);h=tf.nn.max_pool2d(h,2,2,"SAME");f2=tf.reshape(h,[tf.shape(x)[0],-1]);f3=gelu(f2@P["wbn"]+P["bbn"])
    return [gelu(f0@P["Wi0"]+P["bi0"]),gelu(f1@P["Wi1"]+P["bi1"]),gelu(f2@P["Wi2"]+P["bi2"]),gelu(f3@P["Wi3"]+P["bi3"])]
def enc_txt(tk):
    B=tf.shape(tk)[0];x=tf.gather(P["emb"],tk)+P["pos"][None];tt=[]
    for b in range(NBLK):
        q,k_,v=x@P[f"Wq{b}"],x@P[f"Wk{b}"],x@P[f"Wv{b}"]
        sp=lambda t: tf.transpose(tf.reshape(t,[B,CAPLEN,HEADS,HEAD]),[0,2,1,3])
        a=tf.nn.softmax(tf.matmul(sp(q),sp(k_),transpose_b=True)/np.sqrt(HEAD),axis=-1)
        ctx=tf.reshape(tf.transpose(tf.matmul(a,sp(v)),[0,2,1,3]),[B,CAPLEN,DM])
        x=x+ctx@P[f"Wo{b}"];x=x+(gelu(x@P[f"f1_{b}"]+P[f"fb1_{b}"])@P[f"f2_{b}"]+P[f"fb2_{b}"])
        tt.append(gelu(tf.reduce_mean(x,1)@P[f"Wt{b}"]+P[f"bt{b}"]))
    return tt
def code_of(S): return tf.concat([gelu(S[k]@P[f"proj{k}"]) for k in range(NS)],axis=1)
def dec_img(S): return tf.nn.sigmoid(code_of(S)@P["W_DI"]+P["B_DI"])
def dec_txt(S): return code_of(S)@P["W_DT"]+P["B_DT"]
def F_energy(S,it,tt,igt,tgt):
    cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
    return 0.5*tf.reduce_mean(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
# one batch, relax S (detached) exactly like the driver
bi=tr[:BATCHJ]; x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi]); igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
it0,tt0=enc_img(x),enc_txt(tk); Sv=[0.5*(it0[k]+tt0[k]) for k in range(NS)]
for _ in range(N_INFER):
    with tf.GradientTape() as tp:
        tp.watch(Sv); f=0.5*tf.reduce_sum(tf.add_n([mse(Sv[k]-it0[k])+mse(Sv[k]-tt0[k]) for k in range(NS)])+A_GEN*(mse(dec_img(Sv)-igt)+mse(dec_txt(Sv)-tgt)))
    gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
Sc=tuple(tf.constant(z) for z in Sv)
def grads(recompute):
    ei=tf.recompute_grad(enc_img) if recompute else enc_img
    et=tf.recompute_grad(enc_txt) if recompute else enc_txt
    with tf.GradientTape() as t: t.watch(ALL_W); F=F_energy(Sc,ei(x),et(tk),igt,tgt)
    return float(F), t.gradient(F,ALL_W)
Fe,ge=grads(False); Fr,gr2=grads(True)
rows=[]; num=0.0; den=0.0; dot=0.0; ne=0.0; nr=0.0
for nm,a,b in zip(NAMES,ge,gr2):
    if a is None or b is None: continue
    a=tf.convert_to_tensor(a); b=tf.convert_to_tensor(b)
    d=float(tf.norm(a-b)); na=float(tf.norm(a)); nb=float(tf.norm(b))
    rel=d/(na+1e-12); rows.append((nm,rel,na)); num+=d*d; den+=na*na
    dot+=float(tf.reduce_sum(a*b)); ne+=na*na; nr+=nb*nb
gcos=dot/(math.sqrt(ne)*math.sqrt(nr)+1e-30); grel=math.sqrt(num)/(math.sqrt(den)+1e-30)
rows.sort(key=lambda r:-r[1])
print(f"F eager={Fe:.8e} recompute={Fr:.8e} relF={abs(Fe-Fr)/(abs(Fe)+1e-12):.2e}",flush=True)
print(f"GLOBAL grad cosine={gcos:.8f} relL2={grel:.2e}",flush=True)
print("worst per-tensor relL2:",[(nm,f"{rl:.1e}") for nm,rl,_ in rows[:6]],flush=True)
PASS=bool(gcos>0.9999 and grel<1e-4)
print(f"GATE {'PASS' if PASS else 'FAIL'} (criterion: cosine>0.9999 and relL2<1e-4)",flush=True)
json.dump(dict(wmul=WMUL,params=NP,F_eager=Fe,F_recompute=Fr,global_cosine=gcos,global_relL2=grel,
               worst=[(nm,rl) for nm,rl,_ in rows[:10]],passed=PASS),open(os.path.join(HERE,f"recompute_gate_w{WMUL}.json"),"w"),indent=2)
print(f"saved: recompute_gate_w{WMUL}.json | JOB_OK_recompute_gate",flush=True)
import sys as _sys; _sys.exit(0 if PASS else 1)   # nonzero on FAIL so an afterok flagship dependency will not launch on a failed gate
