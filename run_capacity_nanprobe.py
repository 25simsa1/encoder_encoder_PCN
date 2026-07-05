"""CAPACITY NaN PROBE -- localize the first non-finite site at large WMUL. Diagnosis only, no training,
no tuning. The 3B rung diverged at both pre-registered learning rates, and the retry's step-0 F was
already NaN, which reclassifies the failure as an init-time fp32 overflow in the forward, relaxation, or
weight-gradient path (learning-rate independent, nondeterministic with reduction order). This probe
walks the exact recipe stages on REAL COCO batches (the trigger is batch-dependent: the sorted trial
batch survived where a shuffled batch did not) and reports, per stage, max magnitudes and the first
tensor to go non-finite, plus the LARS update quantities (trust ratio, per-tensor update magnitude at
the ramped step-0 learning rate).

ENV: NP_WMUL(6.59) NP_BATCH(128) NP_NBATCH(4 batches: the trial batch then the first shuffled ones)
NP_LR(2e-2) plus the usual RUNS1_DATA/RUNS1_COCO/RUNS1_SEED/RUNS1_NTRAIN/RUNS1_NEVAL.
OUT: capacity_nanprobe_w{WMUL}.json plus the printed table.
"""
import os, sys, time, json, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNS1_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE   = os.path.dirname(os.path.abspath(__file__))
SEED   = int(os.environ.get("RUNS1_SEED", 0))
RES, CAPLEN = 64, 64
N_TRAIN= int(os.environ.get("RUNS1_NTRAIN", 8000))
N_EVAL = int(os.environ.get("RUNS1_NEVAL", 2000))
WMUL   = float(os.environ.get("NP_WMUL", 6.59))
BATCH  = int(os.environ.get("NP_BATCH", 128))
NBATCH = int(os.environ.get("NP_NBATCH", 4))
LR     = float(os.environ.get("NP_LR", 2e-2))
RAMP   = 300
DATA   = os.environ.get("RUNS1_DATA", "/root/coco_scale")
COCO   = os.environ.get("RUNS1_COCO", "train2017")

HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER = 8
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]
CH = 3; PIX = RES*RES*CH

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

def build_vocab(caps):
    chars = sorted(set("".join(caps)) | {"\0"}); return chars, {c:i for i,c in enumerate(chars)}
def encode_caps(caps, c2i, caplen):
    nul = c2i["\0"]; toks = np.full((len(caps), caplen), nul, "int32")
    for n,cp in enumerate(caps):
        for t in range(caplen):
            if t < len(cp): toks[n,t] = c2i.get(cp[t], nul)
    return toks

f_img,f_cap = os.path.join(DATA,f"imgs_sc_{COCO}.npy"), os.path.join(DATA,f"caps_sc_{COCO}.txt")
imgs=np.load(f_img); caps=open(f_cap).read().split("\n")[:len(imgs)]
N_HAVE=len(imgs)
perm = np.random.RandomState(SEED+1).permutation(N_HAVE)
tr_idx = perm[:N_TRAIN]
chars, c2i = build_vocab([caps[i] for i in tr_idx]); V = len(chars)
toks = encode_caps(caps, c2i, CAPLEN); toks_oh = tf.one_hot(toks, V).numpy().astype("float32")
print(f"=== NAN PROBE === wmul={WMUL} batch={BATCH} nbatch={NBATCH} V={V} lr={LR} (step-0 effective {LR/RAMP:.2e})",flush=True)

def cfg(wmul):
    r=lambda x:max(4,int(round(x*wmul)))
    DM=r(B_DM); DM-=DM%HEADS
    return dict(DM=max(HEADS,DM),C1=r(B_C1),C2=r(B_C2),C3=r(B_C3),C4=r(B_C4),BN=r(B_BN),
                DIMS=[r(d) for d in B_DIMS],FFN=r(B_FFN),HEAD=max(1,(max(HEADS,DM))//HEADS))

def build(wmul, seed):
    c=cfg(wmul); DM,C1,C2,C3,C4,BN,DIMS,FFN=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"]
    s2=RES//4; s3=RES//8; s4=RES//16; f0d,f1d,f2d=s2*s2*C2, s3*s3*C3, s4*s4*C4
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

P,c=build(WMUL,SEED)
DM,C1,C2,C3,C4,BN,DIMS,FFN,HEAD=c["DM"],c["C1"],c["C2"],c["C3"],c["C4"],c["BN"],c["DIMS"],c["FFN"],c["HEAD"]
betas=[REL_C*d for d in DIMS]; ALL_W=list(P.values()); NAMES=list(P.keys())
NP_=int(sum(int(np.prod(v.shape)) for v in P.values()))
print(f"model: {NP_/1e9:.2f}B params | DM={DM} DIMS={DIMS} betas={betas}",flush=True)

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
def F_energy(S,it,tt,igt,tgt,red):
    cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
    return 0.5*red(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))

def stat(t):
    t=tf.convert_to_tensor(t)
    mx=float(tf.reduce_max(tf.abs(t)))
    fin=bool(tf.reduce_all(tf.math.is_finite(t)))
    return mx, fin

def probe_batch(tag, bi):
    rec=dict(tag=tag, first_nonfinite=None, stages={})
    x=tf.constant(imgs[bi]); tk=tf.constant(toks[bi])
    igt=tf.constant(imgs[bi].reshape(len(bi),-1)); tgt=tf.constant(toks_oh[bi].reshape(len(bi),-1))
    # stage 1: encoder taps
    it=enc_img(x); tt=enc_txt(tk)
    for lbl,taps in (("img_tap",it),("txt_tap",tt)):
        for k in range(NS):
            mx,fin=stat(taps[k]); rec["stages"][f"{lbl}{k}"]=dict(max=mx,finite=fin)
            if not fin and rec["first_nonfinite"] is None: rec["first_nonfinite"]=f"{lbl}{k}"
    # stage 2: relaxation, step by step
    Sv=[0.5*(it[k]+tt[k]) for k in range(NS)]
    for step in range(N_INFER):
        with tf.GradientTape() as tp: tp.watch(Sv); f=F_energy(Sv,it,tt,igt,tgt,tf.reduce_sum)
        gr=tp.gradient(f,Sv)
        fv=float(f); gmax=max(stat(g)[0] for g in gr); gfin=all(stat(g)[1] for g in gr)
        Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        smax=max(stat(s)[0] for s in Sv); sfin=all(stat(s)[1] for s in Sv)
        rec["stages"][f"relax{step+1}"]=dict(F=fv,gmax=gmax,gfin=gfin,smax=smax,sfin=sfin)
        if (not np.isfinite(fv) or not gfin or not sfin) and rec["first_nonfinite"] is None:
            rec["first_nonfinite"]=f"relax{step+1}"
        if rec["first_nonfinite"]: break
    # stage 3: weight tape at the relaxed state
    if rec["first_nonfinite"] is None:
        Sc=tuple(tf.constant(z) for z in Sv)
        with tf.GradientTape() as t:
            t.watch(ALL_W); F=F_energy(Sc,enc_img(x),enc_txt(tk),igt,tgt,tf.reduce_mean)
        gr=t.gradient(F,ALL_W)
        rec["stages"]["F_weightstep"]=dict(F=float(F),finite=bool(np.isfinite(float(F))))
        bad=[]; rows=[]
        lr0=LR/RAMP
        for nm,v,gg in zip(NAMES,ALL_W,gr):
            if gg is None: continue
            gg=tf.convert_to_tensor(gg)
            gmx,gfin=stat(gg)
            nv=float(tf.norm(v)); ng=float(tf.norm(gg))
            tr=(nv+1e-3)/(ng+1e-6); upd=lr0*tr*gmx
            rows.append((nm,gmx,gfin,nv,ng,tr,upd))
            if not gfin or not np.isfinite(ng): bad.append(nm)
        if bad and rec["first_nonfinite"] is None: rec["first_nonfinite"]=f"weight_grad:{bad[:3]}"
        rows.sort(key=lambda r:-(r[6] if np.isfinite(r[6]) else float("inf")))
        rec["top_updates"]=[dict(name=r[0],gmax=r[1],gfin=r[2],wnorm=r[3],gnorm=r[4],tr=r[5],upd_step0=r[6]) for r in rows[:8]]
        print(f"  [{tag}] top step-0 update magnitudes (lr0*tr*max|g|):",flush=True)
        for r in rows[:8]:
            print(f"    {r[0]:8s} max|g|={r[1]:.3e} finite={r[2]} ||w||={r[3]:.3e} ||g||={r[4]:.3e} tr={r[5]:.3e} upd={r[6]:.3e}",flush=True)
    print(f"  [{tag}] FIRST NON-FINITE: {rec['first_nonfinite'] or 'none (all finite through the weight tape)'}",flush=True)
    return rec

records=[]
records.append(probe_batch("trial(sorted head)", tr_idx[:BATCH]))
ep_rs=np.random.RandomState(SEED+7); order=ep_rs.permutation(N_TRAIN)
for b in range(NBATCH-1):
    bi=tr_idx[order[b*BATCH:(b+1)*BATCH]]
    records.append(probe_batch(f"shuffled batch {b}", bi))

out=os.path.join(HERE,f"capacity_nanprobe_w{WMUL}.json")
with open(out+".tmp","w") as fh: json.dump(dict(wmul=WMUL,params=NP_,batch=BATCH,lr=LR,records=records),fh,indent=2,default=str)
os.replace(out+".tmp",out)
print(f"saved: {out}",flush=True)
