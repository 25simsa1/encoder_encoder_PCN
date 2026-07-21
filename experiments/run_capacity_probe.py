"""CAPACITY LADDER PHASE 0 -- memory and throughput probe. No training, no data download, no claims.

For each WMUL in PROBE_WMULS and each batch in PROBE_BATCHES (descending), build the model, run
PROBE_STEPS joint steps on synthetic data shaped exactly like the real pipeline (RES=64, CAPLEN=64,
V=49, the 8k seed-0 vocabulary size), and record peak GPU memory and seconds per step. fp32 throughout;
precision is not a free knob in this project. Three step modes, each probed independently:
  pc    the frozen recipe joint step: get_taps + 8-step relax_full + LARS weight_step
        (byte-copied ops from run_coupling_scale.py)
  adam  the E1 baseline step: InfoNCE forward/backward + manual Adam (m,v slots allocated like
        run_E1_bp_clip_baseline.py; the slot allocation itself is part of the memory question)
  lars  the E1L fallback baseline step: InfoNCE forward/backward + LARS (stateless), for sizes where
        Adam's 2x parameter state cannot fit

OOM is a recorded outcome, not an error: each (wmul, batch, mode) config is wrapped, the session is
rebuilt after an OOM, and the sweep continues at the next smaller batch. The output table is the input
to the wall-clock projection that must be published before any real submission.

ENV: PROBE_WMULS("2.18,3.18") PROBE_BATCHES("128,64,32,16,8,4,2") PROBE_MODES("pc,adam,lars")
PROBE_STEPS(20) PROBE_DEVICE(label for the record, e.g. L4/A100/MIG20) PROBE_V(49) PROBE_OUT
(capacity_probe_<device>.json). RUNS1_SMOKE=1 shrinks everything for a CPU mechanics check.
OUT: one JSON with a record per (wmul, params, mode, batch): fit true/false, peak_gb, sec_per_step.
"""
import os, sys, time, json, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("RUNS1_GPU", "0"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf

HERE   = os.path.dirname(os.path.abspath(__file__))
SMOKE  = os.environ.get("RUNS1_SMOKE", "0") == "1"
RES    = 16 if SMOKE else 64
CAPLEN = 16 if SMOKE else 64
V      = int(os.environ.get("PROBE_V", 24 if SMOKE else 49))
WMULS  = [float(w) for w in os.environ.get("PROBE_WMULS", "0.1,0.2" if SMOKE else "2.18,3.18").split(",")]
BATCHES= [int(b) for b in os.environ.get("PROBE_BATCHES", "2" if SMOKE else "128,64,32,16,8,4,2").split(",")]
MODES  = os.environ.get("PROBE_MODES", "pc,adam,lars").split(",")
STEPS  = int(os.environ.get("PROBE_STEPS", 4 if SMOKE else 20))
DEVICE = os.environ.get("PROBE_DEVICE", "cpu" if SMOKE else "unknown")
OUT    = os.environ.get("PROBE_OUT", os.path.join(HERE, f"capacity_probe_{DEVICE}.json"))
SEED   = 0
TEMP   = 0.07

# recipe constants (identical to run_coupling_scale.py)
HEADS, NBLK, NS = 4, 4, 4
CODE, DEC_SD = 16, 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C = 0.05
N_INFER = 2 if SMOKE else 8
B_C1,B_C2,B_C3,B_C4,B_BN = 32,64,128,256,512
B_DM, B_FFN = 512, 1024
B_DIMS = [2048,2048,1024,1024]
CH = 3; PIX = RES*RES*CH

def gelu(z): return tf.nn.gelu(z)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

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
    def F_energy(S,it,tt,igt,tgt,red):
        cross=tf.add_n([mse(S[k]-it[k])+mse(S[k]-tt[k]) for k in range(NS)])
        return 0.5*red(A_CROSS*cross+A_GEN*(mse(dec_img(S)-igt)+mse(dec_txt(S)-tgt)))
    @tf.function
    def get_taps(x,tk): return enc_img(x),enc_txt(tk)
    def relax_full(S,it,tt,igt,tgt,n):
        Sv=list(S)
        for _ in range(n):
            with tf.GradientTape() as tp: tp.watch(Sv); f=F_energy(Sv,it,tt,igt,tgt,tf.reduce_sum)
            gr=tp.gradient(f,Sv); Sv=[Sv[k]-betas[k]*gr[k] for k in range(NS)]
        return Sv
    @tf.function
    def weight_step(x,tk,S,igt,tgt,lr):
        with tf.GradientTape() as t: t.watch(ALL_W); F=F_energy(S,enc_img(x),enc_txt(tk),igt,tgt,tf.reduce_mean)
        gr=t.gradient(F,ALL_W)
        for v,gg in zip(ALL_W,gr):
            if gg is None: continue
            tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
        return F
    def l2n(z): return z/(tf.norm(z,axis=1,keepdims=True)+1e-9)
    def latents(x,tk): return l2n(tf.concat(enc_img(x),1)), l2n(tf.concat(enc_txt(tk),1))
    def infonce(zi,zt,temp):
        logits=tf.matmul(zi,zt,transpose_b=True)/temp; B=tf.shape(zi)[0]; lab=tf.range(B)
        return 0.5*(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=logits))
                   +tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab,logits=tf.transpose(logits))))
    return dict(get_taps=get_taps,relax_full=relax_full,weight_step=weight_step,latents=latents,
                infonce=infonce,ALL_W=ALL_W)

def synth_batch(B, rs):
    x=rs.rand(B,RES,RES,CH).astype("float32")
    tk=rs.randint(0,V,(B,CAPLEN)).astype("int32")
    toh=np.eye(V,dtype="float32")[tk].reshape(B,-1)
    return tf.constant(x), tf.constant(tk), tf.constant(x.reshape(B,-1)), tf.constant(toh)

def gpu_present():
    return len(tf.config.list_physical_devices("GPU")) > 0

def peak_gb():
    try: return tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: return None

def reset_peak():
    try: tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception: pass

OOM_ERRS=(tf.errors.ResourceExhaustedError, tf.errors.InternalError)

def probe_config(wmul, mode, B, rs):
    """Build fresh, run STEPS steps, return (fit, peak_gb, sec_per_step, err)."""
    tf.keras.backend.clear_session(); reset_peak()
    try:
        P,c=build(wmul,SEED); ops=make_ops(P,c)
        NP=int(sum(int(np.prod(v.shape)) for v in P.values()))
    except OOM_ERRS as e:
        return None, False, peak_gb(), None, f"OOM at build: {type(e).__name__}"
    ALL_W=ops["ALL_W"]
    try:
        if mode=="pc":
            def step(x,tk,igt,tgt):
                it,tt=ops["get_taps"](x,tk)
                Sv=ops["relax_full"]([0.5*(it[k]+tt[k]) for k in range(NS)],it,tt,igt,tgt,N_INFER)
                return ops["weight_step"](x,tk,tuple(tf.constant(z) for z in Sv),igt,tgt,tf.constant(1e-6,tf.float32))
        elif mode in ("adam","lars"):
            if mode=="adam":
                M_=[tf.Variable(tf.zeros_like(v),trainable=False) for v in ALL_W]
                Vv=[tf.Variable(tf.zeros_like(v),trainable=False) for v in ALL_W]
                B1,B2,EPS=0.9,0.999,1e-8
            @tf.function
            def inner(x,tk,lr,tstep):
                with tf.GradientTape() as t:
                    t.watch(ALL_W); zi,zt=ops["latents"](x,tk); L=ops["infonce"](zi,zt,tf.constant(TEMP,tf.float32))
                gr=t.gradient(L,ALL_W)
                for i,(v,gg) in enumerate(zip(ALL_W,gr)):
                    if gg is None: continue
                    gg=tf.convert_to_tensor(gg)
                    if mode=="adam":
                        m,s2=M_[i],Vv[i]
                        m.assign(B1*m+(1-B1)*gg); s2.assign(B2*s2+(1-B2)*tf.square(gg))
                        v.assign_sub(lr*(m/(1-tf.pow(B1,tstep)))/(tf.sqrt(s2/(1-tf.pow(B2,tstep)))+EPS))
                    else:
                        tr=(tf.norm(v)+1e-3)/(tf.norm(gg)+1e-6); v.assign_sub(lr*tr*gg)
                return L
            def step(x,tk,igt,tgt):
                return inner(x,tk,tf.constant(1e-7,tf.float32),tf.constant(1.0,tf.float32))
        else:
            raise ValueError(mode)
        x,tk,igt,tgt=synth_batch(B,rs)
        t_first=time.time(); step(x,tk,igt,tgt); trace_s=time.time()-t_first   # trace + first run
        times=[]
        for i in range(STEPS):
            t0=time.time(); step(x,tk,igt,tgt); times.append(time.time()-t0)
        sec=float(np.mean(times[len(times)//2:]))                              # steady-state half
        return NP, True, peak_gb(), sec, None
    except OOM_ERRS as e:
        return None, False, peak_gb(), None, f"OOM: {type(e).__name__}"

rs=np.random.RandomState(0)
records=[]
print(f"=== CAPACITY PROBE === device={DEVICE} gpu={gpu_present()} smoke={SMOKE} wmuls={WMULS} "
      f"batches={BATCHES} modes={MODES} steps={STEPS} fp32 V={V}",flush=True)
for w in WMULS:
    for mode in MODES:
        for B in BATCHES:
            NP,fit,pk,sec,err=probe_config(w,mode,B,rs)
            rec=dict(device=DEVICE,wmul=w,params=NP,mode=mode,batch=B,fit=bool(fit),
                     peak_gb=pk,sec_per_step=sec,err=err,n_infer=N_INFER,steps=STEPS,fp32=True)
            records.append(rec)
            ps = f"{NP/1e9:.2f}B" if NP else "n/a"
            print(f"  [{DEVICE}] wmul={w} ({ps}) mode={mode} B={B}: "
                  f"{'FIT peak %.1f GB, %.3f s/step'%(pk or -1,sec or -1) if fit else 'NO FIT (%s, peak %s)'%(err, '%.1f'%pk if pk else '?')}",flush=True)
            if fit: break                                                      # largest fitting batch found; smaller ones only waste probe time
tmp=OUT+".tmp"
with open(tmp,"w") as fh: json.dump(dict(device=DEVICE,records=records),fh,indent=2)
os.replace(tmp,OUT)
print(f"saved: {OUT} ({len(records)} records)",flush=True)
if SMOKE: print("[SMOKE] mechanics only; no GPU memory numbers on CPU.",flush=True)
