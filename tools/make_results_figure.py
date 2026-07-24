"""ICLR hero results strip. Numbers trace to committed logs (see spec)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
BP,PC="#0173b2","#de8f05"
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"axes.linewidth":0.8})
f,(a,b,c)=plt.subplots(1,3,figsize=(6.75,2.2),gridspec_kw={"width_ratios":[1,1.1,1.5]})
# (a) matched fit
fit_bp=[.981,.951,.972]; fit_pc=[.997,.998,.999]
for i,(u,v) in enumerate(zip(fit_bp,fit_pc)):
    a.plot([0,1],[u,v],"-",c="#ccc",lw=.8,zorder=1)
    a.plot([0],[u],"o",c=BP,ms=4,zorder=2); a.plot([1],[v],"o",c=PC,ms=4,zorder=2)
a.set_xticks([0,1],["backprop","pred. coding"]); a.set_xlim(-.5,1.5); a.set_ylim(.9,1.005)
a.set_ylabel("train latent retrieval"); a.set_title("(a) matched training fit",fontsize=8)
# (b) headline
pb=[.2146,.2001,.1995]; pp=[.0998,.1080,.1023]; base=.0815
b.bar([0],[np.mean(pb)],.6,color=BP,alpha=.85); b.bar([1],[np.mean(pp)],.6,color=PC,alpha=.85)
b.plot([0]*3,pb,"o",c="k",ms=3,zorder=3); b.plot([1]*3,pp,"o",c="k",ms=3,zorder=3)
b.axhline(base,ls="--",c="#555",lw=.9); b.text(-.45,base+.004,"base rate",fontsize=6.5,color="#555",ha="left")
b.text(0,max(pb)+.008,"~2.5x",ha="center",fontsize=7); b.text(1,np.mean(pp)+.012,"~1.3x",ha="center",fontsize=7)
b.set_xticks([0,1],["backprop","pred. coding"]); b.set_ylabel("held-out category prec@10")
b.set_title("(b) category transfer",fontsize=8); b.set_ylim(0,.26)
# (c) dumbbells (seed 0)
cats=[("elephant",7.2,1.1),("plane",6.1,2.4),("cat",5.6,1.4),("food",5.0,1.8),("train",4.3,1.5),
      ("sports",3.8,1.8),("bathroom",3.2,2.0),("dog",1.8,.6),("cow/sheep",1.6,1.3),
      ("kitchen",1.3,1.8),("person",1.2,.9),("sign",1.0,.8),("bird",.5,.8)]
cats.sort(key=lambda t:t[1])
for i,(n,vb,vp) in enumerate(cats):
    c.plot([vp,vb],[i,i],"-",c="#bbb",lw=1.2,zorder=1)
    c.plot([vb],[i],"o",c=BP,ms=4,zorder=2); c.plot([vp],[i],"o",c=PC,ms=4,zorder=2)
c.set_yticks(range(len(cats)),[t[0] for t in cats],fontsize=6.5)
c.set_xscale("log"); c.axvline(1,ls="--",c="#555",lw=.9)
c.set_xticks([0.5,1,2,4,8]); c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:g}"))
c.xaxis.set_minor_formatter(plt.NullFormatter())
c.text(1.1,4.5,"base rate",fontsize=6.5,color="#555",rotation=90,va="bottom",ha="center")
c.set_xlabel("held-out category lift (x, log)"); c.set_title("(c) per-category (seed 0)",fontsize=8)
c.plot([],[],"o",c=BP,label="backprop"); c.plot([],[],"o",c=PC,label="pred. coding")
c.legend(fontsize=6,frameon=False,loc="lower right")
plt.tight_layout()
for ext in ("pdf","png"): plt.savefig(f"docs/paper/figs/fig2_results.{ext}",dpi=200,bbox_inches="tight")
print("wrote fig2_results.pdf/.png")
