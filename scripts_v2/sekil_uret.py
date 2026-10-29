"""Ana metin Sekil 1-3'u uretir (tablolardaki sayilardan). Cikti: ../sekiller_v2/ (makalede figures/ adiyla kullanilir)"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sekiller_v2"); os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Sekil 1: MAE - ufuk (T57, seed 42)
h=[0.5,1,2,3,4,6]
pers=[77.80,108.41,153.78,187.89,214.98,255.53]
cb=[78.78,106.12,143.35,168.05,187.36,215.69]
rv=[86.76,116.11,158.98,191.00,215.86,247.92]
fs=[53.29,64.71,67.73,73.43,96.81,88.55]
fig,ax=plt.subplots(figsize=(4.6,3.1))
ax.plot(h,pers,"o-",c="0.55",label="Persistence")
ax.plot(h,cb,"s-",c="tab:green",label="CatBoost Direct")
ax.plot(h,rv,"^-",c="tab:blue",label="Rolling VMD (causal)")
ax.plot(h,fs,"v-",c="tab:red",label="Full-series VMD (leaky)")
ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("MAE (kW)"); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(f"{out}/fig_horizon.png",dpi=300); plt.close(fig)

# Sekil 2: look-ahead doz-yanit (T57, 1 sa)
m=["0","1","2","4","8","16","48","full"]
lsi=[2.3,26.0,90.5,74.9,48.7,45.2,46.8,46.7]
sd=[1.4,1.8,1.0,0.6,6.3,2.7,5.1,7.7]
fig,ax=plt.subplots(figsize=(4.6,3.1))
ax.errorbar(range(len(m)),lsi,yerr=sd,fmt="o-",c="tab:red",capsize=3)
ax.set_xticks(range(len(m))); ax.set_xticklabels(m)
ax.set_xlabel("Look-ahead $m$ (steps of 30 min; full = whole series)"); ax.set_ylabel("LSI-LCSF (%)")
ax.annotate("$m=H$: target\nin component",(2,90.5),(3.2,85),fontsize=8,arrowprops=dict(arrowstyle="->",lw=.7))
fig.tight_layout(); fig.savefig(f"{out}/fig_dose.png",dpi=300); plt.close(fig)

# Sekil 3: sertifika profili D(tau)
t=[0,1,2,4,8,16,32,64,128,256,512,1024,2048]
l2=[118.39,110.61,93.94,46.53,47.87,30.79,18.76,12.90,3.45,1.51,0.21,0.003,0.0]
fv=[91.75,182.17,279.15,84.14,70.70,71.56,73.41,109.10,465.04,296.66,108.65,137.90,119.44]
fig,ax=plt.subplots(figsize=(4.6,3.1))
tp=[x if x>0 else 0.7 for x in t]
ax.plot(tp,[max(v,1e-3) for v in l2],"o-",c="tab:orange",label="Two-sided LCSF (leaky)")
ax.plot(tp,fv,"v-",c="tab:red",label="Full-series VMD (leaky)")
ax.axhline(1e-3,c="tab:blue",lw=1.2,ls="--"); ax.text(0.8,2e-3,"Causal LCSF and rolling VMD: $D=0$",color="tab:blue",fontsize=8)
ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_ylim(5e-4,1e3)
ax.set_xlabel(r"Lag $\tau$ (steps; $\tau=0$ plotted at 0.7)"); ax.set_ylabel(r"$D(\tau)$ for $\delta=200$ kW (kW)")
ax.legend(frameon=False,fontsize=8,loc="lower left",bbox_to_anchor=(0.0,0.2))
fig.tight_layout(); fig.savefig(f"{out}/fig_cert.png",dpi=300); plt.close(fig)
