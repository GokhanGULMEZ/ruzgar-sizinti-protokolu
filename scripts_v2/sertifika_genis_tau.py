"""Sertifika, genisletilmis gecikme izgarasi + bilesen bazli D_k.
Neden : Makale sertifikasi tau<=32 adimda bitiyor; sizdiran islecler icin erisim '>=32' olarak kaliyor. Bu betik tau'yu 2048
        adima kadar uzatir ve her bilesen k icin D_k(tau; delta) verir.
Isleclar (yalniz SIZDIRANLAR; nedensel olanlar yapi geregi 0): iki yonlu LCSF, tam seri VMD.
Not   : gecikmeler ayni ayristirmadan okunur; tau'yu uzatmak tam seri VMD icin EK MALIYET getirmez (cagri sayisi = t0 sayisi x delta sayisi).
Girdi : RT_PKG (leakage_min'in acildigi klasor), RT_VERI_CSV (varsayilan Drive yolu), RT_EK_OUT (cikti klasoru).
Kullanim: python sertifika_genis_tau.py [t0_lcsf=20] [t0_vmd=6] [turbin=57] [delta=200]
Sure  : LCSF saniyeler; tam seri VMD cagri basina ~20-60 sn -> ~ (t0_vmd x 1 + 1) cagri.
Cikti : sertifika_genis_T{tid}.csv (uzun biçim: islec, delta, bilesen, tau, D_max, D_ort)"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.environ.get('RT_PKG', '/content/paket'))
from leakage import config
from leakage.data import TurbineData
from leakage.decomposition import LCSF, full_vmd

a = sys.argv[1:]
n_lc = int(a[0]) if len(a) > 0 else 20; n_vmd = int(a[1]) if len(a) > 1 else 6
tid = int(a[2]) if len(a) > 2 else 57; delta = float(a[3]) if len(a) > 3 else 200.0
TAU = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
OUT = os.environ.get('RT_EK_OUT', os.path.join(config.OUT, 'ek_cikti') + os.sep); os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(config.VERI_CSV); td = TurbineData(df, tid); x = td.x_dec.astype(float); N = td.N
lc = LCSF().fit(x[:len(td.tr)])
ISLEC = {'LCSF iki yonlu': (lambda s: lc.transform(s, False), n_lc),
         'Tam seri VMD':   (lambda s: full_vmd(s, len(s)), n_vmd)}
rng = np.random.default_rng(7); sat = []
for ad, (f, nt0) in ISLEC.items():
    t0s = sorted(rng.integers(max(TAU) + 1, N - 2, nt0).tolist())
    Z0 = f(x); K = Z0.shape[1]; D = np.zeros((nt0, len(TAU), K)); prng = np.random.default_rng(11)
    for i, t0 in enumerate(t0s):
        x2 = x.copy(); x2[t0 + 1:] += prng.normal(0.0, delta, N - t0 - 1)
        Z2 = f(x2)
        for j, tau in enumerate(TAU):
            D[i, j] = np.abs(Z0[t0 - tau] - Z2[t0 - tau])
        print(f'{ad}: t0 {i+1}/{nt0} tamam', flush=True)
    for j, tau in enumerate(TAU):
        for k in range(K):
            sat.append(dict(islec=ad, delta_kW=delta, bilesen=k + 1, tau=tau, D_max=float(D[:, j, k].max()), D_ort=float(D[:, j, k].mean()), t0_sayisi=nt0))
r = pd.DataFrame(sat); r.to_csv(OUT + f'sertifika_genis_T{tid}.csv', index=False)
for ad in ISLEC:
    p = r[r.islec == ad].pivot(index='tau', columns='bilesen', values='D_max'); p['max_k'] = p.max(1)
    print('\n' + ad + f' | D_max (kW), delta={delta:g}'); print(p.round(3).to_string())
