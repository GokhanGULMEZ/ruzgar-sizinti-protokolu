"""ADIM 16 — Maliyet ölçümünün doğrulanması: tek çağrı tahmini (22,3 ms) ile toplu koşu (türbin başına 54-67 dk) arasındaki ~4 kat farkı açıkla.
Kullanım:  python 16_maliyet_dogrulama.py 57        (~5-10 dk, tek çekirdek, TEK süreçte ARDIŞIK)
Yöntem : Seri boyunca 200 EŞİT ARALIKLI konumda Rolling VMD (W=672) bir kez çağrılır (yalnız bir konumda değil); ortalama, ortanca, IQR,
         en yavaş %10. Tüm seri tahmini = ortalama x N. Ardından aynı 200 konum n_jobs=2 ile paralel çalıştırılır; toplam süre ve hızlanma.
Not    : Aynı oturumda yeniden ölçün; ilk çağrılar ısınma (JIT/önbellek) içerebilir, bu yüzden ilk 5 çağrı atılır."""
from _ortak import *
import sys
import time, platform, os, numpy as np
from joblib import Parallel, delayed
from leakage.data import TurbineData
from leakage.decomposition import LCSF, last_mode
from leakage.config import W
from scipy.signal import lfilter, lfilter_zi

tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
td = TurbineData(veri_yukle(), tid); x, N = td.x_dec, td.N
print(f'Donanım: {platform.processor() or platform.machine()} | CPU {os.cpu_count()} | Python {platform.python_version()}')
pos = np.linspace(W, N - 1, 205).astype(int)
for p in pos[:5]: last_mode(x[p - W + 1:p + 1])                                   # ısınma
ts = []
for p in pos[5:]:
    t0 = time.perf_counter(); last_mode(x[p - W + 1:p + 1]); ts.append(time.perf_counter() - t0)
ts = np.array(ts) * 1e3
print(f'Rolling VMD tek adım (ms): ortalama {ts.mean():.1f} | ortanca {np.median(ts):.1f} | IQR {np.percentile(ts,25):.1f}-{np.percentile(ts,75):.1f} | en yavaş %10 ort. {np.sort(ts)[-20:].mean():.1f}')
print(f'  tüm seri tahmini (tek çekirdek): ortalama x N = {ts.mean()*N/6e4:.0f} dk | ortanca x N = {np.median(ts)*N/6e4:.0f} dk')
t0 = time.perf_counter(); Parallel(n_jobs=2)(delayed(last_mode)(x[p - W + 1:p + 1]) for p in pos[5:]); tp = time.perf_counter() - t0
print(f'  n_jobs=2 ile 200 çağrı: {tp:.1f} sn (ardışık toplam {ts.sum()/1e3:.1f} sn) -> hızlanma {ts.sum()/1e3/tp:.2f}x')
lc = LCSF().fit(x[:len(td.tr)]); st = [lfilter_zi(b, a) * x[0] for b, a in lc.bands]; t0 = time.perf_counter()
for i in range(5000):
    for k, (b, a) in enumerate(lc.bands): _, st[k] = lfilter(b, a, [x[i]], zi=st[k])
tl = (time.perf_counter() - t0) / 5000
print(f'LCSF çevrimiçi tek örnek: {tl*1e6:.0f} µs | oran (ortalama) ≈ {ts.mean()*1e-3/tl:.0f}x | (ortanca) ≈ {np.median(ts)*1e-3/tl:.0f}x')
