"""ADIM 6 — Hesaplama maliyeti (UYARI: tek konumda 50 ardışık pencere ölçer ve VMD süresini ~4 kat düşük gösterebilir; makaledeki değerler scripts/16_maliyet_dogrulama.py (200 eşit aralıklı konum) ile alınmıştır.) (Bölüm 6.4): LCSF ve Rolling VMD'nin gecikmesi.
Tanım: "gecikme" = YENİ BİR örnek geldiğinde bir sonraki tahmin için bileşenleri üretme süresi
  Rolling VMD : bir pencere (W=672) için bir VMD çağrısı      (ms)
  LCSF        : durum-tutan filtre ile tek yeni örnek          (µs)
  Oran        = Rolling VMD gecikmesi / LCSF gecikmesi
Ölçüm TEK ÇEKİRDEKTE, Python'da yapılır; makineye bağlıdır -> makalede donanım bilgisi verilmelidir.
(Eski bir ölçümde 46,3 ms / 112 µs / 413× çıkmıştı; makaledeki 236,9 ms / 65,1 µs / ~3640× ayrı bir koşudur.
 Her ikisi de aynı makinede, aynı oturumda yeniden ölçülüp tek bir set rapor edilmelidir.)"""
from _ortak import *
import time, platform, os, numpy as np
from scipy.signal import lfilter, lfilter_zi
from leakage.data import TurbineData
from leakage.decomposition import LCSF, last_mode, vmd_sorted
from leakage.config import W

df = veri_yukle(); td = TurbineData(df, 57); x, N = td.x_dec, td.N
lcsf = LCSF().fit(x[:len(td.tr)])
print(f'Donanım: {platform.processor() or platform.machine()} | CPU çekirdek: {os.cpu_count()} | Python {platform.python_version()}')

t0 = time.time()
for _ in range(5): lcsf.transform(x, True)
t_lcsf = (time.time() - t0) / 5                                          # tüm seri, tek seferde

n = 50; t0 = time.time()
for i in range(n): last_mode(x[5000 + i - W + 1:5000 + i + 1])
t_vmd1 = (time.time() - t0) / n                                           # Rolling VMD: tek adım

state = [lfilter_zi(b, a) * x[0] for b, a in lcsf.bands]                  # LCSF çevrimiçi: tek örnek
t0 = time.time()
for i in range(2000):
    for k, (b, a) in enumerate(lcsf.bands):
        _, state[k] = lfilter(b, a, [x[i]], zi=state[k])
t_lcsf1 = (time.time() - t0) / 2000

t0 = time.time(); vmd_sorted(x); t_full = time.time() - t0
print(f'Tüm seri ({N:,} adım): LCSF {t_lcsf:.3f} sn | Full VMD {t_full:.1f} sn | Rolling VMD ≈ {t_vmd1*N/60:.0f} dk (tek çekirdek)')
print(f'Çevrimiçi gecikme: LCSF {t_lcsf1*1e6:.0f} µs | Rolling VMD {t_vmd1*1e3:.1f} ms | oran ≈ {t_vmd1/t_lcsf1:.0f}×')
