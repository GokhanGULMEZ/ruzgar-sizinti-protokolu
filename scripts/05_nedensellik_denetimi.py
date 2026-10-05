"""ADIM 5 — Nedensellik denetimi (Bölüm 4.5): "geleceği boz, bugünkü bileşen değişiyor mu?"
Tek bir t=20000 anı seçilir; t'den SONRAKİ örneklere gürültü (σ=200 kW) eklenir. Nedensel bir yöntemin
t anındaki çıktısı DEĞİŞMEMELİ (fark = 0); sızdıran yöntemin çıktısı değişir.
Beklenen (T57): LCSF nedensel 0,0 | Rolling VMD 0,0 | LCSF çift yönlü 33,79 | Full VMD 137,08 (kW)."""
from _ortak import *
import numpy as np
from leakage.data import TurbineData
from leakage.decomposition import LCSF, last_mode, vmd_sorted
from leakage.config import W

df = veri_yukle(); td = TurbineData(df, 57)
x, N, tau = td.x_dec, td.N, 20000
lcsf = LCSF().fit(x[:len(td.tr)])
rng = np.random.default_rng(0)
x2 = x.copy(); x2[tau + 1:] += rng.normal(0, 200, N - tau - 1)           # GELECEĞİ boz

fark = lambda a, b: float(np.abs(a - b).max())
print('LCSF nedensel      :', round(fark(lcsf.transform(x, True)[tau], lcsf.transform(x2, True)[tau]), 4), '[beklenen 0]')
print('Rolling VMD        :', round(fark(last_mode(x[tau - W + 1:tau + 1]), last_mode(x2[tau - W + 1:tau + 1])), 4), '[beklenen 0]')
print('LCSF çift yönlü    :', round(fark(lcsf.transform(x, False)[tau], lcsf.transform(x2, False)[tau]), 4), '[beklenen >0 → sızıntı]')
print('Full VMD           :', round(fark(vmd_sorted(x).T[tau], vmd_sorted(x2).T[tau]), 4), '[beklenen >0 → sızıntı]  (~1 dk)')
