"""ADIM 2 — Altı türbinde ana koşu (Bölüm 6.2): B1,B2,B3,B6,B7,B8,B8L,B9 @ 1 saat.
Kullanım:  python 02_turbin_kosusu.py 10 83 120 61 84
Süre: türbin başına ~70 dk (çoğu Rolling VMD; 2 çekirdekli CPU'da daha uzun). Rolling VMD diske
önbelleklenir; kesilirse aynı komutla devam eder. Çıktı: adim6_T{tid}.pkl + ozet_T{tid}.csv"""
from _ortak import *
import sys
from leakage.pipeline import run_turbine, ozet
from leakage import config

df = veri_yukle(); out = cikti_klasoru()
tids = [int(a) for a in sys.argv[1:]] or [10]
for tid in tids:
    res = run_turbine(df, tid, Hs=(2,))
    ozet(res, tid, Hs=(2,)).to_csv(out + f'ozet_T{tid}.csv', index=False)
