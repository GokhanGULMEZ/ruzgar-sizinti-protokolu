"""ADIM 3 — Tohum tekrarı (Bölüm 6.3): LSTM'ler tohum 43 ve 44 ile yeniden eğitilir (1 saat ufku).
Önkoşul: 02_turbin_kosusu.py (adim6_T{tid}.pkl ve ayrıştırma önbelleği). Rolling VMD YENİDEN hesaplanmaz.
Kullanım:  python 03_seed_tekrar.py 10 83 120 61 84"""
from _ortak import *
import sys
from leakage.pipeline import seed_tekrar, seed_ozet

df = veri_yukle(); cikti_klasoru()
for tid in [int(a) for a in sys.argv[1:]] or [10]:
    seed_tekrar(df, tid)
    print(f'\n== Türbin {tid}'); seed_ozet(tid)
