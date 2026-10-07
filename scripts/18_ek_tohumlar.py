"""ADIM 18 — Ek tohumlar (45–49): tohum sayısını 3'ten 8'e çıkarır (LSI ortalama ± std için daha güvenilir tahmin).
Önkoşul: 02_turbin_kosusu.py {tid} ve 03_seed_tekrar.py {tid} (tohum 42, 43, 44 sonuçları).
Kullanım:  python 18_ek_tohumlar.py 57            (varsayılan türbin 57; ardından örn. 120 10 isteğe bağlı)
Süre   : türbin başına 4 LSTM x 5 tohum + 1 ... ≈ 20 LSTM -> Colab'da ~1,5-2 saat. Ayrıştırma önbellekten okunur.
Çıktı  : seed_ek_T{tid}.pkl ve ekranda 8 tohumluk özet (LSI ortalama, std, min–maks, t tabanlı %95 aralık)."""
from _ortak import *
import sys, pickle, os
import numpy as np
from scipy import stats
from leakage.pipeline import seed_tekrar
from leakage.evaluation import mae
from leakage import config

df = veri_yukle(); out = cikti_klasoru()
EK = (45, 46, 47, 48, 49)
for tid in [int(a) for a in sys.argv[1:]] or [57]:
    fn = f'seed_ek_T{tid}.pkl'
    if not os.path.exists(os.path.join(out, fn)):
        seed_tekrar(df, tid, seeds=EK, dosya=fn)
    base = pickle.load(open(os.path.join(out, f'adim6_T{tid}.pkl'), 'rb'))
    s3 = pickle.load(open(os.path.join(out, f'seed_T{tid}.pkl'), 'rb'))
    s5 = pickle.load(open(os.path.join(out, fn), 'rb'))
    H = 2
    def M(k):
        v = [mae(base[(k, H)]['y'], base[(k, H)]['p'])]
        v += [mae(s3[(k, s)]['y'], s3[(k, s)]['p']) for s in config.SEEDS_EK]
        v += [mae(s5[(k, s)]['y'], s5[(k, s)]['p']) for s in EK]
        return np.array(v)
    print(f'\n== Türbin {tid}: 8 tohum (42, 43, 44, 45-49)')
    for ad, (a, b) in {'LSI VMD': ('B7', 'B6'), 'LSI LCSF': ('B8', 'B8L')}.items():
        l = 100 * (M(a) - M(b)) / M(a); n = len(l)
        ci = stats.t.interval(0.95, n - 1, loc=l.mean(), scale=l.std(ddof=1) / np.sqrt(n))
        print(f'  {ad}: tohum başına {np.round(l,1)}\n     ort={l.mean():.1f} ± {l.std(ddof=1):.1f} | min–maks {l.min():.1f}–{l.max():.1f} | ortalamanın %95 aralığı {ci[0]:.1f}–{ci[1]:.1f}')
    for k, ad in {'B2': 'Ham LSTM', 'B7': 'Kayan VMD', 'B8': 'LCSF nedensel'}.items():
        v = M(k); print(f'  {ad} MAE: ort={v.mean():.2f} ± {v.std(ddof=1):.2f}')
