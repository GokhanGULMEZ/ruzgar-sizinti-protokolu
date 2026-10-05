"""ADIM 15 — LSI için hareketli blok bootstrap güven aralığı (hakem isteği: yalnız std yetersiz).
Kullanım:  python 15_bootstrap_guven_araligi.py            (altı türbin, 1 saat, tohum 42; adim6_T*.pkl okunur; <1 dk)
Yöntem : Test noktaları ARDIŞIK bloklar halinde (varsayılan 96 adım = 2 gün) yeniden örneklenir; AYNI indeksler sızdıran ve nedensel
         modele uygulanır (eşleştirilmiş). LSI = (MAE_nedensel − MAE_sızdıran)/MAE_nedensel; 2000 yeniden örnekleme; yüzde 2,5-97,5 aralığı.
Not    : Bu aralık yalnız test örnekleme belirsizliğini yansıtır; LSTM eğitim (tohum) değişkenliğini içermez (bkz. Bölüm 6.3)."""
from _ortak import *
import os, pickle, numpy as np, pandas as pd
from leakage.config import OUT

BLOK, B, RNG = 96, 2000, np.random.default_rng(0)


def ci(e_c, e_l):
    n = len(e_c); nb = int(np.ceil(n / BLOK)); st = RNG.integers(0, n - BLOK + 1, size=(B, nb))
    idx = (st[:, :, None] + np.arange(BLOK)).reshape(B, -1)[:, :n]
    c, l = e_c[idx].mean(1), e_l[idx].mean(1)
    v = 100 * (c - l) / c
    return 100 * (e_c.mean() - e_l.mean()) / e_c.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


rows = []
for tid in (57, 10, 83, 120, 61, 84):
    yol = OUT + f'adim6_T{tid}.pkl'
    if not os.path.exists(yol): print('YOK', yol); continue
    r = pickle.load(open(yol, 'rb'))
    ae = lambda k: np.abs(r[(k, 2)]['y'] - r[(k, 2)]['p'])
    for ad, kc, kl in (('LSI-VMD', 'B7', 'B6'), ('LSI-LCSF', 'B8', 'B8L')):
        n1, lo, hi = ci(ae(kc), ae(kl)); rows.append(dict(turbin=tid, olcut=ad, LSI=n1, alt=lo, ust=hi))
        print(f'T{tid} {ad:9s} LSI {n1:5.1f}%  %95 aralık [{lo:5.1f}, {hi:5.1f}]')
R = pd.DataFrame(rows); R.to_csv(OUT + 'lsi_bootstrap_aralik.csv', index=False); print('yazıldı:', OUT + 'lsi_bootstrap_aralik.csv')
