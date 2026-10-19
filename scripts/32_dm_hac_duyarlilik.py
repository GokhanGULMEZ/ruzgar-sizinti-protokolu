"""ADIM 32 — DM testinde uzun dönem varyans (HAC) gecikme sayısına duyarlılık. Yeniden eğitim YOK; kayıtlı tahminler kullanılır.
Soru: sızdıran ve nedensel kollar arasındaki DM sonucu, Bartlett penceresinin gecikme sayısına (varsayılan h-1 = 1) bağlı mı?
Gecikmeler: 1 (varsayılan), 6, 24, 48 adım. Altı türbin, tohum 42, 1 saat, mutlak kayıp. Önkoşul: adim6_T*.pkl.
Kullanım:  python 32_dm_hac_duyarlilik.py"""
from _ortak import *
import os, pickle, numpy as np
from leakage.evaluation import dm_test

out = cikti_klasoru(); H = 2
print(f'{"türbin":7s} {"karşılaştırma":14s} ' + ' '.join(f'lag={l:<3d} DM / p' .ljust(24) for l in (1, 6, 24, 48)))
for tid in (57, 10, 83, 120, 61, 84):
    fn = os.path.join(out, f'adim6_T{tid}.pkl')
    if not os.path.exists(fn): print(f'YOK {fn}'); continue
    r = pickle.load(open(fn, 'rb'))
    for a, b, nm in (('B6', 'B7', 'VMD'), ('B8L', 'B8', 'LCSF')):
        cells = []
        for l in (1, 6, 24, 48):
            d = dm_test(r[(a, H)]['y'], r[(a, H)]['p'], r[(b, H)]['p'], h=H, lags=l)
            cells.append(f'{d["DM"]:7.2f} / {d["p_raw"]:.1e}'.ljust(24))
        print(f'T{tid:<6d} {nm:14s} ' + ' '.join(cells))
