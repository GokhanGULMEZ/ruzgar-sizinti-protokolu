"""EK 01 - Diger bes turbinde 3 ve 6 saat icin DM(+HLN, Holm) ve blok bootstrap LSI araligi.
Neden : v1 sonuclarinda bu hucrelerde (3 ve 6 sa, diger bes turbin) yalniz tek tohum vardi, DM yapilmamisti. Bu betik KAYITLI tahminlerden DM'i hesaplar (egitim YOK, ~1 dk).
Girdi : cok_ufuk_T{10,83,120,61,84}.pkl  (21_cok_ufuk_alti_turbin.py ciktisi; anahtar (model, H); H=6 ve 12)
Cikti : RT_EK_OUT/ek01_dm_3sa_6sa.csv
Karsilastirmalar: VMD  = B6 (tam seri, sizdiran) vs B7 (kayan, nedensel);  LCSF = B8L (iki yonlu) vs B8 (nedensel). DM<0 => sizdiran daha iyi.
Holm ailesi: kayip basina (mutlak, kare) tum turbin x ufuk x karsilastirma testleri (5 x 2 x 2 = 20).
Not   : tek egitilmis model (tohum 42); tohum degiskenligi bu betikte YOK. Aralik yalniz test ornekleme belirsizligidir."""
import numpy as np, pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ortak'))
from ek_ortak import *

TIDS = (10, 83, 120, 61, 84); HS = (6, 12); BLOK, B = 96, 2000
PAIRS = {'VMD': ('B6', 'B7'), 'LCSF': ('B8L', 'B8')}
rng = np.random.default_rng(0); rows = []
for tid in TIDS:
    r = yukle(f'cok_ufuk_T{tid}.pkl')
    if r is None: print(f'T{tid}: cok_ufuk_T{tid}.pkl yok, atlandi'); continue
    for H in HS:
        for ad, (sz, ned) in PAIRS.items():
            if (sz, H) not in r or (ned, H) not in r: continue
            a, b = r[(sz, H)], r[(ned, H)]
            assert len(a['y']) == len(b['y']) and np.allclose(a['y'], b['y']), 'test kumesi farkli'
            y = a['y']; ea, eb = np.abs(y - a['p']), np.abs(y - b['p'])
            m, q = dm_test(y, a['p'], b['p'], h=H, loss='abs'), dm_test(y, a['p'], b['p'], h=H, loss='sq')
            idx = blok_indeks(len(y), BLOK, B, rng)
            v = lsi(eb[idx].mean(1), ea[idx].mean(1))
            rows.append(dict(turbin=tid, H_saat=H / 2, karsilastirma=ad, MAE_sizdiran=ea.mean(), MAE_nedensel=eb.mean(),
                             LSI=lsi(eb.mean(), ea.mean()), LSI_alt=np.percentile(v, 2.5), LSI_ust=np.percentile(v, 97.5),
                             DM_abs=m['DM'], p_abs=m['p_raw'], DM_sq=q['DM'], p_sq=q['p_raw'], T=m['T']))
if not rows: raise SystemExit('Hic hucre islenemedi (cok_ufuk_T*.pkl yok mu?).')
R = pd.DataFrame(rows)
R['holm_abs'] = holm(R.p_abs.values); R['holm_sq'] = holm(R.p_sq.values)
R['anlamli_abs'] = (R.holm_abs < 0.05) & (R.DM_abs < 0); R['anlamli_sq'] = (R.holm_sq < 0.05) & (R.DM_sq < 0)
R.to_csv(EK_OUT + 'ek01_dm_3sa_6sa.csv', index=False)
pd.set_option('display.width', 250)
print(R[['turbin', 'H_saat', 'karsilastirma', 'LSI', 'LSI_alt', 'LSI_ust', 'DM_abs', 'holm_abs', 'DM_sq', 'holm_sq']].round(3).to_string(index=False))
print(f'\nMutlak kayipta anlamli: {int(R.anlamli_abs.sum())}/{len(R)} | kare kayipta: {int(R.anlamli_sq.sum())}/{len(R)}')
print('yazildi:', EK_OUT + 'ek01_dm_3sa_6sa.csv')
