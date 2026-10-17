"""ADIM 28 — Referans modeller: persistence, iklimsel ortalama ve Nielsen vd. (1998) referansı (altı türbin, altı ufuk).
Soru: nedensel modellerin persistence'a göre kazancı, ufuk uzadıkça a_k -> 0 olduğunda daha güçlü bir referansa karşı da sürüyor mu?
Eğitim GEREKTİRMEZ; yalnız veri gerekir (a_k ve ortalama eğitim bölümünden kestirilir). Sızıntı sonuçlarını etkilemez
(eşleştirilmiş çiftler üzerinde tanımlıdır).
Kullanım: python 28_referans_modeller.py            (varsayılan altı türbin)   |   python 28_referans_modeller.py 57 10
Süre   : birkaç dakika (CPU).
Çıktı  : referans_modeller.csv (türbin, ufuk, a_k, MAE/RMSE: persistence, iklimsel, Nielsen).
         Var olan tahmin dosyaları (adim6_T*.pkl, cok_ufuk_T*.pkl) bulunursa modellerin Nielsen'e göre MAE skill'i de yazdırılır."""
from _ortak import *
import os, sys, pickle, numpy as np, pandas as pd
from leakage.config import L_BEST, H_LIST
from leakage.data import TurbineData
from leakage.reference_models import referans_hatalari
from leakage.evaluation import mae

TURBINLER = tuple(int(a) for a in sys.argv[1:]) or (57, 10, 83, 120, 61, 84)
df = veri_yukle(); out = cikti_klasoru(); satir = []
for tid in TURBINLER:
    td = TurbineData(df, tid); p_tr = td.tr['Patv'].values
    for H in H_LIST:
        te = td.get_xy(L_BEST, H)['te']
        r = referans_hatalari(p_tr, te['y'], te['yc'], H)
        satir.append(dict(turbin=tid, ufuk_saat=H / 2, a_k=round(r['a_k'], 4), n_test=len(te['y']),
                          MAE_pers=round(r['persistence'][0], 2), MAE_iklim=round(r['iklimsel'][0], 2), MAE_nielsen=round(r['nielsen'][0], 2),
                          RMSE_pers=round(r['persistence'][1], 2), RMSE_iklim=round(r['iklimsel'][1], 2), RMSE_nielsen=round(r['nielsen'][1], 2)))
    print(f'T{tid} tamam', flush=True)
tab = pd.DataFrame(satir); tab.to_csv(out + 'referans_modeller.csv', index=False)
print(tab.to_string(index=False))

print('\n== Nielsen referansına göre MAE skill (%), eldeki tahmin dosyalarından (nedensel modeller: B2 ham, B7 kayan VMD, B8 LCSF, B3 CatBoost)')
for tid in TURBINLER:
    for dosya, Hs in ((f'adim6_T{tid}.pkl', (2,)), (f'cok_ufuk_T{tid}.pkl', (6, 12))):
        if not os.path.exists(out + dosya): continue
        res = pickle.load(open(out + dosya, 'rb'))
        for H in Hs:
            ref = tab[(tab.turbin == tid) & (tab.ufuk_saat == H / 2)]
            if ref.empty or ('B2', H) not in res: continue
            nm = float(ref.MAE_nielsen.iloc[0]); sk = {}
            for k in ('B2', 'B3', 'B7', 'B8'):
                if (k, H) in res: sk[k] = round(100 * (nm - mae(res[(k, H)]['y'], res[(k, H)]['p'])) / nm, 1)
            print(f'  T{tid} H={H/2} sa  Nielsen MAE={nm:.2f}  skill:', sk)
