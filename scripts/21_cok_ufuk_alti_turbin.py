"""ADIM 21 — Çok ufuklu analizin diğer beş türbine genişletilmesi (RQ2 başlangıçta yalnız Türbin 57'de).
Ufuklar: 3 saat (H=6) ve 6 saat (H=12). Modeller: B1 Persistence, B2 Ham LSTM, B3 CatBoost Direct, B6/B7 (tam seri / kayan VMD),
B8/B8L (nedensel / iki yönlü LCSF). B9 ve ablasyonlar BU adımda yoktur (maliyet).
Önkoşul: 02_turbin_kosusu.py {tid} ile üretilmiş vmd_full_T{tid}.npy ve vmd_roll_T{tid}.npy (ana Drive hesabında var; yoksa
         rolling VMD türbin başına ~1 saat sürer).
Kullanım: python 21_cok_ufuk_alti_turbin.py            (varsayılan: 10 83 120 61 84)
          python 21_cok_ufuk_alti_turbin.py 10 83
Süre   : türbin başına 2 ufuk x 5 LSTM ≈ 1-1,5 saat (Colab). Her (türbin, ufuk) bitince diske yazılır; kesilirse aynı komutla sürer.
Çıktı  : cok_ufuk_T{tid}.pkl, cok_ufuk_ozet.csv (MAE, Skill, LSI-VMD, LSI-LCSF)"""
from _ortak import *
import sys, os, pickle, numpy as np, pandas as pd
from leakage.config import L_BEST
from leakage.data import TurbineData
from leakage.pipeline import build_components, put
from leakage.models import fit_lstm, pred_lstm, catboost_direct
from leakage.evaluation import mae, lsi

HS = (6, 12)
df = veri_yukle(); out = cikti_klasoru()
tids = [int(a) for a in sys.argv[1:]] or [10, 83, 120, 61, 84]
for tid in tids:
    yol = out + f'cok_ufuk_T{tid}.pkl'
    res = pickle.load(open(yol, 'rb')) if os.path.exists(yol) else {}
    td = TurbineData(df, tid); Ut, _ = build_components(td, out)
    for H in HS:
        if ('B8L', H) in res: print(f'T{tid} H={H/2} sa zaten tamam'); continue
        D0 = td.get_xy(L_BEST, H); te = D0['te']
        put(res, 'B1', H, D0, te['yc'])
        put(res, 'B2', H, D0, pred_lstm(fit_lstm(D0, L_BEST), te['X']))
        put(res, 'B3', H, D0, catboost_direct(D0))
        for key in ('B6', 'B7', 'B8', 'B8L'):
            D = td.get_xy_ext(L_BEST, H, Ut[key])
            assert np.array_equal(D['te']['t'], te['t']), 'test kümesi farklı'
            put(res, key, H, D, pred_lstm(fit_lstm(D, L_BEST), D['te']['X']))
        pickle.dump(res, open(yol, 'wb')); print(f'T{tid} H={H/2} sa tamam', flush=True)

rows = []
for tid in tids:
    yol = out + f'cok_ufuk_T{tid}.pkl'
    if not os.path.exists(yol): continue
    res = pickle.load(open(yol, 'rb'))
    for H in HS:
        if ('B8L', H) not in res: continue
        m = {k: mae(res[(k, H)]['y'], res[(k, H)]['p']) for k in ('B1', 'B2', 'B3', 'B6', 'B7', 'B8', 'B8L')}
        rows.append(dict(turbin=tid, H_saat=H / 2, **{k: round(v, 2) for k, v in m.items()},
                         LSI_VMD=round(lsi(m['B7'], m['B6']), 1), LSI_LCSF=round(lsi(m['B8'], m['B8L']), 1),
                         Skill_B3=round(100 * (m['B1'] - m['B3']) / m['B1'], 1)))
R = pd.DataFrame(rows); print(R.to_string(index=False))
R.to_csv(out + 'cok_ufuk_ozet.csv', index=False); print('yazıldı:', out + 'cok_ufuk_ozet.csv')
