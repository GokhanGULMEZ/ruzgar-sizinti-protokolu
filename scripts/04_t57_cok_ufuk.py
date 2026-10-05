"""ADIM 4 — Türbin 57: çok ufuklu karşılaştırma, DM+Holm ve ablasyon (Bölüm 6.1, 6.5, 6.6).

Aşamalar (her aşamadan sonra sonuçlar diske yazılır; kesilirse kaldığı yerden sürer):
  A) B1 Persistence, B2 Raw LSTM, B3 CatBoost Direct          H = 0,5/1/2/3/4/6 sa
  B) B6 Full VMD–LSTM (sızdıran), B7 Rolling VMD–LSTM         H = 0,5/1/2/3/4/6 sa
  C) B8 LCSF (nedensel), B8L LCSF (çift yönlü, sızdıran)      H = 1/3/6 sa
  D) B9 = B8 + OOF kalıntı CatBoost                           H = 1/3/6 sa
  E) Ablasyon: B8I (eğitim-içi kalıntı), B2O (Raw LSTM + OOF) H = 1/3/6 sa
  F) DM + HLN + Holm tabloları (adim5_dm_holm_T57.csv, ablation_T57.csv)
Süre: A+B ~1-2 saat (Rolling VMD önbellekteyse), C-E ~1,5-2 saat (CPU)."""
from _ortak import *
import os, pickle, numpy as np, pandas as pd
from leakage.config import H_LIST, H4, L_BEST
from leakage.data import TurbineData
from leakage.pipeline import build_components, put
from leakage.models import fit_lstm, pred_lstm, catboost_direct, stack
from leakage.evaluation import metrics, lsi, dm_test, holm

df = veri_yukle(); OUT = cikti_klasoru()
PKL = OUT + 'adim2_T57_tahminler.pkl'
results = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}
kaydet = lambda: pickle.dump(results, open(PKL, 'wb'))
td = TurbineData(df, 57); L = L_BEST
Ut, lcsf = build_components(td, OUT)          # vmd_full_T57.npy / vmd_roll_T57.npy önbellekten okunur
yeni = lambda key, H: (key, H) not in results

# A) taban modeller
for H in H_LIST:
    if all(not yeni(k, H) for k in ('B1', 'B2', 'B3')): continue
    D = td.get_xy(L, H); te_ = D['te']
    put(results, 'B1', H, D, te_['yc'])
    put(results, 'B2', H, D, pred_lstm(fit_lstm(D, L), te_['X']))
    put(results, 'B3', H, D, catboost_direct(D))
    print(f'A) H={H/2} sa tamam'); kaydet()

# B) Full ve Rolling VMD
for H in H_LIST:
    for key in ('B6', 'B7'):
        if not yeni(key, H): continue
        D = td.get_xy_ext(L, H, Ut[key])
        assert np.array_equal(D['te']['t'], results[('B1', H)]['t']), 'test örnek kümesi farklı!'
        put(results, key, H, D, pred_lstm(fit_lstm(D, L), D['te']['X']))
    print(f'B) H={H/2} sa tamam'); kaydet()

# C) LCSF nedensel / çift yönlü
base_models, Dcache = {}, {}
for H in H4:
    for key in ('B8', 'B8L'):
        D = td.get_xy_ext(L, H, Ut[key])
        assert np.array_equal(D['te']['t'], results[('B1', H)]['t'])
        m = fit_lstm(D, L)
        put(results, key, H, D, pred_lstm(m, D['te']['X']))
        if key == 'B8': base_models[H] = m; Dcache[H] = D
    print(f'C) H={H/2} sa tamam')
kaydet()

# D) B9 ve E) ablasyon
for H in H4:
    D8, m8 = Dcache[H], base_models[H]
    put(results, 'B9', H, D8, stack(D8, m8, L, H, 'oof'))
    put(results, 'B8I', H, D8, stack(D8, m8, L, H, 'insample'))
    D2 = td.get_xy(L, H); m2 = fit_lstm(D2, L)
    put(results, 'B2O', H, D2, stack(D2, m2, L, H, 'oof'))
    print(f'D/E) H={H/2} sa tamam'); kaydet()

# metrik tabloları
rows = [dict(model=k, H_saat=H / 2, **metrics(r['y'], r['p'], r['yc'])) for (k, H), r in sorted(results.items())]
pd.DataFrame(rows).to_csv(OUT + 'metrikler_T57_hepsi.csv', index=False)

# F) DM + HLN + Holm (Holm: tablodaki TÜM testler üzerinde)
pairs = [('B6', 'B7', 'Sızıntı: Full VMD vs Rolling VMD'), ('B8L', 'B8', 'Sızıntı: LCSF çift yönlü vs nedensel'),
         ('B9', 'B1', 'Önerilen vs Persistence'), ('B9', 'B8', 'OOF katkısı (B9 vs B8)'),
         ('B9', 'B3', 'B9 vs CatBoost Direct'), ('B8', 'B7', 'LCSF vs Rolling VMD (nedensel)'),
         ('B7', 'B2', 'Rolling VMD vs Raw LSTM'), ('B8', 'B2', 'LCSF vs Raw LSTM')]
rows = []
for m1, m2, ad in pairs:
    for H in H_LIST:
        if (m1, H) in results and (m2, H) in results:
            r1, r2 = results[(m1, H)], results[(m2, H)]
            assert np.array_equal(r1['t'], r2['t']); y = r1['y']
            r = dm_test(y, r1['p'], r2['p'], h=H, loss='abs')
            rows.append(dict(karsilastirma=ad, H_saat=H / 2, MAE_1=np.mean(np.abs(y - r1['p'])),
                             MAE_2=np.mean(np.abs(y - r2['p'])), DM=r['DM'], p_ham=r['p_raw']))
dm = pd.DataFrame(rows); dm['p_holm'] = holm(dm['p_ham'].values)
dm['anlamli'] = np.where(dm['p_holm'] < 0.05, 'Evet', 'Hayır')
print(dm.round(4).to_string(index=False)); dm.to_csv(OUT + 'adim5_dm_holm_T57.csv', index=False)

ab = []                                   # ablasyon (Holm: bu 6 test içinde)
for m1, m2, ad in [('B9', 'B2O', 'LCSF katkısı (B9 vs B2O)'), ('B9', 'B8I', 'OOF gerekli mi (B9 vs B8I)')]:
    for H in H4:
        r1, r2 = results[(m1, H)], results[(m2, H)]; y = r1['y']
        r = dm_test(y, r1['p'], r2['p'], h=H, loss='abs')
        ab.append(dict(karsilastirma=ad, H_saat=H / 2, MAE_1=np.mean(np.abs(y - r1['p'])),
                       MAE_2=np.mean(np.abs(y - r2['p'])), DM=r['DM'], p_ham=r['p_raw']))
ab = pd.DataFrame(ab); ab['p_holm'] = holm(ab['p_ham'].values)
print(ab.round(4).to_string(index=False)); ab.to_csv(OUT + 'ablation_T57.csv', index=False)
