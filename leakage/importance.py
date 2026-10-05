"""Ek C — CatBoost öznitelik önemi: sızdıran ve nedensel ayrıştırma bileşenleri aynı CatBoost'a verilince
model neye bakıyor? (Bölüm 6, Ek C). Aynı zamanda ikinci bir model ailesinde (CatBoost) LSI'yı ölçer.

Model: tahmin anındaki satır (37 öznitelik) + K ayrıştırma bileşeni -> Patv(t+H). Test kümesi B2-B9 ile aynıdır."""
import numpy as np, pandas as pd
from catboost import Pool
from .config import OUT, L_BEST, W, SEED
from .data import TurbineData, FEATURES
from .models import fit_catboost
from .pipeline import build_components
from .evaluation import mae, lsi

MODELLER = {'B3': 'CatBoost (yalnız 37 öznitelik)', 'B7': 'CatBoost + kayan VMD (nedensel)',
            'B8': 'CatBoost + LCSF (nedensel)', 'B6': 'CatBoost + tam seri VMD (sızdıran)',
            'B8L': 'CatBoost + LCSF iki yönlü (sızdıran)'}
_ERA5 = {'T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp'}
_TAKVIM = {'saat', 'haftanin_gunu', 'ay', 'yilin_gunu', 'saat_sin', 'saat_cos', 'ay_sin', 'ay_cos'}
_SENSOR = {'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'wdir_sin', 'wdir_cos', 'Wspd_rol1h'}


def grup(ad):
    if ad.startswith('Bileşen'): return 'Ayrıştırma bileşenleri'
    if ad.startswith('Patv'): return 'Güç ve gecikmeleri'
    if ad in _SENSOR: return 'Türbin sensörleri'
    if ad in _ERA5: return 'Hava (ERA5)'
    if ad in _TAKVIM: return 'Takvim'
    return 'Diğer'


def _perm(cb, X, y, cols, R=3, seed=0):
    """Permütasyon önemi: sütun(lar) karıştırılınca test MAE'sindeki artış (kW). R tekrar ortalaması."""
    rng = np.random.default_rng(seed); base = mae(y, np.clip(cb.predict(X), 0, None)); d = []
    for _ in range(R):
        Xp = X.copy(); idx = rng.permutation(len(X)); Xp[:, cols] = X[idx][:, cols]
        d.append(mae(y, np.clip(cb.predict(Xp), 0, None)) - base)
    return float(np.mean(d))


def ek_c(df_all, tid=57, H=2, out=OUT, L=L_BEST, seed=SEED, window=W, R=3, n_shap=2000):
    """Beş CatBoost'u eğitir; döndürür: (ozet, onem, shap_veri).
    SHAP: CatBoost'un kendi ağaç SHAP'i, test kümesinden eşit aralıklı en çok n_shap örnek üzerinde (zaman sıralı).
    shap_veri[kod] = (X, shap_degerleri, oznitelik_adlari)"""
    td = TurbineData(df_all, tid)
    Ut, _ = build_components(td, out, window)
    ozet, onem, preds, shap_veri = [], [], {}, {}
    for key, ad in MODELLER.items():
        D = td.get_xy(L, H) if key == 'B3' else td.get_xy_ext(L, H, Ut[key])
        Xt, yt = D['tr']['X'][:, -1, :], D['tr']['y']
        Xv, yv = D['va']['X'][:, -1, :], D['va']['y']
        Xe, ye = D['te']['X'][:, -1, :], D['te']['y']
        K = Xe.shape[1] - len(FEATURES)
        names = FEATURES + [f'Bileşen{i+1}' for i in range(K)]
        cb = fit_catboost(Xt, yt, Xv, yv, seed)
        p = np.clip(cb.predict(Xe), 0, None); preds[key] = (ye, p)
        pvc = cb.get_feature_importance(); pvc = 100 * pvc / pvc.sum()          # PredictionValuesChange, toplam 100
        perm = np.array([_perm(cb, Xe, ye, [j], R) for j in range(len(names))])
        gperm = _perm(cb, Xe, ye, list(range(len(FEATURES), len(names))), R) if K else 0.0
        ix = np.linspace(0, len(Xe) - 1, min(n_shap, len(Xe))).astype(int)
        sv = cb.get_feature_importance(Pool(Xe[ix]), type='ShapValues')[:, :-1]          # son sütun = beklenen değer
        shap_veri[key] = (Xe[ix], sv, names); msh = np.abs(sv).mean(0)
        for n, a, b, c in zip(names, pvc, perm, msh):
            onem.append(dict(model=ad, kod=key, oznitelik=n, grup=grup(n), onem_pvc=a, perm_dMAE_kW=b, shap_ort_mutlak_kW=c))
        g = pd.DataFrame(dict(grup=[grup(n) for n in names], pvc=pvc))
        pay = g.groupby('grup')['pvc'].sum()
        shp = pd.Series(msh, index=names); shp_pay = 100 * shp[[n for n in names if grup(n) == 'Ayrıştırma bileşenleri']].sum() / shp.sum()
        ozet.append(dict(model=ad, kod=key, K=K, MAE=mae(ye, p), bilesen_pay_pvc=pay.get('Ayrıştırma bileşenleri', 0.0),
                         guc_pay_pvc=pay.get('Güç ve gecikmeleri', 0.0), bilesen_grup_perm_dMAE=gperm, bilesen_pay_shap=shp_pay))
        print(f'  {ad:40s} MAE={ozet[-1]["MAE"]:7.2f}  bileşen payı={ozet[-1]["bilesen_pay_pvc"]:5.1f}%  '
              f'güç payı={ozet[-1]["guc_pay_pvc"]:5.1f}%  permütasyon ΔMAE={gperm:6.2f} kW  SHAP bileşen payı={shp_pay:5.1f}%')
    ozet = pd.DataFrame(ozet)
    m = ozet.set_index('kod')['MAE']
    print(f'  CatBoost LSI-VMD  = {lsi(m["B7"], m["B6"]):.1f}%   |   LSI-LCSF = {lsi(m["B8"], m["B8L"]):.1f}%')
    ozet.attrs['LSI_VMD'] = lsi(m['B7'], m['B6']); ozet.attrs['LSI_LCSF'] = lsi(m['B8'], m['B8L'])
    return ozet, pd.DataFrame(onem), shap_veri
