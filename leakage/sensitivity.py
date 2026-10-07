"""Duyarlılık analizleri: uzun eksik veri boşluklarının (ör. Nisan 2020) ileri doldurulmasının
LCSF tabanlı sızıntı ölçümüne (LSI) etkisi.

Uzun boşluk: Patv'nin art arda > UZUN_ADIM (varsayılan 48 = 24 saat) eksik olduğu bölüm.
Temel yöntem (makale): boşluk ileri doldurma (ffill) ile sabit bir platoya çevrilir.
  A  ileri doldurma (temel)
  B  uzun boşluklar eğitim ortalamasıyla doldurulur (plato yerine sabit ortalama)
  C  LCSF bantları, uzun boşlukları ÇIKARILMIŞ eğitim kesitinden öğrenilir (filtre girdisi A ile aynı)
Her yöntemde B8 (nedensel LCSF-LSTM) ve B8L (iki yönlü, sızdıran) eğitilir; LSI hesaplanır. Tohum değişkenliği büyük
olduğundan her yöntem birkaç tohumla çalıştırılır."""
import numpy as np, pandas as pd
from .config import L_BEST, W, SEED
from .data import TurbineData
from .decomposition import LCSF
from .models import fit_lstm, pred_lstm
from .evaluation import mae, lsi

UZUN_ADIM = 48


def uzun_bosluk_maskesi(patv_nan, uzun=UZUN_ADIM):
    """Art arda > uzun adım eksik olan konumlar için True."""
    m = np.zeros(len(patv_nan), bool); i = 0
    while i < len(patv_nan):
        if patv_nan[i]:
            j = i
            while j < len(patv_nan) and patv_nan[j]: j += 1
            if j - i > uzun: m[i:j] = True
            i = j
        else:
            i += 1
    return m


def nisan_duyarlilik(df_all, tid=57, H=2, seeds=(42, 43, 44), L=L_BEST, window=W, epochs=None, uzun=UZUN_ADIM):
    td = TurbineData(df_all, tid); xb = td.x_dec.copy(); ntr = len(td.tr)
    long = uzun_bosluk_maskesi(td.d['Patv'].isna().values, uzun)
    print(f'T{tid}: uzun boşluk (> {uzun} adım) içindeki satır sayısı: {int(long.sum())} (eğitimde {int(long[:ntr].sum())})')
    xm = xb.copy(); xm[long] = float(td.tr['Patv'].mean())
    ktemiz = np.delete(xb[:ntr], np.where(long[:ntr])[0])
    yontemler = {'A ileri doldurma (temel)': (xb, LCSF().fit(xb[:ntr])),
                 'B boşluk eğitim ortalamasıyla': (xm, LCSF().fit(xm[:ntr])),
                 'C bantlar boşluksuz eğitim kesitinden': (xb, LCSF().fit(ktemiz))}
    rows, t_ref = [], None
    for ad, (x, lc) in yontemler.items():
        print(f'\n{ad} | merkez frekanslar: {lc.fc.round(4)}')
        D = {}
        for key, causal in (('B8', True), ('B8L', False)):
            U = lc.transform(x, causal); U[:window - 1] = np.nan
            D[key] = td.get_xy_ext(L, H, td.scale_comp(U))
            t = D[key]['te']['t']; t_ref = t if t_ref is None else t_ref
            assert np.array_equal(t, t_ref), 'test kümesi farklı'
        for sd in seeds:
            m = {k: mae(D[k]['te']['y'], pred_lstm(fit_lstm(D[k], L, sd, epochs), D[k]['te']['X'])) for k in D}
            rows.append(dict(yontem=ad, tohum=sd, MAE_nedensel=m['B8'], MAE_sizdiran=m['B8L'], LSI_LCSF=lsi(m['B8'], m['B8L'])))
            print(f'  tohum {sd}: MAE nedensel={m["B8"]:.2f}  sızdıran={m["B8L"]:.2f}  LSI={rows[-1]["LSI_LCSF"]:.1f}%')
    d = pd.DataFrame(rows)
    ozet = d.groupby('yontem', sort=False).agg(MAE_nedensel=('MAE_nedensel', 'mean'), MAE_sizdiran=('MAE_sizdiran', 'mean'),
                                               LSI_ort=('LSI_LCSF', 'mean'), LSI_std=('LSI_LCSF', lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)).reset_index()
    return d, ozet


# --------------------------------------------------------------------------- Rolling VMD pencere uzunluğu duyarlılığı
def pencere_duyarlilik(df_all, tid=57, H=2, windows=(336, 672, 1344), seed=SEED, L=L_BEST, epochs=None, out=None):
    """Rolling VMD pencere uzunluğuna (adım; 336=7 gün, 672=14 gün, 1344=28 gün) duyarlılık.
    Her pencere için: Rolling VMD (nedensel, B7), Full VMD (sızdıran, B6) ve LCSF (B8, B8L) LSTM'leri aynı tohumla eğitilir.
    Test kümesi tüm pencerelerde aynıdır (assert ile denetlenir). Rolling VMD ön hesabı pencere başına ayrı klasörde
    önbelleklenir ('<out>/pencere_<W>/'); yarıda kalırsa kaldığı yerden sürer. W=672 için mevcut önbellek (<out>) kullanılır.
    Döner: DataFrame (pencere, MAE_B7, MAE_B6, MAE_B8, MAE_B8L, LSI_VMD, LSI_LCSF, n_test)."""
    import os, shutil, time
    from .config import OUT, W as W0
    from .decomposition import full_vmd, rolling_vmd
    out = out or OUT
    rows = []; t_ref = None
    for w in windows:
        t0 = time.time()
        d = out if w == W0 else os.path.join(out, f'pencere_{w}')
        os.makedirs(d, exist_ok=True)
        td = TurbineData(df_all, tid); tag = f'T{tid}'
        fa = os.path.join(d, f'vmd_full_{tag}.npy'); fa0 = os.path.join(out, f'vmd_full_{tag}.npy')
        if not os.path.exists(fa) and os.path.exists(fa0): shutil.copy(fa0, fa)
        uA = np.load(fa) if os.path.exists(fa) else full_vmd(td.x_dec, td.N)
        if not os.path.exists(fa): np.save(fa, uA)
        uB = rolling_vmd(td.x_dec, tag, out_dir=d, window=w)
        lcsf = LCSF().fit(td.x_dec[:len(td.tr)])
        cC = lcsf.transform(td.x_dec, True); cL = lcsf.transform(td.x_dec, False)
        uA_m = uA.copy(); uA_m[:w - 1] = np.nan; cC[:w - 1] = np.nan; cL[:w - 1] = np.nan
        Ut = {'B6': td.scale_comp(uA_m), 'B7': td.scale_comp(uB), 'B8': td.scale_comp(cC), 'B8L': td.scale_comp(cL)}
        m = {}
        for k in ('B6', 'B7', 'B8', 'B8L'):
            D = td.get_xy_ext(L, H, Ut[k])
            if t_ref is None: t_ref = D['te']['t']
            assert np.array_equal(D['te']['t'], t_ref), f'test kümesi pencereye göre değişti (W={w}, {k})'
            mdl = fit_lstm(D, L, seed, epochs)
            m[k] = mae(D['te']['y'], pred_lstm(mdl, D['te']['X']))
        rows.append(dict(pencere=w, gun=w / 48, MAE_B7=m['B7'], MAE_B6=m['B6'], MAE_B8=m['B8'], MAE_B8L=m['B8L'],
                         LSI_VMD=lsi(m['B7'], m['B6']), LSI_LCSF=lsi(m['B8'], m['B8L']), n_test=len(t_ref)))
        print(f'   pencere {w} ({w/48:.0f} gün): LSI VMD {rows[-1]["LSI_VMD"]:.1f}% | LSI LCSF {rows[-1]["LSI_LCSF"]:.1f}% | {(time.time()-t0)/60:.0f} dk')
    return pd.DataFrame(rows)
