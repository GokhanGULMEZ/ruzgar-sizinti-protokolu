"""İleri-bakış (look-ahead) doz–tepki deneyi: LCSF bantları her t anında YALNIZ x[..t+m] üzerinde, sıfır fazlı (filtfilt) uygulanır.
  m = 0  : sıfır fazlı süzme, gelecek YOK (yalnız geçmiş; uç etkisi ve genlik/faz farkı var)
  m = H  : t+H adımındaki hedef Patv(t+H) bileşene dahil
  m = ∞  : tam seri filtfilt (ana deneydeki 'iki yönlü' sızdıran LCSF)
Amaç: iki yönlü LCSF ile nedensel LCSF arasındaki farkın (i) gelecek bilgisinden mi, (ii) süzmenin genlik/faz farkından mı
geldiğini ayırmak. m=0 sıfır fazlı ama gelecek içermez; LSI(m)'nin m ile artması sızıntının gelecek bilgisinden kaynaklandığını gösterir."""
import numpy as np, pandas as pd
from scipy.signal import filtfilt
from .config import L_BEST, W, SEED
from .data import TurbineData
from .decomposition import LCSF, K
from .models import fit_lstm, pred_lstm
from .evaluation import mae, lsi

GERI = 720   # filtre geçişi için geriye bakılan adım (IIR geçici etkisi bu uzunlukta sönmüştür)


def lcsf_ileri_bakis(lcsf, x, m, bas=W - 1, geri=GERI):
    """t >= bas için bileşen(t) = filtfilt(x[t-geri+1 : t+m+1])[t]. t < bas için NaN. m=0 ise gelecek örnek kullanılmaz."""
    N = len(x); out = np.full((N, K), np.nan)
    for t in range(bas, N):
        a, b = max(0, t - geri + 1), min(N, t + m + 1)
        seg = x[a:b]
        for k, (bb, aa) in enumerate(lcsf.bands):
            out[t, k] = filtfilt(bb, aa, seg)[t - a]
    return out


def ileri_bakis_egrisi(df_all, tid=57, H=2, ms=(0, 1, 2, 4, 8, 16, 48), seeds=(42, 43, 44), L=L_BEST, window=W, epochs=None):
    """Her m için LCSF-LSTM eğitilir (aynı tohumlar). Referanslar: nedensel LCSF (B8) ve tam seri iki yönlü LCSF (B8L).
    Döner: (tohum başına tablo, özet). LSI_m = (MAE_nedensel − MAE_m) / MAE_nedensel × 100."""
    td = TurbineData(df_all, tid); lc = LCSF().fit(td.x_dec[:len(td.tr)]); x = td.x_dec
    girdi = {}
    U = lc.transform(x, True); U[:window - 1] = np.nan; girdi['nedensel'] = td.get_xy_ext(L, H, td.scale_comp(U))
    U = lc.transform(x, False); U[:window - 1] = np.nan; girdi['m=inf (tam seri)'] = td.get_xy_ext(L, H, td.scale_comp(U))
    for m in ms:
        U = lcsf_ileri_bakis(lc, x, m, bas=window - 1); girdi[f'm={m}'] = td.get_xy_ext(L, H, td.scale_comp(U))
        print(f'   m={m} bileşenleri hazır')
    t_ref = girdi['nedensel']['te']['t']
    for ad, D in girdi.items():
        assert np.array_equal(D['te']['t'], t_ref), f'test kümesi farklı ({ad})'
    rows = []
    for sd in seeds:
        M = {ad: mae(D['te']['y'], pred_lstm(fit_lstm(D, L, sd, epochs), D['te']['X'])) for ad, D in girdi.items()}
        for ad in girdi:
            rows.append(dict(tohum=sd, kosul=ad, m=(np.inf if 'inf' in ad else (-1 if ad == 'nedensel' else int(ad[2:]))),
                             MAE=M[ad], LSI=(0.0 if ad == 'nedensel' else lsi(M['nedensel'], M[ad]))))
        print(f'  tohum {sd}: ' + ' | '.join(f'{ad} {M[ad]:.1f}' for ad in girdi))
    d = pd.DataFrame(rows)
    ozet = d.groupby('kosul', sort=False).agg(m=('m', 'first'), MAE=('MAE', 'mean'), LSI_ort=('LSI', 'mean'),
                                              LSI_std=('LSI', lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)).reset_index()
    return d, ozet
