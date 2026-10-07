"""Kayan VMD için ileri-bakış doz–tepki deneyi (LCSF'deki Bölüm 6.10 deneyinin VMD karşılığı).
  m = 0 : ana deneydeki kayan VMD — pencere [t-W+1, t]; t pencerenin SON noktasıdır (VMD'nin sınır etkisi burada en büyüktür).
  m > 0 : pencere [t-W+1, t+m]; t pencerenin sonundan m adım içeridedir, yani sınır etkisinden uzaklaşır AMA gelecek bilgi girer.
  m = ∞ : tam seri VMD (ana deneydeki sızdıran koşul).
Yorum: LSI(m) küçük m için çok artıyorsa fark gelecek bilgisinden; m=0 ile m=∞ arasındaki farkın büyük bölümü m'nin küçük
değerlerinde kapanmıyorsa sınır etkisi payı büyüktür. m=H hedefin kendisini bileşene sokar (üst sınır, mekanizma kontrolü).
Uyarı: vmdpy tek uzunluklu girdinin son örneğini atar; bu yüzden uzunluk tekse İLK örnek atılır ve konum buna göre kaydırılır."""
import os, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from .config import W, K, OUT, L_BEST
from .data import TurbineData
from .decomposition import vmd_sorted, blok_temizle
from .models import fit_lstm, pred_lstm
from .evaluation import mae, lsi
from .pipeline import build_components


def _mod_t(x, t, m, window):
    N = len(x); bas = t - window + 1; son = min(N, t + m + 1)
    if (son - bas) % 2: bas += 1                 # vmdpy: tek uzunluk -> son örnek atılır; biz ilk örneği atarız
    return vmd_sorted(x[bas:son])[:, t - bas]


def vmd_ileri_bilesen(x_dec, tag, m, out_dir=OUT, window=W, block=2000, n_jobs=-1):
    """t >= window-1 için VMD(x[t-window+1 : t+m+1]) içindeki t anındaki mod değerleri. Bloklar diske yazılır (kaldığı yerden sürer).
    Süre: m=0 ile aynı mertebede (~1 saat/türbin, tek çekirdek)."""
    N = len(x_dec); fn_all = os.path.join(out_dir, f'vmd_ileri_{tag}_m{m}.npy')
    if os.path.exists(fn_all): return np.load(fn_all)
    U = np.full((N, K), np.nan)
    for s in range(window - 1, N, block):
        e = min(s + block, N); fn = os.path.join(out_dir, f'vmdileri_{tag}_m{m}_blk_{s}.npy')
        if os.path.exists(fn): U[s:e] = np.load(fn); continue
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(_mod_t)(x_dec, t, m, window) for t in range(s, e))
        blk = blok_temizle(res); np.save(fn, blk); U[s:e] = blk
        print(f'   VMD ileri m={m} {tag}: {e}/{N} | {time.time()-t0:.0f} sn', flush=True)
    np.save(fn_all, U); return U


def vmd_ileri_egrisi(df_all, tid=57, H=2, ms=(1, 2, 8), seeds=(42, 43, 44), L=L_BEST, window=W, out=OUT, epochs=None):
    """Her m için VMD–LSTM eğitilir. Referanslar: kayan VMD (m=0, B7) ve tam seri VMD (B6) — build_components önbelleğinden.
    LSI_m = (MAE_kayan − MAE_m) / MAE_kayan × 100. Döner: (tohum başına tablo, özet)."""
    td = TurbineData(df_all, tid); Ut, _ = build_components(td, out, window)
    girdi = {'m=0 (kayan, ana deney)': td.get_xy_ext(L, H, Ut['B7'])}
    for m in ms:
        U = vmd_ileri_bilesen(td.x_dec, f'T{tid}', m, out, window); U[:window - 1] = np.nan
        girdi[f'm={m}'] = td.get_xy_ext(L, H, td.scale_comp(U))
    girdi['m=inf (tam seri)'] = td.get_xy_ext(L, H, Ut['B6'])
    t_ref = girdi['m=0 (kayan, ana deney)']['te']['t']
    for ad, D in girdi.items():
        assert np.array_equal(D['te']['t'], t_ref), f'test kümesi farklı ({ad})'
    ref = 'm=0 (kayan, ana deney)'; rows = []
    for sd in seeds:
        M = {ad: mae(D['te']['y'], pred_lstm(fit_lstm(D, L, sd, epochs), D['te']['X'])) for ad, D in girdi.items()}
        for ad in girdi:
            rows.append(dict(tohum=sd, kosul=ad, MAE=M[ad], LSI=(0.0 if ad == ref else lsi(M[ref], M[ad]))))
        print(f'  tohum {sd}: ' + ' | '.join(f'{ad} {M[ad]:.1f}' for ad in girdi), flush=True)
    d = pd.DataFrame(rows)
    ozet = d.groupby('kosul', sort=False).agg(MAE=('MAE', 'mean'), LSI_ort=('LSI', 'mean'),
                                              LSI_std=('LSI', lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)).reset_index()
    return d, ozet
