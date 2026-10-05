"""Kayan VMD için 'geriye kaydırılmış nedensel' sınır-etkisi kontrolü (Bölüm 6.11'in tamamlayıcısı).
Pencere ana deneydeki gibi [t-W+1, t] aralığında biter (gelecek bilgisi YOK); fakat bileşen penceresinin SON noktasından değil,
d adım gerisinden alınır: u(t-d). Böylece bileşen pencere sonundaki sınır etkisinden uzaklaşır, hâlâ tamamen nedenseldir.
  d = 0 : ana deneydeki kayan VMD.
Yorum: MAE d>0'da d=0'a göre belirgin düşüyorsa sınır etkisi gerçekten büyüktür (ve LSI-VMD'ye karışmıştır).
        Düşmüyorsa (veya artıyorsa) fark ağırlıkla gelecek bilgisinden gelir. Uyarı: bileşen d adım ESKİDİR (güncellik kaybı);
        bu yüzden küçük d (1, 2, 4) ve izlenen yön birlikte yorumlanmalıdır, tek başına 'sınır etkisi yoktur' kanıtı değildir.
Bir pencere için tek VMD yapılır ve tüm d değerlerinin konumları aynı çözümden okunur (maliyet ≈ ana kayan VMD kadar)."""
import os, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from .config import W, K, OUT, L_BEST
from .data import TurbineData
from .decomposition import vmd_sorted
from .models import fit_lstm, pred_lstm
from .evaluation import mae
from .pipeline import build_components


def _modlar(x, t, ds, window):
    u = vmd_sorted(x[t - window + 1:t + 1])            # (K, window)
    return np.stack([u[:, window - 1 - d] for d in ds])  # (len(ds), K)


def geri_bilesen(x_dec, tag, ds, out_dir=OUT, window=W, block=2000, n_jobs=-1):
    """Döner: {d: (N, K) dizisi}. Bloklar diske yazılır (kaldığı yerden sürer). Süre ≈ ana kayan VMD (~1 saat, tek çekirdek)."""
    N = len(x_dec); ads = '-'.join(map(str, ds))
    fn_all = os.path.join(out_dir, f'vmd_geri_{tag}_d{ads}.npy')
    if os.path.exists(fn_all):
        A = np.load(fn_all)
    else:
        A = np.full((N, len(ds), K), np.nan)
        for s in range(window - 1, N, block):
            e = min(s + block, N); fn = os.path.join(out_dir, f'vmdgeri_{tag}_d{ads}_blk_{s}.npy')
            if os.path.exists(fn): A[s:e] = np.load(fn); continue
            t0 = time.time()
            res = Parallel(n_jobs=n_jobs)(delayed(_modlar)(x_dec, t, ds, window) for t in range(s, e))
            blk = np.nan_to_num(np.array(res)); np.save(fn, blk); A[s:e] = blk
            print(f'   VMD geri {tag}: {e}/{N} | {time.time()-t0:.0f} sn', flush=True)
        np.save(fn_all, A)
    return {d: A[:, i, :] for i, d in enumerate(ds)}


def geri_egrisi(df_all, tid=57, H=2, ds=(1, 2, 4), seeds=(42, 43, 44), L=L_BEST, window=W, out=OUT, epochs=None):
    """Her d için VMD–LSTM eğitilir. Referanslar: d=0 (kayan VMD, B7) ve tam seri VMD (B6).
    IYILESME_d = (MAE_0 − MAE_d)/MAE_0 × 100 (pozitif: sınır etkisinden uzaklaşmak hatayı azaltıyor). Döner: (tablo, özet)."""
    td = TurbineData(df_all, tid); Ut, _ = build_components(td, out, window)
    girdi = {'d=0 (kayan, ana deney)': td.get_xy_ext(L, H, Ut['B7'])}
    for d, U in geri_bilesen(td.x_dec, f'T{tid}', ds, out, window).items():
        U = U.copy(); U[:window - 1] = np.nan
        girdi[f'd={d}'] = td.get_xy_ext(L, H, td.scale_comp(U))
    girdi['tam seri (sızdıran)'] = td.get_xy_ext(L, H, Ut['B6'])
    t_ref = girdi['d=0 (kayan, ana deney)']['te']['t']
    for ad, D in girdi.items():
        assert np.array_equal(D['te']['t'], t_ref), f'test kümesi farklı ({ad})'
    ref = 'd=0 (kayan, ana deney)'; rows = []
    for sd in seeds:
        M = {ad: mae(D['te']['y'], pred_lstm(fit_lstm(D, L, sd, epochs), D['te']['X'])) for ad, D in girdi.items()}
        for ad in girdi:
            rows.append(dict(tohum=sd, kosul=ad, MAE=M[ad], IYILESME=(0.0 if ad == ref else 100 * (M[ref] - M[ad]) / M[ref])))
        print(f'  tohum {sd}: ' + ' | '.join(f'{ad} {M[ad]:.1f}' for ad in girdi), flush=True)
    d = pd.DataFrame(rows)
    ozet = d.groupby('kosul', sort=False).agg(MAE=('MAE', 'mean'), IYILESME_ort=('IYILESME', 'mean'),
                                              IYILESME_std=('IYILESME', lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)).reset_index()
    return d, ozet
