"""ADIM 20 — VMD ileri-bakış eğrisinin (Bölüm 6.11) monoton olmayışı için hızlı tanı (LSTM EĞİTİLMEZ, birkaç dakika).
Önkoşul: vmd_roll_T57.npy, vmd_full_T57.npy ve 17. adımın bileşenleri (vmd_ileri_T57_m{m}.npy; m=1,2,8) Drive'da olmalı.
Fikir : Bileşenlerin t anındaki toplamı S_m(t), gözlenen x(t)'yi yeniden kurar. Artık r_m(t) = S_m(t) − x(t) gelecek bilgisi taşıyorsa,
        hedefe giden değişimle Δ(t) = x(t+H) − x(t) ilişkili olur. Her koşul için şunlar hesaplanır (yalnız test noktaları):
        std(r), corr(r, Δ) ve eğim β (Δ'nın r üzerine regresyonu). Beklenti: sızıntılı koşullarda corr/β büyük, kayan VMD'de (m=0) ~0.
Kullanım: python 20_vmd_uc_nokta_tani.py [türbin=57] [H=2]
Çıktı : vmd_tani_T57.csv (ekrana da yazılır)"""
from _ortak import *
import sys, numpy as np, pandas as pd
from leakage.config import OUT, W
from leakage.data import TurbineData


def tani(x, U, t_idx, H):
    """x: gözlenen seri, U: (N, K) bileşenler, t_idx: test zamanları. Döner: std(r), corr(r, Δ), eğim."""
    t = np.asarray(t_idx); t = t[(t + H < len(x))]
    S = np.nansum(U[t], axis=1); ok = ~np.isnan(U[t]).any(1)
    t, S = t[ok], S[ok]
    r = S - x[t]; delta = x[t + H] - x[t]
    c = np.corrcoef(r, delta)[0, 1]; beta = np.cov(r, delta)[0, 1] / np.var(r, ddof=1)
    return r.std(ddof=1), c, beta, len(t)


if __name__ == '__main__':
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    df = veri_yukle(); out = cikti_klasoru()
    td = TurbineData(df, tid); x = td.x_dec
    t_idx = td.d.index.get_indexer(td.te.index)
    kaynak = [('m=0 (kayan VMD)', f'vmd_roll_T{tid}.npy')] + [(f'm={m}', f'vmd_ileri_T{tid}_m{m}.npy') for m in (1, 2, 8)] \
             + [('m=inf (tam seri)', f'vmd_full_T{tid}.npy')]
    rows = []
    for ad, fn in kaynak:
        yol = out + fn
        if not os.path.exists(yol): print('YOK', yol); continue
        U = np.load(yol).astype(float); U[:W - 1] = np.nan
        sd, c, b, n = tani(x, U, t_idx, H)
        rows.append(dict(kosul=ad, n=n, std_artik=sd, corr_artik_delta=c, egim=b))
        print(f'{ad:18s} n={n}  std(r)={sd:8.2f} kW  corr(r,Δ)={c:6.3f}  eğim={b:6.3f}')
    pd.DataFrame(rows).to_csv(out + f'vmd_tani_T{tid}.csv', index=False); print('yazıldı:', out + f'vmd_tani_T{tid}.csv')
