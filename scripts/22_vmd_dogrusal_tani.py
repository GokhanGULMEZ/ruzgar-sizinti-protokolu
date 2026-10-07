"""ADIM 22 — VMD ileri-bakış eğrisinin monoton olmayışı: "LSTM eğitimi mi, öznitelik özelliği mi?" sınaması (Bölüm 6.11).
LSTM EĞİTİLMEZ; tohumsuz, deterministik bir doğrusal model (Ridge) kullanılır, bu yüzden rastgele başlangıç değişkenliği YOKTUR.
Aynı öznitelik pencereleri (37 öznitelik + 4 bileşen, L=4, H=2) her koşul için ayrı Ridge ile eğitilir; test MAE ve LSI hesaplanır.
Yorum anahtarı:
  - Ridge'de LSI eğrisi de zikzak yapıyorsa (m=8 < m=2 ve m=8 < tam seri): neden LSTM değil, bileşenlerin kendisidir
    (m=8'de hedef pencerenin içinde kalır; pencere-sonu sınır etkisi yoktur, bilgi yumuşatılmıştır).
  - Ridge'de eğri monotona yakınsa: zikzak LSTM'in bu öznitelikleri öğrenme/optimizasyon davranışından gelir.
Ek olarak her koşul için artığın (r = S_m − x) pürüzlülüğü verilir: std(Δr) ve 1. gecikme otokorelasyonu.
Önkoşul: vmd_roll_T57.npy, vmd_full_T57.npy ve vmd_ileri_T57_m{1,2,8}.npy Drive'da olmalı (VMD yeniden hesaplanmaz).
Kullanım: python 22_vmd_dogrusal_tani.py [türbin=57] [H=2]
Süre   : birkaç dakika (yalnız Ridge). Çıktı: vmd_dogrusal_tani_T57.csv"""
from _ortak import *
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from leakage.config import W, L_BEST, KAPASITE
from leakage.data import TurbineData
from leakage.pipeline import build_components
from leakage.evaluation import mae, lsi

tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
H = int(sys.argv[2]) if len(sys.argv) > 2 else 2
df = veri_yukle(); out = cikti_klasoru()
td = TurbineData(df, tid); Ut, _ = build_components(td, out, W)
x = td.x_dec; t_te = td.d.index.get_indexer(td.te.index)

kaynak = {'m=0 (kayan)': None}
for m in (1, 2, 8):
    kaynak[f'm={m}'] = os.path.join(out, f'vmd_ileri_T{tid}_m{m}.npy')
kaynak['m=inf (tam seri)'] = None

girdi, ham = {}, {}
for ad, yol in kaynak.items():
    if ad == 'm=0 (kayan)': Uham = Ut['B7']
    elif ad.startswith('m=inf'): Uham = Ut['B6']
    else:
        if not os.path.exists(yol): print('YOK', yol); continue
        U = np.load(yol).astype(float); U[:W - 1] = np.nan; Uham = td.scale_comp(U)
    girdi[ad] = td.get_xy_ext(L_BEST, H, Uham)

ref = 'm=0 (kayan)'
for ad, D in girdi.items():
    assert np.array_equal(D['te']['t'], girdi[ref]['te']['t']), ad

def ridge_mae(D):
    f = lambda k: D[k]['X'].reshape(len(D[k]['X']), -1)
    Xt, Xv, Xe = f('tr'), f('va'), f('te')
    best = None
    for a in (0.01, 0.1, 1, 10, 100):                  # alpha doğrulama MAE'sine göre seçilir
        r = Ridge(alpha=a).fit(Xt, D['tr']['y'])
        e = mae(D['va']['y'], np.clip(r.predict(Xv), 0, KAPASITE))
        if best is None or e < best[0]: best = (e, a, r)
    r = best[2]
    return mae(D['te']['y'], np.clip(r.predict(Xe), 0, KAPASITE)), best[1]

rows = []
for ad, D in girdi.items():
    e, a = ridge_mae(D); rows.append(dict(kosul=ad, MAE_ridge=e, alpha=a))
d = pd.DataFrame(rows); e0 = d.loc[d.kosul == ref, 'MAE_ridge'].iloc[0]
d['LSI_ridge'] = [0.0 if k == ref else lsi(e0, v) for k, v in zip(d.kosul, d.MAE_ridge)]

# artık pürüzlülüğü (ham bileşen dosyalarından; m=0 ve tam seri için de ham dosya varsa)
ham_dosya = {'m=0 (kayan)': f'vmd_roll_T{tid}.npy', 'm=1': f'vmd_ileri_T{tid}_m1.npy', 'm=2': f'vmd_ileri_T{tid}_m2.npy',
             'm=8': f'vmd_ileri_T{tid}_m8.npy', 'm=inf (tam seri)': f'vmd_full_T{tid}.npy'}
rz = []
t = t_te[t_te + H < len(x)]
for ad, fn in ham_dosya.items():
    yol = os.path.join(out, fn)
    if not os.path.exists(yol): continue
    U = np.load(yol).astype(float); U[:W - 1] = np.nan
    ok = ~np.isnan(U[t]).any(1); tt = t[ok]
    r = U[tt].sum(1) - x[tt]
    dr = np.diff(r); ac = np.corrcoef(r[:-1], r[1:])[0, 1]
    rz.append(dict(kosul=ad, std_r=r.std(ddof=1), std_dr=dr.std(ddof=1), ac1_r=ac))
d = d.merge(pd.DataFrame(rz), on='kosul', how='left')
d.to_csv(out + f'vmd_dogrusal_tani_T{tid}.csv', index=False)
print('\n', d.round(3).to_string(index=False)); print('yazıldı:', out + f'vmd_dogrusal_tani_T{tid}.csv')
