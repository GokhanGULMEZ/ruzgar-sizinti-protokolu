"""ADIM 13 — Makaledeki sayıların ham sonuç dosyalarından yeniden hesaplanıp denetlenmesi (hakem sorusu: "LSI değerleri
hangi ham MAE çiftlerinden üretildi?").
Kullanım:  python 13_tablo_tutarlilik.py            (RT_OUT içindeki adim6_T*.pkl, seed_T*.pkl ve duyarlilik_*.csv okunur)
Denetler  : (1) Tablo 4/5: altı türbin, 1 saat, tohum 42 — MAE ve LSI makale değerleriyle aynı mı;
            (2) Tablo 6: beş türbin, üç tohum — her tohumun LSI'si ve ortalama ± std (n=3, ddof=1);
            (3) Tablo 11/12: duyarlılık CSV'lerinde LSI, aynı satırdaki ham MAE çiftinden yeniden hesaplanır.
Tolerans  : MAE 0,05 kW, LSI 0,1 puan (makale 1 ondalık yazar). Çıktı: tutarlilik_raporu.csv; FARK varsa çıkış kodu 1."""
from _ortak import *
import os, sys, pickle, numpy as np, pandas as pd
from leakage.config import OUT
from leakage.evaluation import lsi, mae

# Makalede yazılı değerler (Tablo 4/5, tohum 42, H=1 saat): MAE nedensel Rolling VMD, sızdıran Full VMD, nedensel LCSF, sızdıran LCSF; LSI VMD, LSI LCSF
MAKALE = {57: (116.11, 64.71, 116.29, 57.34, 44.3, 50.7), 10: (128.17, 69.58, 127.71, 55.57, 45.7, 56.5),
          83: (111.64, 61.67, 111.34, 57.89, 44.8, 48.0), 120: (95.59, 63.42, 96.83, 57.29, 33.7, 40.8),
          61: (119.48, 70.15, 119.26, 56.01, 41.3, 53.0), 84: (116.34, 72.61, 110.15, 55.03, 37.6, 50.0)}
TOL_MAE, TOL_LSI = 0.05, 0.1
satirlar = []


def kayit(kontrol, anahtar, makale, hesap, tol):
    ok = abs(makale - hesap) <= tol + 1e-9
    satirlar.append(dict(kontrol=kontrol, anahtar=anahtar, makale=makale, yeniden_hesap=round(hesap, 3), durum='TAMAM' if ok else 'FARK'))
    print(f'  {"TAMAM" if ok else "FARK ":5s} {kontrol:10s} {anahtar:28s} makale {makale:8.2f}  hesap {hesap:8.3f}')


print('(1) Tablo 4/5 — altı türbin, tohum 42')
for tid, (a, b, c, d, lv, ll) in MAKALE.items():
    yol = OUT + f'adim6_T{tid}.pkl'
    if not os.path.exists(yol): print(f'  YOK   {yol}'); continue
    r = pickle.load(open(yol, 'rb'))
    m = {k: mae(r[(k, 2)]['y'], r[(k, 2)]['p']) for k in ('B7', 'B6', 'B8', 'B8L')}
    for ad, mk, hs in (('Rolling VMD', a, m['B7']), ('Full VMD', b, m['B6']), ('LCSF ned.', c, m['B8']), ('LCSF çift', d, m['B8L'])):
        kayit('MAE', f'T{tid} {ad}', mk, hs, TOL_MAE)
    kayit('LSI-VMD', f'T{tid}', lv, lsi(m['B7'], m['B6']), TOL_LSI); kayit('LSI-LCSF', f'T{tid}', ll, lsi(m['B8'], m['B8L']), TOL_LSI)

print('\n(2) Tablo 6 — tohum tekrarı (42, 43, 44): LSI ortalama ± std')
MAKALE5 = {10: ((43.4, 4.3), (55.8, 0.8)), 83: ((46.6, 1.6), (54.2, 5.4)), 120: ((39.4, 5.0), (48.9, 7.0)),
           61: ((40.5, 3.8), (50.2, 4.2)), 84: ((39.6, 2.6), (51.3, 2.1))}
for tid, (v, l) in MAKALE5.items():
    try:
        from leakage.pipeline import seed_ozet
        o = seed_ozet(tid, verbose=False)
    except Exception as e:
        print(f'  YOK   T{tid}: {e}'); continue
    for ad, (mo, sd), arr in (('VMD', v, o['LSI_VMD']), ('LCSF', l, o['LSI_LCSF'])):
        kayit('LSI ort', f'T{tid} {ad}', mo, float(arr.mean()), TOL_LSI); kayit('LSI std', f'T{tid} {ad}', sd, float(arr.std(ddof=1)), TOL_LSI)

print('\n(3) Duyarlılık CSV\'leri — LSI, ham MAE çiftinden')
f = OUT + 'duyarlilik_nisan2020_T57.csv'
if os.path.exists(f):
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        kayit('LSI Nisan', f'{r["yontem"]} tohum {int(r["tohum"])}', float(r['LSI_LCSF']), lsi(r['MAE_nedensel'], r['MAE_sizdiran']), TOL_LSI)
f = OUT + 'duyarlilik_pencere_T57.csv'
if os.path.exists(f):
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        kayit('LSI-VMD pnc', f'W={int(r["pencere"])}', float(r['LSI_VMD']), lsi(r['MAE_B7'], r['MAE_B6']), TOL_LSI)
        kayit('LSI-LCSF pnc', f'W={int(r["pencere"])}', float(r['LSI_LCSF']), lsi(r['MAE_B8'], r['MAE_B8L']), TOL_LSI)

R = pd.DataFrame(satirlar); R.to_csv(OUT + 'tutarlilik_raporu.csv', index=False)
fark = int((R['durum'] == 'FARK').sum()) if len(R) else 0
print(f'\nToplam {len(R)} kontrol, FARK: {fark}. Rapor: {OUT}tutarlilik_raporu.csv'); sys.exit(1 if fark else 0)
