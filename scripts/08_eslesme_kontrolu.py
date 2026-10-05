"""ADIM 8 — Paket kodunun özgün Colab sonuçlarıyla eşleştiğini doğrula (Türbin 10 örneği).

Neden: Paket, makale sonuçlarını üreten defterlerden yeniden düzenlendi. Bu betik, paketin ürettiği
adim6_T{tid}.pkl ile özgün defterin ürettiği kayıtlı dosyayı model model karşılaştırır.

Kullanım (RT_OUT içinde özgün adim6_T10.pkl varken):
  1) python scripts/08_eslesme_kontrolu.py yedekle 10     # özgün dosyayı *_ESKI_yedek.pkl olarak sakla
  2) python scripts/02_turbin_kosusu.py 10                # paketle yeniden üret (önbellekten ~10-20 dk)
  3) python scripts/08_eslesme_kontrolu.py karsilastir 10 # karşılaştır

Beklenen: test kümesi aynı, MAE farkı %0,00, maksimum tahmin farkı 0,000
(5 Ekim 2026'da T10 için bu sonuç alındı, Colab Python 3.13)."""
from _ortak import *
import sys, shutil, pickle, numpy as np
from leakage.config import OUT

komut = sys.argv[1] if len(sys.argv) > 1 else 'karsilastir'
tid = int(sys.argv[2]) if len(sys.argv) > 2 else 10
yol, yedek = OUT + f'adim6_T{tid}.pkl', OUT + f'adim6_T{tid}_ESKI_yedek.pkl'
if komut == 'yedekle':
    shutil.copy(yol, yedek); print('yedek alındı:', yedek)
else:
    eski, yeni = pickle.load(open(yedek, 'rb')), pickle.load(open(yol, 'rb'))
    for k in ['B1', 'B2', 'B3', 'B6', 'B7', 'B8', 'B8L', 'B9']:
        a, b = eski[(k, 2)], yeni[(k, 2)]
        me = np.mean(np.abs(a['y'] - a['p'])); my = np.mean(np.abs(b['y'] - b['p']))
        print(f"{k:4s} test kümesi aynı: {np.array_equal(a['t'], b['t'])} | MAE eski {me:8.2f} yeni {my:8.2f} "
              f"fark {100*(my-me)/me:+.2f}% | maks tahmin farkı {np.abs(a['p']-b['p']).max():.3f}")
