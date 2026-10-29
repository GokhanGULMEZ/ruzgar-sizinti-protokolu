"""Ek paketin ortak yardimcilari. v1 paketine (ruzgar_sizinti_paketi) DOKUNMAZ; yalniz 'leakage' modulunu OKUR.
Ortam degiskenleri:
  RT_PKG : 'leakage' klasorunun oldugu dizin (v1 paketi)
  RT_OUT : v1 paketinin sonuc klasoru (adim6_T*.pkl, seed_T*.pkl, cok_ufuk_T*.pkl burada; SADECE OKUNUR)
  RT_EK_OUT : bu paketin ciktilari (varsayilan: RT_OUT + 'ek_cikti/')  -> v1 dosyalari ezilmez."""
import os, sys, pickle
import numpy as np

PKG = os.environ.get('RT_PKG')
if not PKG or not os.path.isdir(os.path.join(PKG, 'leakage')):
    sys.exit("RT_PKG verilmedi veya icinde 'leakage' yok. Ornek: %env RT_PKG=/content/drive/MyDrive/.../ruzgar_sizinti_paketi")
sys.path.insert(0, PKG)
from leakage.evaluation import dm_test, holm   # noqa: E402  (yalniz numpy/scipy gerektirir)

OUT = os.environ.get('RT_OUT', '/content/drive/MyDrive/RuzgarTahmin/yeni/')
EK_OUT = os.environ.get('RT_EK_OUT', os.path.join(OUT, 'ek_cikti') + os.sep)
os.makedirs(EK_OUT, exist_ok=True)


def yukle(ad):
    yol = os.path.join(OUT, ad)
    return pickle.load(open(yol, 'rb')) if os.path.exists(yol) else None


def lsi(e_nedensel, e_sizdiran):
    return 100.0 * (e_nedensel - e_sizdiran) / e_nedensel


def blok_indeks(n, blok, B, rng):
    """Hareketli blok bootstrap indeksleri (B x n)."""
    nb = int(np.ceil(n / blok))
    st = rng.integers(0, n - blok + 1, size=(B, nb))
    return (st[:, :, None] + np.arange(blok)).reshape(B, -1)[:, :n]
