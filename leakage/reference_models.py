"""Referans modeller: persistence, iklimsel ortalama ve Nielsen vd. (1998) referansı.

Nielsen referansı:  p_hat(t+k) = a_k * p_t + (1 - a_k) * ortalama
  a_k  : eğitim serisinin k. gecikme özilişkisi (ortalamadan sapmalar üzerinden),
  ortalama: eğitim ortalaması.   a_k -> 1 ise persistence, a_k -> 0 ise iklimsel ortalama.
Her ikisi de YALNIZ eğitim bölümünden kestirilir; test verisine bakılmaz."""
import numpy as np


def egitim_ortalamasi(p_train):
    return float(np.nanmean(p_train))


def nielsen_a(p_train, k):
    """Eğitim gücünden k. gecikme özilişkisi (geçerli çiftler üzerinden; NaN çiftler atlanır)."""
    p = np.asarray(p_train, float); m = np.nanmean(p); d = p - m
    if k == 0:
        return 1.0
    a, b = d[:-k], d[k:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    den = np.sum(a[ok] ** 2)
    return float(np.sum(a[ok] * b[ok]) / den) if den > 0 else 0.0


def nielsen_tahmin(p_now, a_k, ortalama):
    return a_k * np.asarray(p_now, float) + (1.0 - a_k) * ortalama


def referans_hatalari(p_train, y, y_cur, k):
    """Test kümesinde üç referansın MAE ve RMSE değerleri. y: hedef P(t+k), y_cur: P(t).
    Döner: {'a_k', 'persistence': (MAE, RMSE), 'iklimsel': ..., 'nielsen': ...}"""
    y = np.asarray(y, float); y_cur = np.asarray(y_cur, float)
    mu = egitim_ortalamasi(p_train); a = nielsen_a(p_train, k)
    tahmin = {'persistence': y_cur, 'iklimsel': np.full_like(y, mu), 'nielsen': nielsen_tahmin(y_cur, a, mu)}
    out = {'a_k': a}
    for ad, p in tahmin.items():
        e = y - p
        out[ad] = (float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))))
    return out
