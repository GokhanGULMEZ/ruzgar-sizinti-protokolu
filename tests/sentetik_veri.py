"""Test için SDWPF biçiminde (aynı sütunlar, 10 dk) sentetik tek-türbin verisi üretir. Gerçek veri DEĞİLDİR."""
import numpy as np, pandas as pd


def uret(gun=150, tid=1, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2020-01-01 00:10', periods=gun * 144, freq='10min')
    n = len(ts)
    r = np.zeros(n)
    for i in range(1, n): r[i] = 0.98 * r[i - 1] + rng.normal(0, 0.35)        # AR(1): rüzgâr hızı
    wspd = np.clip(7 + 3 * r + 1.5 * np.sin(2 * np.pi * np.arange(n) / 144), 0, 25)
    patv = np.clip(1500 * np.clip((wspd - 3) / 9, 0, 1) ** 3 + rng.normal(0, 20, n), 0, 1500)
    d = pd.DataFrame({'TurbID': tid, 'Tmstamp': ts, 'Patv': patv, 'Wspd': wspd})
    d['Wdir'] = rng.uniform(0, 360, n); d['Etmp'] = 15 + rng.normal(0, 2, n); d['Itmp'] = 30 + rng.normal(0, 2, n)
    d['Ndir'] = rng.uniform(0, 360, n); d['Prtv'] = patv * 0.1
    for c in ('Pab1', 'Pab2', 'Pab3'): d[c] = rng.uniform(0, 5, n)
    for c in ('T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp'): d[c] = rng.normal(10, 3, n)
    # bir günlük boşluk (gerçek veride de var): satırları sil
    d = d.drop(d.index[3000:3144]).reset_index(drop=True)
    return d
