"""ADIM 1 — Geçmiş pencere uzunluğunu doğrulama kümesinde seç (Bölüm 4.1).
Raw LSTM, ufuk = 1 saat (H=2); aday L ∈ {4, 8, 12} (2/4/6 saat). Seçim YALNIZ doğrulama MAE'sine göre.
Beklenen: L=4 (2 saat). Süre: ~10-20 dk (CPU)."""
from _ortak import *
import time, numpy as np
from leakage.data import TurbineData
from leakage.models import fit_lstm, pred_lstm
from leakage.config import SEQ_OPTS

df = veri_yukle(); td = TurbineData(df, 57)
sec = {}
for L in SEQ_OPTS:
    t0 = time.time(); D = td.get_xy(L, 2); m = fit_lstm(D, L)
    p = pred_lstm(m, D['va']['X']); sec[L] = float(np.mean(np.abs(D['va']['y'] - p)))
    print(f'L={L} ({L/2} sa): doğrulama MAE={sec[L]:.2f} | süre={time.time()-t0:.0f} sn')
print('\nSeçilen geçmiş pencere: L =', min(sec, key=sec.get), '→ config.L_BEST ile aynı olmalı')
