"""Tahmin modelleri (makale Bölüm 4.6). TensorFlow ve CatBoost burada import edilir."""
import numpy as np
from catboost import CatBoostRegressor
from .config import (KAPASITE, SEED, LSTM_UNITS, LSTM_DROPOUT, LSTM_LR, LSTM_EPOCHS, LSTM_BATCH,
                     LSTM_PATIENCE, CB_PARAMS, OOF_BLOCKS)


def _tf():
    import tensorflow as tf
    return tf


def build_lstm(L, F):
    """2 katmanlı LSTM (64 -> 32), dropout 0.2, Adam 1e-3, MSE."""
    tf = _tf()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    m = Sequential([Input(shape=(L, F)),
                    LSTM(LSTM_UNITS[0], return_sequences=True), Dropout(LSTM_DROPOUT),
                    LSTM(LSTM_UNITS[1]), Dropout(LSTM_DROPOUT), Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(LSTM_LR), loss='mse')
    return m


def fit_arrays(Xt, yt, Xv, yv, L, seed=SEED, epochs=None):
    """Dizilerle LSTM eğit. Hedef seviye: y/KAPASITE. Erken durdurma doğrulamada (patience=5)."""
    tf = _tf()
    from tensorflow.keras.callbacks import EarlyStopping
    tf.keras.utils.set_random_seed(seed)
    m = build_lstm(L, Xt.shape[2])
    m.fit(Xt, yt / KAPASITE, validation_data=(Xv, yv / KAPASITE),
          epochs=epochs or LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0,
          callbacks=[EarlyStopping(patience=LSTM_PATIENCE, restore_best_weights=True)])
    return m


def fit_lstm(D, L, seed=SEED, epochs=None):
    """D = TurbineData.get_xy*(...) çıktısı."""
    return fit_arrays(D['tr']['X'], D['tr']['y'], D['va']['X'], D['va']['y'], L, seed, epochs)


def pred_lstm(m, X):
    return np.clip(m.predict(X, verbose=0).flatten() * KAPASITE, 0, None)


def fit_catboost(Xt, yt, Xv, yv, seed=SEED):
    cb = CatBoostRegressor(random_seed=seed, **CB_PARAMS)
    cb.fit(Xt, yt, eval_set=(Xv, yv))
    return cb


def catboost_direct(D, seed=SEED):
    """B3: yalnız tahmin anındaki satırdan (penceredeki SON zaman adımı) doğrudan tahmin."""
    cb = fit_catboost(D['tr']['X'][:, -1, :], D['tr']['y'], D['va']['X'][:, -1, :], D['va']['y'], seed)
    return np.clip(cb.predict(D['te']['X'][:, -1, :]), 0, None)


def oof_predictions(Dtr, L, H, n_blocks=OOF_BLOCKS, seed=SEED, epochs=None):
    """Out-of-fold LSTM tahminleri (ileri zincirleme): eğitimin blok j'si, yalnız ondan ÖNCEKİ veriyle
    eğitilmiş LSTM ile tahmin edilir; eğitim ile tahmin bloğu arası boşluk = H+L (sızıntı yok).
    İlk blok için OOF yoktur (NaN)."""
    n = len(Dtr['y']); ed = np.linspace(0, n, n_blocks + 1).astype(int)
    oof = np.full(n, np.nan); gap = H + L
    for j in range(1, n_blocks):
        te_end = ed[j] - gap
        cut = int(te_end * 0.85)                      # erken durdurma kümesi: zaman sıralı son %15
        m = fit_arrays(Dtr['X'][:cut], Dtr['y'][:cut], Dtr['X'][cut:te_end], Dtr['y'][cut:te_end],
                       L, seed, epochs)
        oof[ed[j]:ed[j + 1]] = pred_lstm(m, Dtr['X'][ed[j]:ed[j + 1]])
    return oof


def stack(D, base, L, H, mode, seed=SEED, epochs=None):
    """LSTM + kalıntı CatBoost.
    mode='oof'      : kalıntı modeli OOF tahminlerle eğitilir (B9, B2O)
    mode='insample' : eğitim-içi LSTM tahminleriyle (B8I; ablasyon)
    Girdi: son zaman adımı öznitelikleri + LSTM tahmini; hedef: y - LSTM tahmini."""
    F = lambda X, p: np.column_stack([X[:, -1, :], p])
    if mode == 'oof':
        oof = oof_predictions(D['tr'], L, H, seed=seed, epochs=epochs); ok = ~np.isnan(oof)
    else:
        oof = pred_lstm(base, D['tr']['X']); ok = np.ones(len(oof), bool)
    base_va = pred_lstm(base, D['va']['X']); base_te = pred_lstm(base, D['te']['X'])
    cb = fit_catboost(F(D['tr']['X'][ok], oof[ok]), D['tr']['y'][ok] - oof[ok],
                      F(D['va']['X'], base_va), D['va']['y'] - base_va, seed)
    return np.clip(base_te + cb.predict(F(D['te']['X'], base_te)), 0, None)
