"""Veri yükleme, öznitelik üretimi, kronolojik bölme ve pencereleme (Bölüm 3 ve 4.1).

ÖNEMLİ: Burada hiçbir işlem gelecekteki değere bakmaz. Ölçekleme yalnız EĞİTİM bölümünden öğrenilir.
"""
import numpy as np, pandas as pd
from .config import STEP_MIN, MEASURE_MINUTES, SPLIT, PURGE

BASE_COLS = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv',
             'T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp']
TIME_COLS = ['saat', 'haftanin_gunu', 'ay', 'yilin_gunu',
             'saat_sin', 'saat_cos', 'ay_sin', 'ay_cos', 'wdir_sin', 'wdir_cos']
LAG_H = [1, 2, 3, 4, 6, 12, 24]                # saat cinsinden geçmiş gecikmeler
LAGS = [2 * h for h in LAG_H]                  # 30 dk adım sayısı
AR_COLS = (['Patv'] + [f'Patv_lag{h}h' for h in LAG_H] +
           ['Patv_rol1h', 'Patv_rol4h', 'Wspd_rol1h'])
FEATURES = BASE_COLS + ['is_missing'] + TIME_COLS + AR_COLS      # 37 öznitelik


def load_turbine(df_all, turb_id):
    """Tek türbini ayır, :10 ve :40 ölçüm anlarını al ve kesintisiz 30 dk ızgaraya oturt (eksikler NaN)."""
    d = df_all[df_all['TurbID'] == turb_id].copy()
    d['Tmstamp'] = pd.to_datetime(d['Tmstamp'])
    d = d.sort_values('Tmstamp').drop_duplicates('Tmstamp').set_index('Tmstamp')
    d = d[d.index.minute.isin(list(MEASURE_MINUTES))]
    full = pd.date_range(d.index.min(), d.index.max(), freq=f'{STEP_MIN}min')
    d = d.reindex(full); d.index.name = 'Tmstamp'
    return d


def add_features(d):
    """37 özniteliği üret. Gecikmeler ve kayan ortalamalar yalnız geçmişe bakar (shift)."""
    d = d.copy()
    era5 = ['T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp']
    # ERA5 eksikleri yalnız GEÇMİŞE bakan ileri doldurmayla (en çok 2 adım) doldurulur. (İlk sürümde doğrusal ara değerleme vardı,
    # t+1 gibi gelecek komşudan değer üretebilirdi; scripts/25 ile ölçüldü: hiçbir eğitim/doğrulama/test penceresinin girdisi değişmedi.)
    d[era5] = d[era5].ffill(limit=2)
    d['is_missing'] = d['Patv'].isna().astype(int)
    d['Patv'] = d['Patv'].clip(lower=0)
    idx = d.index
    d['saat'] = idx.hour; d['haftanin_gunu'] = idx.dayofweek
    d['ay'] = idx.month; d['yilin_gunu'] = idx.dayofyear
    d['saat_sin'] = np.sin(2 * np.pi * d['saat'] / 24); d['saat_cos'] = np.cos(2 * np.pi * d['saat'] / 24)
    d['ay_sin'] = np.sin(2 * np.pi * d['ay'] / 12);     d['ay_cos'] = np.cos(2 * np.pi * d['ay'] / 12)
    d['wdir_sin'] = np.sin(np.deg2rad(d['Wdir']));      d['wdir_cos'] = np.cos(np.deg2rad(d['Wdir']))
    for h, l in zip(LAG_H, LAGS):
        d[f'Patv_lag{h}h'] = d['Patv'].shift(l)
    d['Patv_rol1h'] = d['Patv'].shift(1).rolling(2).mean()
    d['Patv_rol4h'] = d['Patv'].shift(1).rolling(8).mean()
    d['Wspd_rol1h'] = d['Wspd'].shift(1).rolling(2).mean()
    return d


def chrono_split(d, frac=SPLIT, purge=0):
    """Kronolojik %70/%15/%15 bölme. Doğrulama ve test başlangıcında `purge` adımlık boşluk bırakılır
    (pencere ve hedef komşu bölüme taşmasın diye)."""
    n = len(d); i1 = int(n * frac[0]); i2 = int(n * (frac[0] + frac[1]))
    return d.iloc[:i1], d.iloc[i1 + purge:i2], d.iloc[i2 + purge:]


def make_windows(X, patv, seq_len, H):
    """Pencere: t-L+1..t satırları -> hedef Patv(t+H). Pencerede NaN veya hedef/geçerli güçte NaN varsa atılır.
    Döner: (pencereler, t indeksleri, hedef y, persistence yc = Patv(t))."""
    N, F = X.shape
    rowok = ~np.isnan(X).any(1)
    csum = np.concatenate([[0], np.cumsum(rowok)])
    t = np.arange(seq_len - 1, N - H)
    win_ok = (csum[t + 1] - csum[t + 1 - seq_len]) == seq_len
    tgt_ok = ~np.isnan(patv[t + H]) & ~np.isnan(patv[t])
    t = t[win_ok & tgt_ok]
    Xw = np.stack([X[ti - seq_len + 1:ti + 1] for ti in t]) if len(t) else np.empty((0, seq_len, F))
    return Xw.astype(np.float32), t, patv[t + H], patv[t]


class TurbineData:
    """Bir türbin için bölmeler + eğitim-ölçekleyici + pencere üreticisi (global değişken yok).

    Notebook'taki global d, tr, va, te, mn, mx, rng_, sc, x_dec, N karşılığıdır."""

    def __init__(self, df_all, turb_id, purge=PURGE):
        self.tid = turb_id
        self.d = add_features(load_turbine(df_all, turb_id))
        self.tr, self.va, self.te = chrono_split(self.d, purge=purge)
        self.mn = self.tr[FEATURES].min().values; mx = self.tr[FEATURES].max().values
        self.rng = np.where(mx - self.mn == 0, 1, mx - self.mn)           # YALNIZ eğitimden
        # Ayrıştırma girdisi: nedensel doldurma (ileri doldurma). Geriye doldurma YOK.
        self.x_dec = self.d['Patv'].ffill().fillna(0).values.astype(float)
        self.N = len(self.x_dec)

    def sc(self, x):
        return (x[FEATURES].values - self.mn) / self.rng

    def get_xy(self, L, H, skip=0):
        """Yalnız 37 öznitelikle pencereler (B1-B3). skip>0: serinin ilk `skip` satırı pencerelerden dışlanır
        (ayrıştırma kollarında rolling VMD penceresi nedeniyle dışlanan örneklerle eşit eğitim kümesi için; skip=W-1)."""
        return self.get_xy_ext(L, H, np.zeros((self.N, 0)), skip)

    def get_xy_ext(self, L, H, U, skip=0):
        """37 öznitelik + ayrıştırma bileşenleri U (N x K, eğitimde ölçeklenmiş) ile pencereler."""
        out = {}
        for ad, x in (('tr', self.tr), ('va', self.va), ('te', self.te)):
            idx = self.d.index.get_indexer(x.index)
            Xs = np.hstack([self.sc(x), U[idx]])
            if skip: Xs[idx < skip] = np.nan
            Xw, t, y, yc = make_windows(Xs, x['Patv'].values, L, H)
            out[ad] = dict(X=Xw, t=t, y=y, yc=yc)
        return out

    def scale_comp(self, U):
        """Bileşenleri min-max ölçekle; min/max YALNIZ eğitim satırlarından alınır."""
        pos = self.d.index.get_indexer(self.tr.index)
        lo, hi = np.nanmin(U[pos], 0), np.nanmax(U[pos], 0)
        return (U - lo) / np.where(hi - lo == 0, 1, hi - lo)
