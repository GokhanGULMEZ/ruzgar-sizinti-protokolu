"""Ayrıştırma yöntemleri (makale Bölüm 4.2-4.4).

  Full VMD     SIZDIRAN : tüm seriye (test dahil) tek VMD uygulanır.
  Rolling VMD  NEDENSEL : her t anı için yalnız [t-W+1, t] penceresine VMD; son değer alınır.
  LCSF         NEDENSEL : bantları eğitimden öğrenilmiş Butterworth filtre bankası, tek yönlü (lfilter).
  LCSF-L       SIZDIRAN : aynı filtreler çift yönlü (filtfilt) -> geleceğe bakar (karşı örnek).
"""
import os, time
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt, sosfilt, sosfilt_zi, sosfiltfilt
from vmdpy import VMD
from joblib import Parallel, delayed
from .config import ALPHA, TAU, K, DC, INIT, TOL, W, BUTTER_ORDER, OUT


def blok_temizle(res):
    """Blok sonuçlarını diziye çevirir; NaN varsa sıfırlar ve uyarı verir (sessiz yutmaz)."""
    a = np.array(res)
    n = int(np.isnan(a).sum())
    if n:
        import warnings
        warnings.warn(f'VMD bloğunda {n} NaN değer 0 ile değiştirildi.')
    return np.nan_to_num(a)


def vmd_sorted(sig):
    """VMD; modlar merkez frekansına göre (düşükten yükseğe) sıralı döner: şekil (K, len)."""
    u, _, om = VMD(sig, ALPHA, TAU, K, DC, INIT, TOL)
    return u[np.argsort(om[-1])]


def last_mode(sig):
    """Penceredeki VMD'nin SON zaman adımındaki K bileşen değeri (Rolling VMD'nin çıktısı)."""
    return vmd_sorted(sig)[:, -1]


def full_vmd(x_dec, N):
    """SIZDIRAN: tüm seriye tek VMD. Döner (N, K). (vmdpy çift uzunlukta keser; sonuncu satır tekrarlanır.)"""
    uA = vmd_sorted(x_dec).T
    if uA.shape[0] < N:
        uA = np.vstack([uA, np.repeat(uA[-1:], N - uA.shape[0], 0)])
    return uA


def rolling_vmd(x_dec, tag, out_dir=OUT, window=W, block=2000, n_jobs=-1):
    """NEDENSEL: her t >= window-1 için VMD(x[t-window+1:t+1]) son değeri.
    Maliyetli (~1 saat/türbin, çok çekirdek). Bloklar diske kaydedilir; yarıda kalırsa kaldığı yerden sürer.
    tag: dosya adı eki (ör. 'T57')."""
    N = len(x_dec)
    fn_all = os.path.join(out_dir, f'vmd_roll_{tag}.npy')
    if os.path.exists(fn_all):
        return np.load(fn_all)
    U = np.full((N, K), np.nan)
    for s in range(window - 1, N, block):
        e = min(s + block, N); fn = os.path.join(out_dir, f'rollvmd_{tag}_blk_{s}.npy')
        if os.path.exists(fn):
            U[s:e] = np.load(fn); continue
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(last_mode)(x_dec[t - window + 1:t + 1]) for t in range(s, e))
        blk = blok_temizle(res); np.save(fn, blk); U[s:e] = blk
        print(f'   rolling VMD {tag}: {e}/{N} | {time.time()-t0:.0f} sn')
    np.save(fn_all, U)
    return U


class LCSF:
    """Learned Causal Spectral Filterbank.

    fit(x_train): YALNIZ eğitim kesitinde VMD çalıştırıp K merkez frekansı öğrenir; ardışık merkez
    frekansların geometrik ortalaması bant sınırı olur. Sonra frekanslar sabittir (test verisi öğrenmeye karışmaz).
    transform(x, causal=True): tek yönlü (nedensel) filtre; causal=False: çift yönlü (SIZDIRAN karşı örnek)."""

    def fit(self, x_train):
        _, _, om = VMD(x_train, ALPHA, TAU, K, DC, INIT, TOL)
        self.fc = np.clip(np.sort(om[-1]), 1e-4, 0.49)          # döngü/adım
        edges = np.sqrt(self.fc[:-1] * self.fc[1:])               # K-1 sınır (geometrik orta)
        self.edges = edges
        self.bands = []; self.sos = []
        for k in range(K):
            if k == 0:       ba = butter(BUTTER_ORDER, edges[0], 'low', fs=1.0)
            elif k == K - 1: ba = butter(BUTTER_ORDER, edges[K - 2], 'high', fs=1.0)
            else:            ba = butter(BUTTER_ORDER, [edges[k - 1], edges[k]], 'band', fs=1.0)
            self.bands.append(ba)
            self.sos.append(butter(BUTTER_ORDER, edges[0] if k == 0 else edges[K - 2] if k == K - 1 else [edges[k - 1], edges[k]],
                                   'low' if k == 0 else 'high' if k == K - 1 else 'band', fs=1.0, output='sos'))
        return self

    def transform(self, x, causal=True, sos=False):
        """sos=False: (b,a) biçimi (makaledeki sonuçların üretildiği varsayılan). sos=True: ikinci dereceden kesitler
        (sayısal olarak daha kararlı; scripts/30 iki biçimin çıktısını karşılaştırır)."""
        out = np.zeros((len(x), K))
        if sos:
            for k, s_ in enumerate(self.sos):
                if causal:
                    out[:, k] = sosfilt(s_, x, zi=sosfilt_zi(s_) * x[0])[0]
                else:
                    out[:, k] = sosfiltfilt(s_, x)
            return out
        for k, (b, a) in enumerate(self.bands):
            if causal:
                zi = lfilter_zi(b, a) * x[0]                      # başlangıç durumu yalnız ilk örnekten
                out[:, k] = lfilter(b, a, x, zi=zi)[0]
            else:
                out[:, k] = filtfilt(b, a, x)                     # ileri+geri -> gelecek örnekleri kullanır
        return out
