"""Nedensellik sertifikası: gelecek-pertürbasyon duyarlılık profili ve önek değişmezliği testi.

Her ikisi de eğitim gerektirmez ve ayrıştırma yönteminden bağımsızdır; yalnız işleci ("operator") çağırır.
İşleç tanımı: Operator(ad, at=...) veya Operator(ad, full=...). at(x, t) -> t anındaki K bileşen vektörü (x dizisi
t'den sonrasını da içerebilir); full(x) -> (N, K) tüm seri çıktısı (tam seri VMD gibi işleçlerde hız için).

ÖNEMLİ: Sertifika, denenen kesim anları, gecikmeler ve büyüklükler için "bağımlılık saptanmadı" der.
Nedenselliğin kanıtı değil, gerekli bir koşuldur; sonuç bu parametrelerle birlikte raporlanmalıdır.

Profil:  D(tau; delta) = || Phi(x)_{t0-tau} - Phi(x + delta*gurultu 1{s>t0})_{t0-tau} ||_inf
  Nedensel işleçte tüm tau >= 0 için tam 0. 'Erişim' = D > tol olan en büyük tau (kirliliğin geriye yayıldığı mesafe)."""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Operator:
    ad: str
    at: Callable = None   # at(x, t) -> (K,) vektör
    full: Callable = None # isteğe bağlı: full(x) -> (N, K); verilirse bütün gecikmeler tek hesapta okunur

    def deger(self, x, t, onbellek=None):
        """t anındaki bileşen vektörü; full varsa onbellek (sözlük) ile tekrar hesap önlenir."""
        if self.full is not None:
            key = id(x)
            if onbellek is not None:
                if key not in onbellek: onbellek.clear(); onbellek[key] = self.full(x)
                return onbellek[key][t]
            return self.full(x)[t]
        return self.at(x, t)


def _bozulmus(x, t0, delta, rng):
    x2 = np.array(x, float, copy=True)
    x2[t0 + 1:] += rng.normal(0.0, delta, len(x) - t0 - 1)        # yalnız t0'dan SONRAKİ gözlemler
    return x2


def pertürbasyon_profili(op, x, t0_listesi, deltalar, taular, tohum=0):
    """Döner: dizi (len(deltalar), len(taular)); her hücre, t0 anları üzerinden en büyük D değeri."""
    rng = np.random.default_rng(tohum)
    S = np.zeros((len(deltalar), len(taular)))
    x = np.asarray(x, float); ob0 = {}
    for t0 in t0_listesi:
        for i, dl in enumerate(deltalar):
            x2 = _bozulmus(x, t0, dl, rng); ob2 = {}
            for j, tau in enumerate(taular):
                t = t0 - tau
                d = float(np.max(np.abs(np.asarray(op.deger(x, t, ob0)) - np.asarray(op.deger(x2, t, ob2)))))
                S[i, j] = max(S[i, j], d)
    return S


def erisim(S_satir, taular, tol=0.0):
    """Tek bir delta için erişim: D > tol olan en büyük tau (yoksa -1 = nedensel görünüm)."""
    idx = [t for t, s in zip(taular, S_satir) if s > tol]
    return max(idx) if idx else -1


def onek_degismezligi(op, x, kesim_noktalari, tol=0.0):
    """Phi(x[:t+1])_t == Phi(x)_t mi? Döner: (en büyük mutlak fark, tolerans üstü kesim sayısı, kesim sayısı)."""
    x = np.asarray(x, float); fark = []
    for t in kesim_noktalari:
        fark.append(float(np.max(np.abs(np.asarray(op.deger(x[:t + 1], t)) - np.asarray(op.deger(x, t))))))
    fark = np.array(fark)
    return float(fark.max()), int((fark > tol).sum()), len(fark)
