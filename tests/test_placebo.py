"""leakage/placebo.py testleri (sentetik veri; TensorFlow gerekmez).
1) Sahte kuyruklar (tut, ayna, artim) t anındaki bileşeni YALNIZ geçmişe bağlar: x(t+1..) değişince bileşen değişmez.
2) 'gercek' kuyruk, x(t+1..) değişince bileşeni değiştirir (testin duyarlı olduğunu gösterir).
3) 'gercek' kuyruk, lookahead.py'deki orijinal LCSF ileri-bakış bileşenini birebir üretir."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from scipy.signal import butter
from leakage.decomposition import LCSF
from leakage.placebo import kuyruk, lcsf_placebo, _vmd_t, MODLAR


def _lcsf():
    lc = LCSF(); lc.bands = [butter(3, 0.02, 'low'), butter(3, [0.02, 0.08], 'band'),
                             butter(3, [0.08, 0.2], 'band'), butter(3, 0.2, 'high')]
    return lc


def _seri(n=1800, seed=0):
    r = np.random.default_rng(seed); x = np.zeros(n)
    for i in range(1, n): x[i] = 0.97 * x[i - 1] + r.normal(0, 1)
    return x


def test_kuyruk_nedensel():
    x = _seri(); y = x.copy(); t, n, ntr = 1500, 8, 1000
    y[t + 1:] += np.random.default_rng(1).normal(0, 50, len(y) - t - 1)       # geleceği boz
    for mod in ('tut', 'ayna', 'artim'):
        assert np.array_equal(kuyruk(x, t, n, mod, ntr, 0), kuyruk(y, t, n, mod, ntr, 0)), mod
    assert not np.array_equal(kuyruk(x, t, n, 'gercek', ntr), kuyruk(y, t, n, 'gercek', ntr))


def test_lcsf_placebo_nedensel_ve_duyarlilik():
    lc = _lcsf(); x = _seri(); y = x.copy(); t0 = 1500
    y[t0 + 1:] += np.random.default_rng(2).normal(0, 50, len(y) - t0 - 1)
    ts = slice(t0 - 5, t0 + 1)                                               # t <= t0 olan örnekler
    for m in (1, 2, 8):
        for mod in ('tut', 'ayna', 'artim'):
            a = lcsf_placebo(lc, x[:t0 + 20], m, mod, 1000, bas=t0 - 5)[ts]
            b = lcsf_placebo(lc, y[:t0 + 20], m, mod, 1000, bas=t0 - 5)[ts]
            assert np.allclose(a, b), (m, mod)
        a = lcsf_placebo(lc, x[:t0 + 20], m, 'gercek', 1000, bas=t0 - 5)[ts]
        b = lcsf_placebo(lc, y[:t0 + 20], m, 'gercek', 1000, bas=t0 - 5)[ts]
        assert not np.allclose(a, b), m


def test_lcsf_gercek_orijinalle_ayni():
    from scipy.signal import filtfilt
    lc = _lcsf(); x = _seri(1000); m, geri = 2, 720; N = len(x)
    ref = np.full((N, 4), np.nan)
    for t in range(900, N):
        a, b = max(0, t - geri + 1), min(N, t + m + 1)
        for k, (bb, aa) in enumerate(lc.bands): ref[t, k] = filtfilt(bb, aa, x[a:b])[t - a]
    got = lcsf_placebo(lc, x, m, 'gercek', 600, bas=900)
    assert np.allclose(ref[900:], got[900:])


def test_vmd_placebo_nedensel():
    x = _seri(900); y = x.copy(); t, w = 700, 120
    y[t + 1:] += np.random.default_rng(3).normal(0, 50, len(y) - t - 1)
    for m in (2, 8):
        for mod in ('tut', 'ayna', 'artim'):
            assert np.allclose(_vmd_t(x, t, m, w, mod, 500), _vmd_t(y, t, m, w, mod, 500)), (m, mod)
        assert not np.allclose(_vmd_t(x, t, m, w, 'gercek', 500), _vmd_t(y, t, m, w, 'gercek', 500)), m


def test_vmd_gercek_orijinalle_ayni():
    from leakage.decomposition import vmd_sorted
    x = _seri(900); w = 120
    for t, m in ((700, 2), (700, 8), (701, 1)):
        bas = t - w + 1; son = min(len(x), t + m + 1)                          # vmd_ileri._mod_t ile aynı hesap
        if (son - bas) % 2: bas += 1
        ref = vmd_sorted(x[bas:son])[:, t - bas]
        assert np.allclose(ref, _vmd_t(x, t, m, w, 'gercek', 500)), (t, m)
