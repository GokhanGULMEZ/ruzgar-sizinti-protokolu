"""Hızlı doğruluk testleri (sentetik veri, GPU/gerçek veri gerekmez; ~2-4 dk).
Çalıştırma:  python tests/test_sentetik.py     (veya: pytest tests/)
Neyi kanıtlar: (1) bölmeler sızıntısız, (2) LCSF nedensel / çift yönlü sızdırıyor, (3) Rolling VMD nedensel,
(4) istatistik fonksiyonları doğru, (5) uçtan uca hat çalışıyor ve sızdıran model daha 'iyi' görünüyor."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sentetik_veri import uret
from leakage.data import TurbineData, FEATURES, make_windows
from leakage.decomposition import LCSF, last_mode, vmd_sorted
from leakage.evaluation import holm, dm_test, lsi, metrics

df = uret()


def test_veri_ve_bolme():
    td = TurbineData(df, 1)
    assert len(FEATURES) == 37
    assert (td.d.index.to_series().diff().dropna() == np.timedelta64(30, 'm')).all()      # kesintisiz 30 dk ızgara
    assert td.tr.index.max() < td.va.index.min() < td.va.index.max() < td.te.index.min()  # kronolojik
    assert (td.va.index.min() - td.tr.index.max()) >= np.timedelta64(30 * 12, 'm')        # purge = 12 adım
    D = td.get_xy(4, 2)
    for ad in ('tr', 'va', 'te'):                                                         # hedef = Patv(t+H)
        pv = {'tr': td.tr, 'va': td.va, 'te': td.te}[ad]['Patv'].values
        assert np.allclose(D[ad]['y'], pv[D[ad]['t'] + 2]) and np.allclose(D[ad]['yc'], pv[D[ad]['t']])
    assert td.mn.shape == (37,) and np.isclose(np.nanmin(td.sc(td.tr)), 0, atol=1e-9) and np.nanmax(td.sc(td.tr)) <= 1 + 1e-9           # eğitimden ölçek


def test_lcsf_nedensellik():
    td = TurbineData(df, 1); x, N = td.x_dec, td.N; tau = N // 2
    lc = LCSF().fit(x[:len(td.tr)])
    x2 = x.copy(); x2[tau + 1:] += np.random.default_rng(0).normal(0, 200, N - tau - 1)
    assert np.abs(lc.transform(x, True)[tau] - lc.transform(x2, True)[tau]).max() == 0.0      # nedensel: tam 0
    assert np.abs(lc.transform(x, False)[tau] - lc.transform(x2, False)[tau]).max() > 1e-3     # çift yönlü: sızıntı
    assert np.all(np.diff(lc.fc) > 0)                                                          # frekanslar sıralı


def test_rolling_vmd_nedensel():
    td = TurbineData(df, 1); x, tau, w = td.x_dec, 3000, 96
    x2 = x.copy(); x2[tau + 1:] += 300
    assert np.abs(last_mode(x[tau - w + 1:tau + 1]) - last_mode(x2[tau - w + 1:tau + 1])).max() == 0.0
    assert np.abs(vmd_sorted(x)[:, tau] - vmd_sorted(x2)[:, tau]).max() > 1e-3                 # tam seri: sızıntı


def test_istatistik():
    assert np.isclose(lsi(100, 60), 40.0)
    p = np.array([0.01, 0.04, 0.03, 0.005]); a = holm(p)
    # elle: sıralı 0.005,0.01,0.03,0.04 -> x4,x3,x2,x1 = 0.02,0.03,0.06,0.04 -> kümülatif maks: 0.02,0.03,0.06,0.06
    assert np.allclose(a, [0.03, 0.06, 0.06, 0.02])
    rng = np.random.default_rng(1); y = rng.normal(size=2000)
    r = dm_test(y, y + rng.normal(0, 0.5, 2000), y + rng.normal(0, 1.5, 2000), h=2)
    assert r['DM'] < -5 and r['p_raw'] < 1e-6                                                   # 1. model belirgin daha iyi
    r0 = dm_test(y, y + rng.normal(0, 1, 2000), y + rng.normal(0, 1, 2000), h=1)
    assert r0['p_raw'] > 0.01                                                                   # eşit modeller
    m = metrics(np.array([1., 2, 3]), np.array([1., 2, 3]), np.array([1., 1, 1]))
    assert m['MAE'] == 0 and m['Skill'] == 100


def test_uctan_uca():
    from leakage.pipeline import run_turbine, ozet
    with tempfile.TemporaryDirectory() as out:
        res = run_turbine(df, 1, Hs=(2,), out=out + '/', epochs=2, window=96)
        for k in ('B1', 'B2', 'B3', 'B6', 'B7', 'B8', 'B8L', 'B9'):
            assert (k, 2) in res and np.isfinite(res[(k, 2)]['p']).all()
        ts = [r['t'] for r in res.values()]; assert all(np.array_equal(ts[0], t) for t in ts)  # aynı test kümesi
        tab = ozet(res, 1, verbose=False)
        v = tab.set_index('model')['MAE']
        print('LSI VMD %.1f | LSI LCSF %.1f' % (v['LSI_VMD(%)'], v['LSI_LCSF(%)']))
        assert v['LSI_LCSF(%)'] > 0                                                              # sızdıran LCSF daha 'iyi' görünür


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('✓', n)
    print('TÜM TESTLER GEÇTİ')
