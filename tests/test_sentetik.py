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


def test_lambda_ve_referans():
    from leakage.evaluation import lambda_orani
    from leakage.reference_models import nielsen_a, referans_hatalari
    lam, uyari = lambda_orani(115.94, 61.73, 11.05)
    assert abs(lam - 0.5168) < 1e-3 and uyari == ''                                           # makale tablosu (LCSF, T57)
    lam, _ = lambda_orani(116.78, 68.66, 64.64); assert abs(lam - 0.9228) < 1e-3
    l2, u2 = lambda_orani(100., 50., 60.); assert l2 > 1 and u2 != ''                          # oracle sızdırandan kötü -> aralık dışı uyarısı
    l3, u3 = lambda_orani(100., 50., 110.); assert np.isnan(l3) and u3 != ''                   # payda <= 0 -> tanımsız
    rng = np.random.default_rng(3); n = 20000; e = rng.normal(size=n); p = np.zeros(n)
    for t in range(1, n): p[t] = 0.9 * p[t - 1] + e[t]                                         # AR(1): a_k = 0.9^k
    assert abs(nielsen_a(p, 1) - 0.9) < 0.02 and abs(nielsen_a(p, 4) - 0.9 ** 4) < 0.03 and nielsen_a(p, 0) == 1.0
    k = 6; r = referans_hatalari(p[:15000], p[15000 + k:], p[15000:-k], k)
    assert r['nielsen'][1] <= r['persistence'][1] and r['nielsen'][1] <= r['iklimsel'][1] + 1e-9   # MSE-eniyi: ikisinden de kötü olamaz
    q = p.copy(); q[100] = np.nan; assert np.isfinite(nielsen_a(q, 2))                         # NaN çiftleri atlanır


def test_sertifika():
    from leakage.certificate import Operator, pertürbasyon_profili, erisim, onek_degismezligi
    td = TurbineData(df, 1); x = td.x_dec; N = td.N
    lc = LCSF().fit(x[:len(td.tr)]); w = 96
    nedensel = Operator('lcsf', at=lambda s, t: lc.transform(s, True)[t])
    sizdiran = Operator('lcsf-L', at=lambda s, t: lc.transform(s, False)[t])
    kayan = Operator('kayan', at=lambda s, t: last_mode(s[t - w + 1:t + 1]))
    from leakage.decomposition import full_vmd
    tam = Operator('tam', full=lambda s: full_vmd(s, len(s)))
    taular = [0, 1, 4, 16]; t0 = [N // 2, N // 2 + 500]
    S = pertürbasyon_profili(nedensel, x, t0, [100, 200], taular); assert (S == 0).all() and erisim(S[0], taular) == -1
    S = pertürbasyon_profili(sizdiran, x, t0, [200], taular); assert S[0, 0] > 1e-3 and erisim(S[0], taular) >= 0
    S = pertürbasyon_profili(kayan, x, [N // 2], [200], taular); assert (S == 0).all()
    S = pertürbasyon_profili(tam, x, [N // 2], [200], [0, 4]); assert S[0, 0] > 1e-3
    kes = [(N // 3) | 1, (N // 2) | 1, (2 * N // 3) | 1]                                      # tek t: önek uzunluğu çift (vmdpy tek uzunlukta son örneği keser)
    assert onek_degismezligi(nedensel, x, kes)[1] == 0 and onek_degismezligi(kayan, x, kes)[1] == 0
    assert onek_degismezligi(sizdiran, x, kes)[1] > 0                                          # çift yönlü: önek değişmez DEĞİL
    assert onek_degismezligi(tam, x, kes[:1])[1] == 1                                          # tam seri VMD: önek değişmez DEĞİL


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
