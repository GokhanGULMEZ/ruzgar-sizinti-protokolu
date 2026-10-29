"""EK 02 - Turbin = analiz birimi ve turbinler arasi bagimlilik (1 saat, 8 tohum, 6 turbin). Egitim YOK (~1 dk).
Olasi itiraz: '48 turbin-tohum cifti bagimsiz birim degildir' (ayni ciftlik, ayni hava, ayni test donemi; tohumlar ayni veriyi paylasir).
Cozum  : (a) Tohumlari TURBIN ICINDE ortala -> birim = turbin (n=6): LSI ortalamasi, t araligi (df=5), isaret sayisi.
         (b) ORTAK-ZAMAN blok bootstrap: ayni zaman bloklari TUM turbinler icin birlikte yeniden orneklenir (turbinler arasi bagimlilik korunur);
             istatistik = turbinler uzerinden ortalama LSI. Hesap, her turbinin 8 tohumluk ortalama mutlak hatasi uzerinden yapilir.
Girdi  : adim6_T*.pkl (tohum 42), seed_T*.pkl (43,44), seed_ek_T*.pkl (45-49). Cikti: RT_EK_OUT/ek02_birim_bagimlilik.csv
Not    : (b) yalniz test ornekleme belirsizligini yansitir; tohum (egitim) degiskenligi (a)'da turbin icinde ortalanarak ele alinir."""
import numpy as np, pandas as pd
from scipy import stats
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ortak'))
from ek_ortak import *

TIDS = (57, 10, 83, 120, 61, 84); H = 2; EK = (45, 46, 47, 48, 49); SEEDS_EK = (43, 44)
PAIRS = {'VMD': ('B6', 'B7'), 'LCSF': ('B8L', 'B8')}; BLOK, B = 96, 2000

veri = {}                                   # tid -> {'t': t, 'y': y, 'e': {model: tohum-ort. mutlak hata}}
for tid in TIDS:
    base, s3, s5 = yukle(f'adim6_T{tid}.pkl'), yukle(f'seed_T{tid}.pkl'), yukle(f'seed_ek_T{tid}.pkl')
    if base is None or s3 is None or s5 is None: print(f'T{tid}: eksik dosya, atlandi'); continue
    e, t0, y0 = {}, base[('B6', H)]['t'], base[('B6', H)]['y']
    for k in ('B6', 'B7', 'B8', 'B8L'):
        liste = [base[(k, H)]] + [s3[(k, s)] for s in SEEDS_EK] + [s5[(k, s)] for s in EK]
        for d in liste: assert len(d['y']) == len(y0) and np.allclose(d['y'], y0), 'test kumesi farkli'
        e[k] = np.mean([np.abs(d['y'] - d['p']) for d in liste], axis=0)
    veri[tid] = dict(t=t0, e=e)
tids = list(veri)
if len(tids) < 2: raise SystemExit('En az 2 turbin icin dosya gerekli.')

rows = []
for ad, (sz, ned) in PAIRS.items():
    L = np.array([lsi(veri[t]['e'][ned].mean(), veri[t]['e'][sz].mean()) for t in tids])
    n = len(L); ci = stats.t.interval(0.95, n - 1, loc=L.mean(), scale=L.std(ddof=1) / np.sqrt(n))
    # (b) ortak-zaman blok bootstrap
    ortak = veri[tids[0]]['t']
    for t in tids[1:]: ortak = np.intersect1d(ortak, veri[t]['t'])
    E = {t: {k: veri[t]['e'][k][np.isin(veri[t]['t'], ortak)] for k in (sz, ned)} for t in tids}
    for t in tids:                          # sira: t artan; isin maskesi sirayi korur
        assert np.array_equal(veri[t]['t'][np.isin(veri[t]['t'], ortak)], ortak)
    rng = np.random.default_rng(0); idx = blok_indeks(len(ortak), BLOK, B, rng)
    bs = np.mean([lsi(E[t][ned][idx].mean(1), E[t][sz][idx].mean(1)) for t in tids], axis=0)
    rows.append(dict(karsilastirma=ad, turbin_sayisi=n, ortak_test_noktasi=len(ortak), LSI_turbin_ort=L.mean(), LSI_turbin_sd=L.std(ddof=1),
                     t_alt=ci[0], t_ust=ci[1], turbinde_pozitif=int((L > 0).sum()),
                     ortak_blok_alt=np.percentile(bs, 2.5), ortak_blok_ust=np.percentile(bs, 97.5),
                     LSI_min=L.min(), LSI_maks=L.max()))
    print(f'\n{ad}: turbin basina LSI (tohum-ort.) = {np.round(L, 1)}')
    print(f'  turbin birim, n={n}: ort={L.mean():.1f} sd={L.std(ddof=1):.1f}  t %95 [{ci[0]:.1f}, {ci[1]:.1f}]  pozitif {int((L > 0).sum())}/{n}')
    print(f'  ortak-zaman blok bootstrap %95 [{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}]  (ortak test noktasi {len(ortak)})')
pd.DataFrame(rows).to_csv(EK_OUT + 'ek02_birim_bagimlilik.csv', index=False)
print('\nyazildi:', EK_OUT + 'ek02_birim_bagimlilik.csv')
