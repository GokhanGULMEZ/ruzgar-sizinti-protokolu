"""EK 08 - LCSF ile rolling VMD (nedensel) arasinda dogruluk farki: guven araligi ve esdegerlik (TOST) kontrolu.
Neden : 'anlamli fark yok' esdegerlik demek degildir. Bu betik fark icin guven araligi verir; okuyucu kendi marjini uygulayabilir.
Olcu   : d = 100 * (MAE_LCSF - MAE_rollingVMD) / MAE_rollingVMD  (%, 1 saat). d < 0 ise LCSF daha iyi.
         MAE'ler her turbinde 8 tohum uzerinden ortalanmis mutlak hatalardan hesaplanir (ek02 ile ayni girdi).
Cikti  : (a) turbin basina d; (b) birim = turbin (n=6) t araligi (%90 ve %95); (c) ortak-zaman blok bootstrap (%90 ve %95).
         TOST: %90 araligi (-MARJ, +MARJ) icindeyse esdegerlik (alfa=0.05). Varsayilan marj %2 (MARJ ortam degiskeni ile degisir).
Girdi  : adim6_T*.pkl, seed_T*.pkl, seed_ek_T*.pkl (v1 betikleri 02, 03, 18). Egitim YOK (~1 dk). Cikti: RT_EK_OUT/ek08_esdegerlik.csv"""
import numpy as np, pandas as pd
from scipy import stats
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ortak'))
from ek_ortak import *

TIDS = (57, 10, 83, 120, 61, 84); H = 2; EK = (45, 46, 47, 48, 49); SEEDS_EK = (43, 44)
BLOK, B = 96, 2000; MARJ = float(os.environ.get('EK08_MARJ', '2.0'))

veri = {}
for tid in TIDS:
    base, s3, s5 = yukle(f'adim6_T{tid}.pkl'), yukle(f'seed_T{tid}.pkl'), yukle(f'seed_ek_T{tid}.pkl')
    if base is None or s3 is None or s5 is None: print(f'T{tid}: eksik dosya, atlandi'); continue
    y0, t0, e = base[('B7', H)]['y'], base[('B7', H)]['t'], {}
    for k in ('B7', 'B8'):
        liste = [base[(k, H)]] + [s3[(k, s)] for s in SEEDS_EK] + [s5[(k, s)] for s in EK]
        for d in liste: assert len(d['y']) == len(y0) and np.allclose(d['y'], y0), 'test kumesi farkli'
        e[k] = np.mean([np.abs(d['y'] - d['p']) for d in liste], axis=0)
    veri[tid] = dict(t=t0, e=e)
tids = list(veri)
if len(tids) < 2: raise SystemExit('En az 2 turbin icin dosya gerekli.')

d_pct = lambda a, b: 100.0 * (a.mean(-1) - b.mean(-1)) / b.mean(-1)      # a = LCSF hatalari, b = rolling VMD hatalari
D = np.array([d_pct(veri[t]['e']['B8'], veri[t]['e']['B7']) for t in tids]); n = len(D)
ortak = veri[tids[0]]['t']
for t in tids[1:]: ortak = np.intersect1d(ortak, veri[t]['t'])
E = {t: {k: veri[t]['e'][k][np.isin(veri[t]['t'], ortak)] for k in ('B7', 'B8')} for t in tids}
rng = np.random.default_rng(0); idx = blok_indeks(len(ortak), BLOK, B, rng)
bs = np.mean([100.0 * (E[t]['B8'][idx].mean(1) - E[t]['B7'][idx].mean(1)) / E[t]['B7'][idx].mean(1) for t in tids], axis=0)

rows = []
for t, d in zip(tids, D): rows.append(dict(satir=f'T{t}', d_yuzde=d))
for g in (0.90, 0.95):
    a, u = stats.t.interval(g, n - 1, loc=D.mean(), scale=D.std(ddof=1) / np.sqrt(n))
    p = (100 * (1 - g) / 2, 100 - 100 * (1 - g) / 2)
    ba, bu = np.percentile(bs, p[0]), np.percentile(bs, p[1])
    rows.append(dict(satir=f'ortalama (turbin birim, n={n}) %{int(g*100)}', d_yuzde=D.mean(), alt=a, ust=u, marj=MARJ, esdeger=bool(a > -MARJ and u < MARJ)))
    rows.append(dict(satir=f'ortak-zaman blok bootstrap %{int(g*100)} (nokta={len(ortak)})', d_yuzde=bs.mean(), alt=ba, ust=bu, marj=MARJ, esdeger=bool(ba > -MARJ and bu < MARJ)))
out = pd.DataFrame(rows); out.to_csv(EK_OUT + 'ek08_esdegerlik.csv', index=False)
print(out.to_string()); print(f'\nd<0: LCSF daha iyi. TOST: %90 araligi (-{MARJ:.1f}, +{MARJ:.1f}) icindeyse esdeger.'); print('yazildi:', EK_OUT + 'ek08_esdegerlik.csv')
