"""ADIM 24 — Diebold–Mariano (+HLN, Holm) testinin TOHUM BAZINDA raporlanması (hakem isteği: anlamlılık tohum seçimine bağlı mı?).
Girdi : adim6_T{tid}.pkl (tohum 42), seed_T{tid}.pkl (43, 44), seed_ek_T{tid}.pkl (45–49), OUT klasöründe. Model EĞİTİLMEZ (~1 dakika).
Kullanım: python 24_dm_tohum_bazinda.py            (6 türbin: 57 10 83 120 61 84)
          python 24_dm_tohum_bazinda.py 57 10      (seçili türbinler)
Karşılaştırmalar (H = 1 saat = 2 adım, mutlak kayıp; kare kayıp yan sütunda):
  VMD  : B6 (tam seri VMD, sızdıran) vs B7 (kayan VMD, nedensel)
  LCSF : B8L (çift yönlü, sızdıran) vs B8 (nedensel)
DM < 0 -> sızdıran model daha iyi. Holm ailesi: türbin başına 8 tohum x 2 karşılaştırma = 16 test.
Çıktı : dm_tohum_bazinda.csv (her tohum satırı) ve dm_tohum_ozet.csv (türbin x karşılaştırma özeti)."""
from _ortak import *
import sys, os, pickle, numpy as np, pandas as pd
from leakage.evaluation import dm_test, holm
from leakage import config

OUT = cikti_klasoru()
H = 2
EK = (45, 46, 47, 48, 49)
PAIRS = {'VMD': ('B6', 'B7'), 'LCSF': ('B8L', 'B8')}
tids = [int(a) for a in sys.argv[1:]] or [57, 10, 83, 120, 61, 84]

def yukle(ad):
    yol = os.path.join(OUT, ad)
    return pickle.load(open(yol, 'rb')) if os.path.exists(yol) else None

rows = []
for tid in tids:
    base, s3, s5 = yukle(f'adim6_T{tid}.pkl'), yukle(f'seed_T{tid}.pkl'), yukle(f'seed_ek_T{tid}.pkl')
    if base is None or s3 is None or s5 is None:
        print(f'T{tid}: eksik dosya (adim6 / seed / seed_ek), atlandı'); continue
    def al(k, sd):
        if sd == 42: return base[(k, H)]
        return s3[(k, sd)] if sd in config.SEEDS_EK else s5[(k, sd)]
    seeds = [42] + list(config.SEEDS_EK) + list(EK)
    aile = []
    for ad, (sz, ned) in PAIRS.items():
        for sd in seeds:
            a, b = al(sz, sd), al(ned, sd)
            assert len(a['y']) == len(b['y']) and np.allclose(a['y'], b['y']), 'test kümesi farklı'
            y = a['y']
            m = dm_test(y, a['p'], b['p'], h=H, loss='abs'); q = dm_test(y, a['p'], b['p'], h=H, loss='sq')
            e1, e2 = np.mean(np.abs(y - a['p'])), np.mean(np.abs(y - b['p']))
            aile.append(dict(turbin=tid, karsilastirma=ad, tohum=sd, MAE_sizdiran=e1, MAE_nedensel=e2,
                             LSI=100 * (e2 - e1) / e2, DM_abs=m['DM'], p_abs=m['p_raw'], DM_sq=q['DM'], p_sq=q['p_raw']))
    d = pd.DataFrame(aile)
    d['holm_abs'] = holm(d['p_abs'].values); d['holm_sq'] = holm(d['p_sq'].values)   # aile: bu türbinin 16 testi
    rows.append(d)

if not rows: sys.exit('Hiç türbin işlenemedi.')
R = pd.concat(rows, ignore_index=True)
R['anlamli_abs'] = (R.holm_abs < 0.05) & (R.DM_abs < 0); R['anlamli_sq'] = (R.holm_sq < 0.05) & (R.DM_sq < 0)
R.to_csv(OUT + 'dm_tohum_bazinda.csv', index=False)
oz = (R.groupby(['turbin', 'karsilastirma'])
        .agg(tohum_sayisi=('tohum', 'size'), yon_dogru=('DM_abs', lambda x: int((x < 0).sum())),
             anlamli_mutlak=('anlamli_abs', 'sum'), anlamli_kare=('anlamli_sq', 'sum'),
             maks_holm_p_mutlak=('holm_abs', 'max'), medyan_DM_mutlak=('DM_abs', 'median'),
             LSI_min=('LSI', 'min'), LSI_maks=('LSI', 'max')).reset_index())
oz.to_csv(OUT + 'dm_tohum_ozet.csv', index=False)
pd.set_option('display.width', 250)
print('\n== Tohum bazında DM özeti (her hücre 8 tohum üzerinden; anlamlı = Holm<0,05 ve DM<0)')
print(oz.round(4).to_string(index=False))
for ad in PAIRS:
    t = oz[oz.karsilastirma == ad]
    print(f'\n{ad}: toplam {int(t.tohum_sayisi.sum())} tohum-testinden mutlak kayıpta {int(t.anlamli_mutlak.sum())}, '
          f'kare kayıpta {int(t.anlamli_kare.sum())} anlamlı; yön (sızdıran daha iyi) {int(t.yon_dogru.sum())}/{int(t.tohum_sayisi.sum())}')
print('\nTohum bazında tüm satırlar:'); print(R[['turbin', 'karsilastirma', 'tohum', 'LSI', 'DM_abs', 'holm_abs', 'DM_sq', 'holm_sq']].round(4).to_string(index=False))
print('\nyazıldı:', OUT + 'dm_tohum_bazinda.csv ve dm_tohum_ozet.csv')
