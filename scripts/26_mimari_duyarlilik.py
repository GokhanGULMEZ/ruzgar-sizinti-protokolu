"""ADIM 26 — Mimari duyarlılığı: aynı protokol GRU ve TCN (nedensel evrişim) ile (Türbin 57, 1 saat).
Soru: sızıntıyla ilişkili fark (LSI) LSTM'e özgü mü? Yalnız ağ türü değişir; veri, öznitelik, ayrıştırma
bileşenleri, bölmeler, eğitim ayarları (Adam 1e-3, MSE, erken durdurma) ve tohumlar (42,43,44) aynıdır.
Önkoşul: 02_turbin_kosusu.py 57 (ayrıştırma önbelleği ve adim6_T57.pkl).
Kullanım:  python 26_mimari_duyarlilik.py gru 57      |   python 26_mimari_duyarlilik.py tcn 57
Süre   : mimari başına 5 model x 3 tohum = 15 eğitim (GPU ~1 saat, CPU 2-3 saat). Her eğitim bitince kaydedilir;
         oturum kesilirse aynı komut kaldığı yerden sürer.
Çıktı  : mimari_{arch}_T{tid}.pkl ve ekranda özet (MAE, tohum başına LSI, DM)."""
from _ortak import *
import sys, os, pickle, time
import numpy as np
from leakage import config
from leakage.config import L_BEST, W
from leakage.data import TurbineData
from leakage.pipeline import build_components
from leakage.models import fit_arrays, pred_lstm
from leakage.evaluation import mae, lsi, dm_test

arch = (sys.argv[1] if len(sys.argv) > 1 else 'gru').lower()
tid = int(sys.argv[2]) if len(sys.argv) > 2 else 57
assert arch in ('gru', 'tcn')
SEEDS = (42, 43, 44); H = 2; L = L_BEST
df = veri_yukle(); out = cikti_klasoru()
fn = os.path.join(out, f'mimari_{arch}_T{tid}.pkl')
res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}

td = TurbineData(df, tid)
Ut, _ = build_components(td, out, W)
D_all = {'B2': td.get_xy(L, H)}
for k in ('B6', 'B7', 'B8', 'B8L'):
    D_all[k] = td.get_xy_ext(L, H, Ut[k])
t_ref = D_all['B2']['te']['t']
for k, D in D_all.items():
    assert np.array_equal(D['te']['t'], t_ref), 'test kümesi farklı'

t0 = time.time()
for k, D in D_all.items():
    for sd in SEEDS:
        if (k, sd) in res: continue
        m = fit_arrays(D['tr']['X'], D['tr']['y'], D['va']['X'], D['va']['y'], L, sd, None, arch)
        res[(k, sd)] = dict(t=D['te']['t'], y=D['te']['y'], p=pred_lstm(m, D['te']['X']))
        pickle.dump(res, open(fn, 'wb'))
        print(f'   {arch} T{tid} {k} seed={sd} MAE={mae(res[(k,sd)]["y"], res[(k,sd)]["p"]):.2f} | {(time.time()-t0)/60:.1f} dk', flush=True)

base = pickle.load(open(os.path.join(out, f'adim6_T{tid}.pkl'), 'rb'))
pers = mae(base[('B1', H)]['y'], base[('B1', H)]['p'])
lstm = {k: mae(base[(k, H)]['y'], base[(k, H)]['p']) for k in ('B2', 'B6', 'B7', 'B8', 'B8L')}
M = {k: np.array([mae(res[(k, s)]['y'], res[(k, s)]['p']) for s in SEEDS]) for k in D_all}
print(f'\n== {arch.upper()} | Türbin {tid} | 1 saat | tohumlar {SEEDS}')
print(f'  Persistence MAE = {pers:.2f}')
ad = {'B2': 'Ham', 'B7': 'Kayan VMD (ned.)', 'B6': 'Tam seri VMD (sızd.)', 'B8': 'LCSF ned.', 'B8L': 'LCSF iki yönlü (sızd.)'}
for k in ad:
    print(f'  {ad[k]:24s} MAE tohumlar {np.round(M[k],2)}  ort={M[k].mean():.2f} ± {M[k].std(ddof=1):.2f}  | LSTM(42)={lstm[k]:.2f}')
lv = 100 * (M['B7'] - M['B6']) / M['B7']; ll = 100 * (M['B8'] - M['B8L']) / M['B8']
print(f'  LSI-VMD  tohum başına {np.round(lv,1)}  ort={lv.mean():.1f} ± {lv.std(ddof=1):.1f}')
print(f'  LSI-LCSF tohum başına {np.round(ll,1)}  ort={ll.mean():.1f} ± {ll.std(ddof=1):.1f}')
print('  DM (HLN) sızdıran < nedensel, tohum başına (mutlak | kare kayıp):')
for s in SEEDS:
    r = res
    for a, b, nm in (('B6', 'B7', 'VMD'), ('B8L', 'B8', 'LCSF')):
        d1 = dm_test(r[(a, s)]['y'], r[(a, s)]['p'], r[(b, s)]['p'], h=H)
        d2 = dm_test(r[(a, s)]['y'], r[(a, s)]['p'], r[(b, s)]['p'], h=H, loss='sq')
        print(f'    tohum {s} {nm:5s}: DM={d1["DM"]:7.2f} (p={d1["p_raw"]:.1e}) | kare DM={d2["DM"]:7.2f} (p={d2["p_raw"]:.1e})')
print(f'  Nedensel modeller persistence\'a göre (MAE skill %): ham {100*(pers-M["B2"].mean())/pers:.1f}, '
      f'kayan VMD {100*(pers-M["B7"].mean())/pers:.1f}, LCSF {100*(pers-M["B8"].mean())/pers:.1f}')
