"""ADIM 31 — Eşit eğitim kümesiyle ham LSTM kontrolü.
Soru: ayrıştırma kollarında rolling VMD penceresi (W=672) yüzünden serinin ilk W-1 satırı pencerelerden dışlanır;
ham LSTM ise bu satırları da eğitimde kullanır. 'Nedensel ayrıştırma ham girdiye üstün değil' bulgusu bu
~%4'lük eğitim kümesi farkından etkileniyor mu? Burada ham LSTM, ilk W-1 satır dışlanarak (skip=W-1) yeniden eğitilir;
test kümesi aynıdır. Tohumlar 42, 43, 44; altı türbin; 1 saat.
Önkoşul: 02_turbin_kosusu.py (adim6_T*.pkl) ve 03/18/24 (seed_T*.pkl).
Kullanım:  python 31_esit_egitim_ham.py            (tüm türbinler)   |   python 31_esit_egitim_ham.py 57
Süre   : türbin başına 3 eğitim; kaldığı yerden sürer. Çıktı: esit_ham_T{tid}.pkl ve ekranda özet."""
from _ortak import *
import sys, os, pickle, time
import numpy as np
from leakage.config import L_BEST, W
from leakage.data import TurbineData
from leakage.models import fit_arrays, pred_lstm
from leakage.evaluation import mae, dm_test

TURB = [int(a) for a in sys.argv[1:]] or [57, 10, 83, 120, 61, 84]
SEEDS = (42, 43, 44); H = 2; L = L_BEST
df = veri_yukle(); out = cikti_klasoru(); t0 = time.time()
satir = []
for tid in TURB:
    fn = os.path.join(out, f'esit_ham_T{tid}.pkl')
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    td = TurbineData(df, tid)
    D_un = td.get_xy(L, H); D = td.get_xy(L, H, skip=W - 1)
    assert np.array_equal(D['te']['t'], D_un['te']['t']), 'test kümesi farklı'
    for sd in SEEDS:
        if sd in res: continue
        m = fit_arrays(D['tr']['X'], D['tr']['y'], D['va']['X'], D['va']['y'], L, sd, None)
        res[sd] = dict(t=D['te']['t'], y=D['te']['y'], p=pred_lstm(m, D['te']['X']))
        pickle.dump(res, open(fn, 'wb'))
        print(f'   T{tid} seed={sd} eşit-küme ham MAE={mae(res[sd]["y"], res[sd]["p"]):.2f} | {(time.time()-t0)/60:.1f} dk', flush=True)
    base = pickle.load(open(os.path.join(out, f'adim6_T{tid}.pkl'), 'rb'))
    ek = pickle.load(open(os.path.join(out, f'seed_T{tid}.pkl'), 'rb'))
    def g(k, s): r = base[(k, H)] if s == 42 else ek[(k, s)]; return r
    M = {k: np.array([mae(g(k, s)['y'], g(k, s)['p']) for s in SEEDS]) for k in ('B2', 'B7', 'B8')}
    Me = np.array([mae(res[s]['y'], res[s]['p']) for s in SEEDS])
    n_un, n_eq = len(D_un['tr']['y']), len(D['tr']['y'])
    dm = np.mean([dm_test(res[s]['y'], res[s]['p'], g('B7', s)['p'], h=H)['DM'] for s in SEEDS])
    print(f'T{tid}: eğitim örneği {n_un} -> {n_eq} (-{100*(n_un-n_eq)/n_un:.1f}%) | MAE ham(tüm)={M["B2"].mean():.2f}  '
          f'ham(eşit)={Me.mean():.2f}  rolling VMD={M["B7"].mean():.2f}  LCSF ned.={M["B8"].mean():.2f}  | DM(ham-eşit vs rolling VMD, ort.)={dm:.2f}')
    satir.append((tid, M['B2'].mean(), Me.mean(), M['B7'].mean(), M['B8'].mean()))
a = np.array(satir)
print('\nTürbinler arası ortalama MAE (3 tohum):  ham(tüm) %.2f | ham(eşit küme) %.2f | rolling VMD %.2f | LCSF nedensel %.2f' % tuple(a[:, 1:].mean(0)))
