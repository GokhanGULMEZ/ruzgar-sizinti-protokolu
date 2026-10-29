"""ek05 - Rolling-origin kontrolu: seri farkli kesim noktalarinda kesilir, her kesimde
70/15/15 bolme yapilir, test donemi kayar. T57, 1 sa, LSTM. Sizdiran Full VMD her kesim icin O KESIMIN serisinde
yeniden hesaplanir (test sonrasi veri sizdirilmasin diye); rolling VMD onbellegi nedenseldir, ilk n satir ayni kalir.
Ortam: RT_PKG, RT_OUT (vmd_roll_T57.npy burada), RT_EK_OUT, RT_VERI_CSV. pip install vmdpy.
"""
import os, sys, time
sys.path.insert(0, os.environ.get('RT_PKG', '/content/paket'))
import numpy as np, pandas as pd
from leakage import config
from leakage.config import K, W, L_BEST, OUT, PURGE
from leakage.data import TurbineData, chrono_split, FEATURES
from leakage.decomposition import LCSF, full_vmd
from leakage.models import fit_lstm, pred_lstm
from leakage.evaluation import mae, lsi

EK = os.environ.get('RT_EK_OUT', os.path.join(OUT, 'ek_cikti')); os.makedirs(EK, exist_ok=True)
H = 2; SEEDS = (42, 43); FRACS = (0.70, 0.80, 0.90, 1.00)


class Cut(TurbineData):
    def __init__(self, df, tid, frac):
        super().__init__(df, tid)
        n = int(self.N * frac); self.d = self.d.iloc[:n]
        self.tr, self.va, self.te = chrono_split(self.d, purge=PURGE)
        self.mn = self.tr[FEATURES].min().values; mx = self.tr[FEATURES].max().values
        self.rng = np.where(mx - self.mn == 0, 1, mx - self.mn)
        self.x_dec = self.d['Patv'].ffill().fillna(0).values.astype(float); self.N = len(self.x_dec)


def main():
    df = pd.read_csv(config.VERI_CSV); U_roll = np.load(os.path.join(OUT, 'vmd_roll_T57.npy')); rows = []
    for fr in FRACS:
        td = Cut(df, 57, fr); n = td.N; t0 = time.time()
        lc = LCSF().fit(td.x_dec[:len(td.tr)])
        cC = lc.transform(td.x_dec, True); cL = lc.transform(td.x_dec, False)
        uA = full_vmd(td.x_dec, n)
        uB = U_roll[:n].copy()
        for a in (cC, cL, uA): a[:W - 1] = np.nan
        Ut = {'B6': td.scale_comp(uA), 'B7': td.scale_comp(uB), 'B8': td.scale_comp(cC), 'B8L': td.scale_comp(cL)}
        D0 = td.get_xy(L_BEST, H); per = mae(D0['te']['y'], D0['te']['yc'])
        for sd in SEEDS:
            m = {}
            for k, U in Ut.items():
                D = td.get_xy_ext(L_BEST, H, U); mod = fit_lstm(D, L_BEST, sd, None)
                m[k] = mae(D['te']['y'], pred_lstm(mod, D['te']['X']))
            r = dict(frac=fr, test_basi=str(td.te.index[0].date()), test_sonu=str(td.te.index[-1].date()), seed=sd,
                     MAE_pers=per, **m, LSI_VMD=lsi(m['B7'], m['B6']), LSI_LCSF=lsi(m['B8'], m['B8L']))
            print(r, f'| {(time.time()-t0)/60:.1f} dk'); rows.append(r)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(EK, 'ek05_rolling_origin.csv'), index=False)
    print(out.groupby('frac')[['MAE_pers', 'B7', 'B8', 'LSI_VMD', 'LSI_LCSF']].mean().round(2))

if __name__ == '__main__': main()
