"""Deney 3 - lambda'yi baska turbinlerde hesapla.  lambda = (E_nedensel - E_sizdiran) / (E_nedensel - E_oracle)
E_nedensel ve E_sizdiran: Drive'daki mevcut tahminlerden (adim6_T*.pkl seed 42, seed_T*.pkl seed 43,44), 3 tohum ortalamasi.
E_oracle: pencere hedefi icerir (m = H = 2): LCSF icin iki yonlu (sifir faz) suzme, pencere [t-719, t+2];
          VMD icin kayan VMD, pencere W=672 ve t+2'de biter, t anindaki mod degeri. Sonra LSTM, 3 tohum.
DOGRULAMA: T57 LCSF icin beklenen E_nedensel 115.94, E_sizdiran 61.73, E_oracle 11.05, lambda 0.52; VMD: 116.78 / 68.66 / 64.64 / 0.92.
Ortam: RT_PKG, RT_OUT (adim6/seed pkl), RT_EK_OUT, RT_VERI_CSV. Ayarlar: LAM_TURB="57,10,83,120,61,84" (LCSF; hizli),
       LAM_VMD_TURB="10" (VMD oracle; turbin basina ~30-60 dk, 2 cekirdekle paralel; yarida kalirsa devam eder).
Cikti: deney3_lambda.csv"""
import os, sys, time, pickle
sys.path.insert(0, os.environ.get('RT_PKG', '/content/paket'))
import numpy as np, pandas as pd
from multiprocessing import Pool
from leakage import config
from leakage.config import W, L_BEST, OUT
from leakage.data import TurbineData
from leakage.decomposition import LCSF, vmd_sorted
from leakage.models import fit_lstm, pred_lstm
from leakage.evaluation import mae, lsi

EK = os.environ.get('RT_EK_OUT', '/content/drive/MyDrive/RuzgarTahmin/ek_cikti/'); os.makedirs(EK, exist_ok=True)
H = 2; M = 2; SEEDS = (42, 43, 44); WIN = 720
LCSF_T = [int(x) for x in os.environ.get('LAM_TURB', '57,10,83,120,61,84').split(',') if x]
VMD_T = [int(x) for x in os.environ.get('LAM_VMD_TURB', '').split(',') if x]


def oracle_lcsf(lc, x, N):
    U = np.full((N, 4), np.nan)
    for t in range(W - 1, N):
        e = min(t + M + 1, N); s = max(0, e - WIN); z = lc.transform(x[s:e], False)
        U[t] = z[t - s]
    return U


def _vmd_chunk(args):
    x, lo, hi = args; N = len(x); out = np.full((hi - lo, 4), np.nan)
    for i, t in enumerate(range(lo, hi)):
        e = t + M + 1
        if e > N: continue
        u = vmd_sorted(x[e - W:e]); out[i] = u[:, -1 - M]
    return out


def oracle_vmd(x, N, fn):
    if os.path.exists(fn): return np.load(fn)
    ts = np.arange(W - 1, N); chunks = [(x, int(a), int(min(a + 500, N))) for a in range(W - 1, N, 500)]
    t0 = time.time()
    with Pool(2) as p: res = p.map(_vmd_chunk, chunks)
    U = np.full((N, 4), np.nan); U[W - 1:] = np.vstack(res); np.save(fn, U)
    print(f'  VMD oracle hazir ({(time.time()-t0)/60:.1f} dk)'); return U


def eval_oracle(td, U, tag):
    Ut = td.scale_comp(U); D = td.get_xy_ext(L_BEST, H, Ut); out = []
    for sd in SEEDS:
        m = fit_lstm(D, L_BEST, sd, None); p = pred_lstm(m, D['te']['X']); out.append(mae(D['te']['y'], p))
    print(f'  {tag}: E_oracle seed MAE =', np.round(out, 2)); return float(np.mean(out))


def prev(tid, key):
    b = pickle.load(open(os.path.join(OUT, f'adim6_T{tid}.pkl'), 'rb')); s = pickle.load(open(os.path.join(OUT, f'seed_T{tid}.pkl'), 'rb'))
    v = [mae(b[(key, H)]['y'], b[(key, H)]['p'])] + [mae(s[(key, sd)]['y'], s[(key, sd)]['p']) for sd in (43, 44)]
    return float(np.mean(v))


def main():
    df = pd.read_csv(config.VERI_CSV); rows = []
    for tid in sorted(set(LCSF_T) | set(VMD_T)):
        td = TurbineData(df, tid); x = td.x_dec.astype(float); N = td.N; r = dict(turbin=tid)
        if tid in LCSF_T:
            print(f'T{tid}: LCSF oracle'); lc = LCSF().fit(x[:len(td.tr)]); t0 = time.time()
            U = oracle_lcsf(lc, x, N); print(f'  suzme hazir ({time.time()-t0:.0f} sn)')
            Ec, El = prev(tid, 'B8'), prev(tid, 'B8L'); Eo = eval_oracle(td, U, 'LCSF')
            r.update(LCSF_Ec=Ec, LCSF_El=El, LCSF_Eo=Eo, LSI_LCSF=lsi(Ec, El), lam_LCSF=(Ec - El) / (Ec - Eo))
        if tid in VMD_T:
            print(f'T{tid}: VMD oracle (yavas)'); U = oracle_vmd(x, N, os.path.join(EK, f'vmd_oracle_m2_T{tid}.npy'))
            Ec, El = prev(tid, 'B7'), prev(tid, 'B6'); Eo = eval_oracle(td, U, 'VMD')
            r.update(VMD_Ec=Ec, VMD_El=El, VMD_Eo=Eo, LSI_VMD=lsi(Ec, El), lam_VMD=(Ec - El) / (Ec - Eo))
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(EK, 'deney3_lambda.csv'), index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

if __name__ == '__main__': main()
