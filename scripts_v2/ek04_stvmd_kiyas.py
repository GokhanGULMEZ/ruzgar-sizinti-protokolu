"""ek04 - Hizli nedensel VMD karsilastirmasi (STVMD/RSVMD benzeri hizli nedensel baseline).
Fikir: kayan pencere VMD'sini onceki pencerenin merkez frekanslariyla SICAK BASLATIP az iterasyonla calistirmak
(STVMD / recursive-sliding VMD'nin temel hilesi). Tam bir STVMD uygulamasi DEGILDIR; hesap maliyetini azaltan
en yakin nedensel alternatiftir. Ayni pencere W=672, ayni K=4, ayni alpha, nedensel (yalniz gecmis).
Olcer: (a) adim basina sure, (b) bilesenlerin rolling VMD'ye yakinligi, (c) T57 1 sa LSTM MAE (3 seed).
Ortam: RT_PKG (leakage paketi yolu), RT_OUT (Drive sonuc klasoru, vmd_roll_T57.npy ve adim6/seed pkl burada),
RT_EK_OUT (cikti), RT_VERI_CSV. Gerekli: pip install vmdpy
"""
import os, sys, time, pickle
sys.path.insert(0, os.environ.get('RT_PKG', '/content/paket'))
import numpy as np, pandas as pd
from leakage import config
from leakage.config import ALPHA, K, W, L_BEST, OUT
from leakage.data import TurbineData
from leakage.decomposition import vmd_sorted, last_mode, LCSF
from leakage.models import fit_lstm, pred_lstm
from leakage.evaluation import mae, dm_test, lsi

EK = os.environ.get('RT_EK_OUT', os.path.join(OUT, 'ek_cikti')); os.makedirs(EK, exist_ok=True)
H = 2; SEEDS = (42, 43, 44); NITERS = (3, 5, 10, 25); PRIMARY = 10


def vmd_core(f, om0, n_iter, alpha=ALPHA, K=K, tol=1e-7):
    """vmdpy ile ayni cebir (tau=0, DC=0); baslangic frekansi om0 ve iterasyon siniri disaridan verilir.
    Doner: (u: K x len(f) zaman uzayi modlari, om: K son frekanslar)."""
    n0 = len(f); T0 = n0
    fm = np.zeros(2 * T0)
    fm[:T0 // 2] = f[T0 // 2 - 1::-1]; fm[T0 // 2:3 * T0 // 2] = f; fm[3 * T0 // 2:] = f[-1:-T0 // 2 - 1:-1]
    T = len(fm); t = np.arange(1, T + 1) / T; freqs = t - 0.5 - 1 / T
    fh = np.fft.fftshift(np.fft.fft(fm)); fh[:T // 2] = 0
    u = np.zeros((K, T), complex); om = np.array(om0, float).copy()
    half = slice(T // 2, T)
    for n in range(n_iter):
        uold = u.copy(); s = u.sum(0)
        for k in range(K):
            s -= u[k]
            u[k] = (fh - s) / (1 + alpha * (freqs - om[k]) ** 2)
            p = np.abs(u[k, half]) ** 2
            om[k] = (freqs[half] * p).sum() / p.sum()
            s += u[k]
        d = (np.abs(u - uold) ** 2).sum() / T
        if d <= tol: break
    uh = np.zeros((K, T), complex)
    uh[:, half] = u[:, half]
    uh[:, T // 2:0:-1] = np.conj(u[:, half])
    uh[:, 0] = np.conj(uh[:, -1])
    out = np.real(np.fft.ifft(np.fft.ifftshift(uh, axes=1), axis=1))
    return out[:, T // 4:3 * T // 4], om


def main():
    df = pd.read_csv(config.VERI_CSV); td = TurbineData(df, 57); x = td.x_dec; N = td.N
    # --- dogrulama: 500 iterasyonla vmdpy ile ayni mi?
    w = x[5000 - W + 1:5001]
    ref = vmd_sorted(w)[:, -1]
    u, om = vmd_core(w, (0.5 / K) * np.arange(K), 500); o = np.argsort(om); mine = u[o][:, -1]
    print('dogrulama (500 it, vmdpy ile) maks fark kW:', np.abs(ref - mine).max())
    # --- cache: rolling VMD (tam) ve referans sure
    U_roll = np.load(os.path.join(OUT, 'vmd_roll_T57.npy'))
    pos = np.linspace(W + 10, N - 2, 200).astype(int)
    t0 = time.time()
    for p_ in pos: last_mode(x[p_ - W + 1:p_ + 1])
    t_ref = (time.time() - t0) / len(pos) * 1e3
    print(f'rolling VMD (vmdpy, 500 it siniri): {t_ref:.1f} ms/adim (200 konum)')
    # ilk pencere: tam yakinsama, sonra sicak baslat
    u0, om0 = vmd_core(x[:W], (0.5 / K) * np.arange(K), 500); om_prev = np.sort(om0)
    rows = []; comps = {}
    for n_it in NITERS:
        om_prev = np.sort(om0.copy()); U = np.full((N, K), np.nan); t0 = time.time()
        for t in range(W - 1, N):
            u, om = vmd_core(x[t - W + 1:t + 1], om_prev, n_it)
            o = np.argsort(om); U[t] = u[o][:, -1]; om_prev = om[o]
        dt = (time.time() - t0) / (N - W + 1) * 1e3
        diff = np.sqrt(np.nanmean((U - U_roll) ** 2, 0))
        print(f'n_iter={n_it}: {dt:.2f} ms/adim | bilesen RMSE (rolling VMD ile) kW = {np.round(diff, 2)}')
        rows.append(dict(n_iter=n_it, ms_adim=dt, oran_rolling=t_ref / dt, **{f'rmse_k{k+1}': diff[k] for k in range(K)}))
        comps[n_it] = U
    pd.DataFrame(rows).to_csv(os.path.join(EK, 'ek04_zaman_bilesen.csv'), index=False)
    # --- tahmin dogrulugu (birincil n_iter)
    U = comps[PRIMARY]; Us = td.scale_comp(U); D = td.get_xy_ext(L_BEST, H, Us)
    base = pickle.load(open(os.path.join(OUT, 'adim6_T57.pkl'), 'rb')); seedp = pickle.load(open(os.path.join(OUT, 'seed_T57.pkl'), 'rb'))
    res = []
    for sd in SEEDS:
        m = fit_lstm(D, L_BEST, sd, None); p = pred_lstm(m, D['te']['X']); y = D['te']['y']
        if sd == 42: b7, b8 = base[('B7', H)], base[('B8', H)]
        else: b7, b8 = seedp[('B7', sd)], seedp[('B8', sd)]
        assert np.array_equal(D['te']['t'], b7['t']) if 't' in b7 else True
        r = dict(seed=sd, MAE_warm=mae(y, p), MAE_rolling=mae(b7['y'], b7['p']), MAE_LCSF=mae(b8['y'], b8['p']),
                 DM_warm_vs_rolling=dm_test(y, p, b7['p'], h=H)['DM'])
        print(r); res.append(r)
    r = pd.DataFrame(res); r.to_csv(os.path.join(EK, 'ek04_tahmin.csv'), index=False)
    print(r.mean(numeric_only=True))

if __name__ == '__main__': main()
