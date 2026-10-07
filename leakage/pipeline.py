"""Türbin koşuları ve seed tekrarı (makale Bölüm 6). Notebook'taki run_turbine / ozet / seed_tekrar karşılığı."""
import os, time, pickle
import numpy as np, pandas as pd
from .config import OUT, L_BEST, W, SEED, SEEDS_EK
from .data import TurbineData
from .decomposition import LCSF, full_vmd, rolling_vmd
from .models import fit_lstm, pred_lstm, catboost_direct, stack
from .evaluation import lsi, dm_test, mae

AD = {'B1': 'Persistence', 'B2': 'Raw LSTM', 'B3': 'CatBoost Direct',
      'B6': 'Full VMD-LSTM (sızdıran)', 'B7': 'Rolling VMD-LSTM (nedensel)',
      'B8L': 'LCSF çift yönlü (sızdıran)', 'B8': 'LCSF nedensel', 'B9': 'B9 LCSF+OOF CatBoost'}


def put(res, key, H, D, p):
    res[(key, H)] = dict(t=D['te']['t'], y=D['te']['y'], yc=D['te']['yc'], p=p)


def build_components(td, out=OUT, window=W, rolling=None):
    """Bir türbin için ayrıştırma bileşenlerini (eğitimde ölçeklenmiş) üret ve önbellekle.
    Döner: (Ut sözlüğü {'B6','B7','B8','B8L'}, LCSF nesnesi).
    Önbellek: vmd_full_T{tid}.npy, vmd_roll_T{tid}.npy (+ bloklar)."""
    tag = f'T{td.tid}'
    lcsf = LCSF().fit(td.x_dec[:len(td.tr)])                      # bantlar YALNIZ eğitimden
    fa = os.path.join(out, f'vmd_full_{tag}.npy')
    if os.path.exists(fa):
        uA = np.load(fa)
    else:
        uA = full_vmd(td.x_dec, td.N); np.save(fa, uA)
    uB = rolling(td.x_dec, tag) if rolling else rolling_vmd(td.x_dec, tag, out_dir=out, window=window)
    uA_m = uA.copy(); uA_m[:window - 1] = np.nan                  # B7 ile aynı örnek kümesi
    cC = lcsf.transform(td.x_dec, True); cL = lcsf.transform(td.x_dec, False)
    cC[:window - 1] = np.nan; cL[:window - 1] = np.nan
    Ut = {'B6': td.scale_comp(uA_m), 'B7': td.scale_comp(uB),
          'B8': td.scale_comp(cC), 'B8L': td.scale_comp(cL)}
    return Ut, lcsf


def run_turbine(df_all, tid, Hs=(2,), out=OUT, L=L_BEST, seed=SEED, epochs=None, window=W):
    """Bir türbin için tüm modeller (B1,B2,B3,B6,B7,B8,B8L,B9) — her ufuk için. Sonuç adim6_T{tid}.pkl'e yazılır."""
    t_start = time.time()
    td = TurbineData(df_all, tid)
    Ut, lcsf = build_components(td, out, window)
    print(f'T{tid}: N={td.N:,} | eğitim/doğrulama/test = {len(td.tr):,}/{len(td.va):,}/{len(td.te):,}'
          f' | frekanslar={lcsf.fc.round(4)}')
    res = {}
    for H in Hs:
        D0 = td.get_xy(L, H); t0_ = D0['te']
        put(res, 'B1', H, D0, t0_['yc'])                                         # persistence
        put(res, 'B2', H, D0, pred_lstm(fit_lstm(D0, L, seed, epochs), t0_['X']))
        put(res, 'B3', H, D0, catboost_direct(D0, seed))
        for key in ('B6', 'B7', 'B8', 'B8L'):
            D = td.get_xy_ext(L, H, Ut[key])
            assert np.array_equal(D['te']['t'], t0_['t']), 'test kümesi farklı'
            m = fit_lstm(D, L, seed, epochs); put(res, key, H, D, pred_lstm(m, D['te']['X']))
            if key == 'B8': D8, m8 = D, m
        put(res, 'B9', H, D8, stack(D8, m8, L, H, 'oof', seed, epochs))
        print(f'   T{tid} H={H/2} sa modeller tamam | {(time.time()-t_start)/60:.1f} dk')
    pickle.dump(res, open(os.path.join(out, f'adim6_T{tid}.pkl'), 'wb'))
    return res


def ozet(res, tid, Hs=(2,), verbose=True):
    """MAE, Skill, LSI ve DM (sızıntı) özeti. DataFrame döner."""
    rows = []
    for H in Hs:
        m = {k: mae(res[(k, H)]['y'], res[(k, H)]['p']) for k in AD}
        for k in AD:
            rows.append(dict(turbin=tid, H_saat=H / 2, model=AD[k], MAE=m[k],
                             Skill=100 * (m['B1'] - m[k]) / m['B1']))
        lv, ll = lsi(m['B7'], m['B6']), lsi(m['B8'], m['B8L'])
        dmv = dm_test(res[('B6', H)]['y'], res[('B6', H)]['p'], res[('B7', H)]['p'], h=H)
        dml = dm_test(res[('B8L', H)]['y'], res[('B8L', H)]['p'], res[('B8', H)]['p'], h=H)
        rows.append(dict(turbin=tid, H_saat=H / 2, model='LSI_VMD(%)', MAE=lv, Skill=np.nan))
        rows.append(dict(turbin=tid, H_saat=H / 2, model='LSI_LCSF(%)', MAE=ll, Skill=np.nan))
        if verbose:
            print(f'\nTürbin {tid} | H={H/2} sa')
            for k in AD: print(f'  {AD[k]:32s} MAE={m[k]:8.2f}  Skill={100*(m["B1"]-m[k])/m["B1"]:7.2f}%')
            print(f'  LSI VMD  = {lv:.1f}%  |  LSI LCSF = {ll:.1f}%')
            print(f'  DM VMD sızıntı: {dmv["DM"]:.2f}, p={dmv["p_raw"]:.2e}')
            print(f'  DM LCSF sızıntı: {dml["DM"]:.2f}, p={dml["p_raw"]:.2e}')
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- seed tekrarı
def seed_tekrar(df_all, tid, H=2, seeds=SEEDS_EK, out=OUT, L=L_BEST, epochs=None, window=W, dosya=None):
    """Yalnız LSTM'leri farklı tohumlarla yeniden eğitir (ayrıştırma önbellekten okunur, yeniden hesaplanmaz).
    Seed 42 sonuçları run_turbine çıktısındadır. Çıktı: seed_T{tid}.pkl"""
    t0 = time.time()
    td = TurbineData(df_all, tid)
    Ut, _ = build_components(td, out, window)
    Ut = {'B2': None, **{k: Ut[k] for k in ('B6', 'B7', 'B8', 'B8L')}}
    res = {}
    for key, U in Ut.items():
        D = td.get_xy(L, H) if U is None else td.get_xy_ext(L, H, U)
        for sd in seeds:
            m = fit_lstm(D, L, sd, epochs)
            res[(key, sd)] = dict(t=D['te']['t'], y=D['te']['y'], p=pred_lstm(m, D['te']['X']))
            print(f'   T{tid} {key} seed={sd} tamam | {(time.time()-t0)/60:.1f} dk')
    pickle.dump(res, open(os.path.join(out, dosya or f'seed_T{tid}.pkl'), 'wb'))   # dosya: ek tohum setini ayrı kaydetmek için
    return res


def seed_ozet(tid, H=2, seeds=SEEDS_EK, out=OUT, verbose=True):
    """Üç tohumun (42 + ek) ortalaması ve std'si; tohum başına LSI. Döner: dict."""
    base = pickle.load(open(os.path.join(out, f'adim6_T{tid}.pkl'), 'rb'))
    ek = pickle.load(open(os.path.join(out, f'seed_T{tid}.pkl'), 'rb'))
    ad = {'B2': 'Raw LSTM', 'B7': 'Rolling VMD (nedensel)', 'B8': 'LCSF nedensel',
          'B6': 'Full VMD (sızdıran)', 'B8L': 'LCSF çift yön (sızdıran)'}
    M = {}
    for k in ad:
        v = [mae(base[(k, H)]['y'], base[(k, H)]['p'])] + [mae(ek[(k, s)]['y'], ek[(k, s)]['p']) for s in seeds]
        M[k] = np.array(v)
        if verbose: print(f'  {ad[k]:28s} MAE: {np.round(v,2)}  ort={np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}')
    lv = 100 * (M['B7'] - M['B6']) / M['B7']; ll = 100 * (M['B8'] - M['B8L']) / M['B8']
    if verbose:
        print(f'  LSI VMD  (seed başına): {np.round(lv,1)}  ort={lv.mean():.1f} ± {lv.std(ddof=1):.1f}')
        print(f'  LSI LCSF (seed başına): {np.round(ll,1)}  ort={ll.mean():.1f} ± {ll.std(ddof=1):.1f}')
    return dict(MAE=M, LSI_VMD=lv, LSI_LCSF=ll)
