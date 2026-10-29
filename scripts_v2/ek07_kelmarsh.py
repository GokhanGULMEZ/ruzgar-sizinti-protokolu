"""ek07 - Ikinci ciftlik: Kelmarsh (Zenodo 16807551, CC-BY-4.0; 6 x Senvion MM92, anma 2050 kW, 10 dk SCADA).
Ayni protokol: 30 dk izgara (:00 ve :30), kronolojik 70/15/15, purge 12, L=4, H=2 (1 sa), W=672, K=4, 3 seed LSTM.
SDWPF'ten FARKLAR (makalede acikca yazilacak): ERA5 sutunlari (T2m, Sp, RelH, Wspd_w, Wdir_w, Tp) yok -> sabit 0 (bilgisiz
oznitelik; 37 oznitelikten 31'i bilgili); anma gucu 2050 kW; olcum anlari :00/:30; iki yil (2020-2021).
Girdi: RT_KEL_DIR icinde Kelmarsh_SCADA_2020_*.zip ve Kelmarsh_SCADA_2021_*.zip (acilmadan).
Ortam: RT_PKG, RT_EK_OUT, RT_KEL_DIR, istege bagli KEL_TURB="1,3,6" (yoksa eksik-orani siralamasindan 3 turbin).
pip install vmdpy catboost
"""
import os, sys, re, io, glob, zipfile, time, pickle
sys.path.insert(0, os.environ.get('RT_PKG', '/content/paket'))
import numpy as np, pandas as pd
import leakage.config as cfg                       # ONCE yama: diger moduller bu degerleri import aninda baglar
cfg.KAPASITE = 2050.0; cfg.MEASURE_MINUTES = (0, 30)
from leakage.config import W, L_BEST
from leakage.data import TurbineData
from leakage.pipeline import build_components
from leakage.models import fit_lstm, pred_lstm
from leakage.evaluation import mae, lsi, dm_test

KEL = os.environ.get('RT_KEL_DIR', '/content/drive/MyDrive/RuzgarTahmin/veri/kelmarsh')
EK = os.environ.get('RT_EK_OUT', '/content/drive/MyDrive/RuzgarTahmin/ek_cikti/'); os.makedirs(EK, exist_ok=True)
CACHE = os.path.join(EK, 'kelmarsh_onbellek'); os.makedirs(CACHE, exist_ok=True)
H = 2; SEEDS = (42, 43, 44)
MAP = {  # SDWPF adi : Kelmarsh sutununun BASLANGICI (tam esleme icin sonu ile verilir)
    'Wspd': 'Wind speed (m/s)', 'Wdir': 'Wind direction (', 'Etmp': 'Nacelle ambient temperature (',
    'Itmp': 'Nacelle temperature (', 'Ndir': 'Nacelle position (', 'Pab1': 'Blade angle (pitch position) A (',
    'Pab2': 'Blade angle (pitch position) B (', 'Pab3': 'Blade angle (pitch position) C (',
    'Prtv': 'Reactive power (kvar)', 'Patv': 'Power (kW)'}


def read_member(zf, name):
    raw = zf.read(name).decode('utf-8', errors='replace').splitlines()
    hi = max(i for i, l in enumerate(raw[:60]) if l.startswith('#') and 'Date and time' in l)
    raw[hi] = raw[hi].lstrip('# ').strip()
    d = pd.read_csv(io.StringIO('\n'.join(raw[hi:])), low_memory=False)
    d = d.rename(columns={d.columns[0]: 'Tmstamp'})
    out = pd.DataFrame({'Tmstamp': d['Tmstamp']})
    for k, pref in MAP.items():
        c = [x for x in d.columns if (x == pref if pref.endswith(')') and '(' in pref and not pref.endswith('(') else x.startswith(pref))]
        c = [x for x in c if ',' not in x.split('(')[0]]          # 'Power, Standard deviation (kW)' gibi istatistik sutunlari ele
        if not c: raise SystemExit(f'{name}: {k} icin sutun bulunamadi ({pref})')
        out[k] = pd.to_numeric(d[c[0]], errors='coerce')
    return out


def load_all():
    fn = os.path.join(CACHE, 'kelmarsh_df.pkl')
    if os.path.exists(fn): return pd.read_pickle(fn)
    parts = []
    for z in sorted(glob.glob(os.path.join(KEL, 'Kelmarsh_SCADA_202[01]*.zip'))):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if 'Turbine_Data' in n and n.endswith('.csv'):
                    m = re.search(r'Kelmarsh_(\d+)_', n); t = read_member(zf, n); t['TurbID'] = int(m.group(1)); parts.append(t)
                    print('okundu', n, len(t))
    df = pd.concat(parts, ignore_index=True)
    for c in ('T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp'): df[c] = 0.0       # ERA5 yok: sabit, bilgisiz
    df.to_pickle(fn); return df


def main():
    df = load_all(); print(df.groupby('TurbID').size())
    miss = df.groupby('TurbID')['Patv'].apply(lambda s: s.isna().mean()).sort_values(); print('Patv eksik orani:\n', miss)
    tl = [int(x) for x in os.environ['KEL_TURB'].split(',')] if os.environ.get('KEL_TURB') else \
         [int(miss.index[0]), int(miss.index[len(miss) // 2]), int(miss.index[-1])]
    print('secilen turbinler:', tl)
    rows = []
    for tid in tl:
        t0 = time.time(); td = TurbineData(df, tid)
        print(f'T{tid}: N={td.N} egitim/dogrulama/test={len(td.tr)}/{len(td.va)}/{len(td.te)}')
        Ut, lc = build_components(td, CACHE, W)            # rolling VMD: tek turbin ~1 sa; bloklar onbellekte, yarida kalirsa devam eder
        D0 = td.get_xy(L_BEST, H); per = mae(D0['te']['y'], D0['te']['yc'])
        for sd in SEEDS:
            P = {'B2': pred_lstm(fit_lstm(D0, L_BEST, sd, None), D0['te']['X'])}; Y = {'B2': D0['te']['y']}
            for k in ('B6', 'B7', 'B8', 'B8L'):
                D = td.get_xy_ext(L_BEST, H, Ut[k]); assert np.array_equal(D['te']['t'], D0['te']['t'])
                P[k] = pred_lstm(fit_lstm(D, L_BEST, sd, None), D['te']['X']); Y[k] = D['te']['y']
            y = Y['B7']; m = {k: mae(y, P[k]) for k in P}
            r = dict(turbin=tid, seed=sd, n_test=len(y), MAE_pers=per, **{f'MAE_{k}': v for k, v in m.items()},
                     LSI_VMD=lsi(m['B7'], m['B6']), LSI_LCSF=lsi(m['B8'], m['B8L']),
                     DM_VMD=dm_test(y, P['B6'], P['B7'], h=H)['DM'], p_VMD=dm_test(y, P['B6'], P['B7'], h=H)['p_raw'],
                     DM_LCSF=dm_test(y, P['B8L'], P['B8'], h=H)['DM'], p_LCSF=dm_test(y, P['B8L'], P['B8'], h=H)['p_raw'])
            print(r, f'| {(time.time()-t0)/60:.1f} dk'); rows.append(r)
            pd.DataFrame(rows).to_csv(os.path.join(EK, 'ek07_kelmarsh.csv'), index=False)
    out = pd.DataFrame(rows)
    print(out.groupby('turbin')[['MAE_pers', 'MAE_B2', 'MAE_B7', 'MAE_B8', 'LSI_VMD', 'LSI_LCSF']].mean().round(2))

if __name__ == '__main__': main()
