"""ADIM 7 — Makaledeki tablo ve şekilleri kayıtlı sonuçlardan üret (Bölüm 6). Model eğitilmez (saniyeler).
Girdi : adim2_T57_tahminler.pkl, adim6_T{tid}.pkl, seed_T{tid}.pkl  (OUT klasöründe)
Çıktı : tablo_turbinler_1sa.csv, tablo_seed_lsi.csv, tablo_tum_metrikler.csv (RMSE, nMAE, nRMSE, R2, Skill), sekil1_ufuk.png, sekil2_lsi.png
Makaledeki sayılar buradan çıkan değerlerle birebir örtüşmelidir; elle sayı yazılmaz."""
from _ortak import *
import os, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from leakage.evaluation import mae, lsi, metrics
from leakage.pipeline import AD, seed_ozet

OUT = cikti_klasoru(); H = 2
TIDS = [10, 83, 120, 61, 84]
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

# --- Tablo: türbin x model (1 saat, tohum 42) ---
t57 = pickle.load(open(OUT + 'adim2_T57_tahminler.pkl', 'rb'))
sets = {57: t57, **{t: pickle.load(open(OUT + f'adim6_T{t}.pkl', 'rb')) for t in TIDS if os.path.exists(OUT + f'adim6_T{t}.pkl')}}
rows = []
for tid, res in sets.items():
    m = {k: mae(res[(k, H)]['y'], res[(k, H)]['p']) for k in AD if (k, H) in res}
    r = {'turbin': tid, **{AD[k]: round(v, 2) for k, v in m.items()}}
    if 'B6' in m and 'B7' in m: r['LSI_VMD(%)'] = round(lsi(m['B7'], m['B6']), 1)
    if 'B8' in m and 'B8L' in m: r['LSI_LCSF(%)'] = round(lsi(m['B8'], m['B8L']), 1)
    rows.append(r)
tab = pd.DataFrame(rows); tab.to_csv(OUT + 'tablo_turbinler_1sa.csv', index=False); print(tab.to_string(index=False))

# --- Tablo: TÜM ölçütler (hakem isteği: RMSE, nMAE, nRMSE, R2, Skill) — her türbin, ufuk ve model için ---
AD_ALL = {'B1': 'Persistence', 'B2': 'Ham LSTM', 'B3': 'CatBoost Direct', 'B6': 'Tam seri VMD-LSTM (sızdıran)',
          'B7': 'Kayan VMD-LSTM (nedensel)', 'B8': 'LCSF-LSTM (nedensel)', 'B8L': 'LCSF iki yönlü (sızdıran)',
          'B9': 'B9 LCSF+OOF CatBoost', 'B8I': 'B8I (eğitim-içi artık)', 'B2O': 'B2O (Ham LSTM+OOF)'}
rows = []
for tid, res in sets.items():
    for (k, h), r in sorted(res.items()):
        rows.append(dict(turbin=tid, model=AD_ALL.get(k, k), H_saat=h / 2, **metrics(r['y'], r['p'], r['yc']) ) if 'yc' in r else
                    dict(turbin=tid, model=AD_ALL.get(k, k), H_saat=h / 2))
pd.DataFrame(rows).round(3).to_csv(OUT + 'tablo_tum_metrikler.csv', index=False)
print('tablo_tum_metrikler.csv yazıldı:', len(rows), 'satır')

# --- Tablo: tohum tekrarı LSI ---
sl, lsi_v, lsi_l, tids_ok = [], {}, {}, []
for t in TIDS:
    if os.path.exists(OUT + f'seed_T{t}.pkl') and os.path.exists(OUT + f'adim6_T{t}.pkl'):
        s = seed_ozet(t, verbose=False); tids_ok.append(t)
        lsi_v[t], lsi_l[t] = s['LSI_VMD'], s['LSI_LCSF']
        sl.append(dict(turbin=t, LSI_VMD_ort=s['LSI_VMD'].mean(), LSI_VMD_std=s['LSI_VMD'].std(ddof=1),
                       LSI_LCSF_ort=s['LSI_LCSF'].mean(), LSI_LCSF_std=s['LSI_LCSF'].std(ddof=1)))
if sl:
    sd = pd.DataFrame(sl).round(1); sd.to_csv(OUT + 'tablo_seed_lsi.csv', index=False); print(sd.to_string(index=False))
    allv = np.concatenate(list(lsi_v.values())); alll = np.concatenate(list(lsi_l.values()))
    print(f'Havuzlanmış ({len(tids_ok)} türbin x 3 tohum): LSI VMD {allv.mean():.1f} ± {allv.std(ddof=1):.1f} | '
          f'LSI LCSF {alll.mean():.1f} ± {alll.std(ddof=1):.1f}')

# --- Şekil 1: T57 MAE – ufuk ---
Hs = sorted({h for (k, h) in t57 if k == 'B7'})
get = lambda k: [mae(t57[(k, h)]['y'], t57[(k, h)]['p']) for h in Hs]
fig, ax = plt.subplots(figsize=(6.3, 3.5), dpi=300); x = [h / 2 for h in Hs]
ax.plot(x, get('B1'), color=GRAY, ls='--', lw=1.8, marker='s', ms=4, label='Persistence')
ax.plot(x, get('B3'), color=BLUE, lw=2, marker='o', ms=5, label='CatBoost Direct (nedensel)')
ax.plot(x, get('B7'), color=ORANGE, lw=2, marker='^', ms=5, label='Kayan VMD–LSTM (nedensel)')
ax.plot(x, get('B6'), color=AQUA, lw=2, marker='D', ms=4.5, label='Tam seri VMD–LSTM (sızdıran)')
ax.fill_between(x, get('B6'), get('B7'), color=AQUA, alpha=0.10, lw=0)
ax.set_xlabel('Tahmin ufku (saat)'); ax.set_ylabel('MAE (kW)'); ax.set_xticks(x); ax.set_ylim(bottom=0)
ax.grid(axis='y', color='#e5e4df', lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + 'sekil1_ufuk.png'); plt.close(fig)

# --- Şekil 2: türbin başına LSI (T57 tek tohum, diğerleri 3 tohum ort ± std) ---
labels = ['T57*'] + [f'T{t}' for t in tids_ok]
m57 = {k: mae(t57[(k, H)]['y'], t57[(k, H)]['p']) for k in ('B6', 'B7', 'B8', 'B8L')}
vm = [lsi(m57['B7'], m57['B6'])] + [lsi_v[t].mean() for t in tids_ok]; vs = [np.nan] + [lsi_v[t].std(ddof=1) for t in tids_ok]
lc = [lsi(m57['B8'], m57['B8L'])] + [lsi_l[t].mean() for t in tids_ok]; ls = [np.nan] + [lsi_l[t].std(ddof=1) for t in tids_ok]
xs = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(6.3, 3.4), dpi=300)
ax.bar(xs - w / 2 - 0.01, vm, w - 0.02, color=BLUE, label='LSI (VMD)'); ax.bar(xs + w / 2 + 0.01, lc, w - 0.02, color=ORANGE, label='LSI (LCSF)')
ax.errorbar(xs - w / 2 - 0.01, vm, yerr=vs, fmt='none', ecolor='k', elinewidth=1, capsize=2.5)
ax.errorbar(xs + w / 2 + 0.01, lc, yerr=ls, fmt='none', ecolor='k', elinewidth=1, capsize=2.5)
ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylabel('LSI (%)'); ax.set_ylim(0, 70)
ax.grid(axis='y', color='#e5e4df', lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + 'sekil2_lsi.png'); plt.close(fig)
print('✓ Tablolar ve şekiller yazıldı:', OUT)
