"""EK C — CatBoost öznitelik önemi: sızdıran vs nedensel bileşenler (+ CatBoost ailesinde LSI).
Kullanım:  python 09_ek_c_oznitelik_onemi.py 57        (varsayılan 57; 1 saat ufuk)
          RT_DIL=en ortam değişkeniyle şekil etiketleri İngilizce olur (dosya adlarına _en eklenir; tablolar değişmez)
Girdi : tam seri VMD ve Rolling VMD önbelleği (OUT'ta vmd_full_T57.npy, vmd_roll_T57.npy; yoksa hesaplanır, ~1 saat)
Süre  : önbellek varsa ~5-10 dk (5 CatBoost + permütasyon önemi).
Çıktı : ekc_ozet_T{tid}.csv, ekc_onem_T{tid}.csv (permütasyon + SHAP), sekil_ekc_T{tid}.png,
        sekil_ekc_shap_{kod}_T{tid}.png (SHAP beeswarm; B3,B7,B8,B6,B8L)
Gereksinim: pip install shap  (yoksa beeswarm atlanır, tablolar yine yazılır)"""
from _ortak import *
import sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from leakage.importance import ek_c
import os, re
EN = os.environ.get('RT_DIL', '').lower() == 'en'; SFX = '_en' if EN else ''
_cev = lambda ad: re.sub(r'^Bileşen(\d+)', r'Component \1', ad) if EN else ad

df = veri_yukle(); out = cikti_klasoru()
tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
ozet, onem, shap_veri = ek_c(df, tid, H=2, out=out)
ozet.to_csv(out + f'ekc_ozet_T{tid}.csv', index=False); onem.to_csv(out + f'ekc_onem_T{tid}.csv', index=False)

# Şekil: model başına permütasyon önemi (ΔMAE, kW) — en önemli 8 öznitelik
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
kodlar = ['B3', 'B7', 'B8', 'B6', 'B8L']
_MODEL_EN = {'B3': 'CatBoost (37 features only)', 'B7': 'CatBoost + rolling VMD (causal)', 'B8': 'CatBoost + LCSF (causal)',
             'B6': 'CatBoost + full-series VMD (leaky)', 'B8L': 'CatBoost + two-sided LCSF (leaky)'}
fig, ax = plt.subplots(1, 5, figsize=(13, 3.2), sharex=True)
for a, k in zip(ax, kodlar):
    d = onem[onem.kod == k].nlargest(8, 'perm_dMAE_kW').iloc[::-1]
    renk = ['#eb6834' if s.startswith('Bileşen') else '#2a78d6' for s in d.oznitelik]
    a.barh([_cev(x) for x in d.oznitelik], d.perm_dMAE_kW, color=renk)
    a.set_title(_MODEL_EN.get(k, d.model.iloc[0]).replace('CatBoost ', 'CB ') if EN else d.model.iloc[0].replace('CatBoost ', 'CB '), fontsize=7)
ax[0].set_ylabel('feature' if EN else 'öznitelik')
fig.supxlabel('Permutation importance: ΔMAE (kW)  —  orange = decomposition component' if EN else 'Permütasyon önemi: ΔMAE (kW)  —  turuncu = ayrıştırma bileşeni')
fig.tight_layout(); fig.savefig(out + f'sekil_ekc_T{tid}{SFX}.png', dpi=200); print('yazıldı:', out)

# SHAP beeswarm (yalnız `shap` kuruluysa)
try:
    import shap
except ImportError:
    print('shap kurulu değil: beeswarm atlandı (pip install shap)'); shap = None
if shap is not None:
    BASLIK = {'B3': 'CatBoost, 37 öznitelik', 'B7': '+ kayan VMD (nedensel)', 'B8': '+ LCSF (nedensel)',
              'B6': '+ tam seri VMD (sızdıran)', 'B8L': '+ LCSF iki yönlü (sızdıran)'}
    for kod, (X, sv, names) in shap_veri.items():
        plt.figure(figsize=(6, 4)); shap.summary_plot(sv, X, feature_names=[_cev(n) for n in names], max_display=10, show=False)
        plt.title(({'B3': 'CatBoost, 37 features', 'B7': '+ rolling VMD (causal)', 'B8': '+ LCSF (causal)', 'B6': '+ full-series VMD (leaky)', 'B8L': '+ two-sided LCSF (leaky)'}[kod]) + f' (T{tid}, 1 h)' if EN else BASLIK[kod] + f' (T{tid}, 1 saat)', fontsize=9); plt.tight_layout()
        plt.savefig(out + f'sekil_ekc_shap_{kod}_T{tid}{SFX}.png', dpi=200); plt.close()
    print('SHAP şekilleri yazıldı')
