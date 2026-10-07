"""ADIM 25 — ERA5 doğrusal ara değerlemesinin (data.py:34, interpolate(limit=2)) sonuçlara etkisini ÖLÇER. Model eğitilmez (~1-2 dk).
Soru: interpolate, t anındaki eksik ERA5 değerini t+1 gibi gelecek gözlemlerden doldurabilir. Bu kaç pencereyi etkiliyor?
Yöntem: Türbin verisi iki kez kurulur: (A) eski kod (doğrusal ara değerleme) ve (B) güncel kod: yalnız geçmişe bakan ileri doldurma (ffill, aynı limit=2).
Sonuç (7 Ekim 2026, 6 türbin): ERA5 eksikleri yalnız Patv'nin de eksik olduğu satırlardadır (Nisan 2020 boşluğu); etkilenen eğitim/doğrulama/test penceresi sayısı 0.
        Ardından L=4, H=2 pencereleri eğitim/doğrulama/test için karşılaştırılır: kaç pencerenin girdisi iki sürümde farklı?
Kullanım: python 25_era5_interpolasyon_kontrol.py             (6 türbin)   |   python 25_era5_interpolasyon_kontrol.py 57
Çıktı  : ekranda tablo + era5_interpolasyon_kontrol.csv (OUT klasörü). Yalnız ölçüm yapar. `10dk` argümanı: betik 23'ün 2021 / 10 dk ızgarası için yalnız sayım (python 25_era5_interpolasyon_kontrol.py 10dk 57)."""
from _ortak import *
import sys, numpy as np, pandas as pd
from unittest import mock
from leakage import data as D

ERA5 = ['T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp']
L, H = 4, 2
df = veri_yukle(); out = cikti_klasoru()
if '10dk' in sys.argv[1:]:        # Yalnız SAYIM: 2021, yerel 10 dk ızgarada ERA5 eksik ve Patv geçerli hücre var mı? (betik 23 için)
    tid = next((int(a) for a in sys.argv[1:] if a.isdigit()), 57)
    x = df[df['TurbID'] == tid].copy(); x['Tmstamp'] = pd.to_datetime(x['Tmstamp'])
    x = x.sort_values('Tmstamp').drop_duplicates('Tmstamp').set_index('Tmstamp'); x = x[x.index.year == 2021]
    x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq='10min'))
    nan_h = x[ERA5].isna(); ok = x['Patv'].notna()
    print(f'[10dk, 2021, T{tid}] satır={len(x)} | ERA5 eksik hücre={int(nan_h.values.sum())} | ERA5 eksik ve Patv geçerli={int((nan_h.values & ok.values[:, None]).sum())}')
    print('Bu sayı 0 ise ERA5 doldurma yönteminin (ara değerleme / ileri doldurma) 10 dk deneyinde hiçbir etkisi yoktur.')
    sys.exit()
tids = [int(a) for a in sys.argv[1:]] or [57, 10, 83, 120, 61, 84]

def aradeger_yerine(self, **k):          # (A) sürümü: data.py'deki DataFrame.ffill çağrısı yerine ESKİ doğrusal ara değerleme (aynı limit)
    return self.interpolate(method='linear', limit=k.get('limit'))

rows = []
for tid in tids:
    ham = D.load_turbine(df, tid)
    nan_h = ham[ERA5].isna(); patv_ok = ham['Patv'].notna()
    ara = ham[ERA5].interpolate(method='linear', limit=2); ff = ham[ERA5].ffill(limit=2)
    doldurulan = nan_h & ara.notna()                                      # ara değerlemeyle doldurulan hücreler
    gelecek = doldurulan & (ara.values != ff.values)                      # değeri ileri doldurmadan FARKLI çıkanlar (gelecek komşuya bağlı)
    with mock.patch.object(pd.DataFrame, 'ffill', aradeger_yerine):     # A: eski kod (ara değerleme)
        td_a = D.TurbineData(df, tid)
    td_b = D.TurbineData(df, tid)                                       # B: güncel kod (yalnız geçmişe bakan ileri doldurma)
    Da, Db = td_a.get_xy(L, H), td_b.get_xy(L, H)
    r = dict(turbin=tid, era5_nan_hucre=int(nan_h.values.sum()), era5_nan_ve_patv_gecerli=int((nan_h.values & patv_ok.values[:, None]).sum()),
             aradeger_doldurulan=int(doldurulan.values.sum()), gelecege_bagli=int(gelecek.values.sum()))
    for ad in ('tr', 'va', 'te'):
        a, b = Da[ad], Db[ad]
        ayni_t = np.array_equal(a['t'], b['t'])
        if ayni_t and len(a['t']):
            fark = np.any(np.abs(a['X'] - b['X']) > 0, axis=(1, 2)); n = int(fark.sum())
        else:
            n = -1
        r[f'{ad}_pencere'] = len(a['t']); r[f'{ad}_etkilenen'] = n; r[f'{ad}_ayni_pencere_kumesi'] = ayni_t
    rows.append(r)
R = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(R.to_string(index=False)); R.to_csv(out + 'era5_interpolasyon_kontrol.csv', index=False)
print('\nYorum: tr/va/te_etkilenen = girdisi ara değerleme ile ileri doldurma arasında farklı olan pencere sayısı. '
      'te_etkilenen = 0 ise test girdileri etkilenmemiştir; gelecege_bagli = 0 ise hiçbir hücre gelecek komşudan doldurulmamıştır.')
