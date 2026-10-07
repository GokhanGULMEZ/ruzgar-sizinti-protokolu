"""ADIM 0 — Veriyi kontrol et (Makale Bölüm 3, "Veri Kümesi ve Veri Protokolü"). Model eğitilmez (saniyeler).

Üretir:
  1) türbin başına en yüksek Patv ve eksik oranı (makaledeki anma gücü / türbin seçimi cümleleri)
  2) türbin seçim kuralı: eksik orana göre en düşük, %25, %50, %75, en yüksek + T57  -> [10, 57, 61, 83, 84, 120]
  3) zaman adımı teşhisi (ZAMANA GÖRE SIRALANMIŞ): 2020 -> 15 dk, 2021 -> 10 dk, Nisan 2020 boşluğu
     (NOT: ham dosya zamana göre sıralı değildir; farkları sıralamadan almak yanlış 30 dk gösterir!)
  4) 30 dk ızgara ve bölme boyutları. Beklenen (T57): N=35.089 (=731 gün x 48 + 1); 24.562 / 5.251 / 5.252
"""
from _ortak import *
import pandas as pd
from leakage.data import TurbineData

df = veri_yukle()
print('Satır:', len(df), '| türbin sayısı:', df['TurbID'].nunique())

g = df.groupby('TurbID')
tab = pd.DataFrame({'patv_max': g['Patv'].max(), 'eksik_oran': g['Patv'].apply(lambda s: s.isna().mean()),
                    'ort_guc': g['Patv'].mean()})
print('\n1) Tüm türbinler için özet:'); print(tab.describe().round(2))
ranked = tab.sort_values('eksik_oran')
pos = [int(round(q * (len(ranked) - 1))) for q in (0, .25, .5, .75, 1)]
secilen = sorted(set(ranked.index[pos].tolist() + [57]))
print('\n2) Seçilen türbinler (eksik oran kuantilleri + T57):', secilen)
print(tab.loc[secilen].round(3))

raw = df.loc[df.TurbID == 57, ['Tmstamp']].copy()
raw['Tmstamp'] = pd.to_datetime(raw['Tmstamp'])
print('\n3) T57 zaman adımı teşhisi | tekrarlı damga:', raw.Tmstamp.duplicated().sum())
raw = raw.drop_duplicates('Tmstamp').sort_values('Tmstamp')           # SIRALA
dt = raw.Tmstamp.diff().dt.total_seconds() / 60
for yil in (2020, 2021):
    m = raw.Tmstamp.dt.year == yil
    print(f'  {yil}: satır={m.sum()} | adım aralığı (dk):'); print(dt[m].value_counts().head(3).to_string())

td = TurbineData(df, 57)
print(f'\n4) T57: N={td.N:,} | {td.d.index.min()} → {td.d.index.max()}')
for ad, x in (('eğitim', td.tr), ('doğrulama', td.va), ('test', td.te)):
    print(f'  {ad:10s}: {x.index.min()} → {x.index.max()} | satır={len(x):,}')
print('  test eksik oranı:', round(td.te['is_missing'].mean(), 4))
