"""ek06 - SDWPF ornekleme araligi kontrolu.
Kullanim: python ek06_ornekleme_kontrol.py <csv_yolu> [turbin_no]
Hem bizim sdwpf_2001_2112_full.csv'yi hem figshare/Scientific Data'dan indirdiginiz ORIJINAL dosyayi verin; ciktilari karsilastirin.
Orijinal makale (Sci Data 2024) 10 dk diyor; bizim dosyada 2020 verisi 15 dk gorunuyor."""
import sys, pandas as pd
p = sys.argv[1]; tid = int(sys.argv[2]) if len(sys.argv) > 2 else 57
d = pd.read_csv(p); print('sutunlar:', list(d.columns)[:8], '| satir:', len(d))
d = d[d['TurbID'] == tid].copy()
if 'Tmstamp' in d: d['t'] = pd.to_datetime(d['Tmstamp'])
else: print('Tmstamp yok: Day/Tmstamp (HH:MM) bicimi icin elle uyarlayin'); sys.exit()
d = d.sort_values('t'); dt = d['t'].diff().dropna()
print('ilk/son:', d['t'].iloc[0], d['t'].iloc[-1])
for y, g in dt.groupby(d['t'].iloc[1:].dt.year): print(y, 'araliklar:', g.value_counts().head(4).to_dict())
print('dakikalar (2020):', d[d.t.dt.year == 2020].t.dt.minute.value_counts().head(8).to_dict())
