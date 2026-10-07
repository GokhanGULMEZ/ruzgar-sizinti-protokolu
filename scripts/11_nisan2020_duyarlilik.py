"""ADIM 11 — Nisan 2020 boşluğunun ileri doldurulmasına duyarlılık. Yalnız LCSF (Rolling/Full VMD yeniden hesaplanmaz).
Kullanım:  python 11_nisan2020_duyarlilik.py 57        (varsayılan 57; 3 tohum: 42, 43, 44)
Süre   : 3 yöntem x 3 tohum x 2 LSTM = 18 LSTM, Colab'da yaklaşık 30-45 dk.
Çıktı  : duyarlilik_nisan2020_T{tid}.csv (tohum başına), duyarlilik_nisan2020_ozet_T{tid}.csv (ortalama ± std)
Not    : A yöntemi makaledeki temel kurulumdur; tohum 42 sonucu 116,29 / 57,34 kW (LSI %50,7) ile örtüşmelidir (T57, 1 saat)."""
from _ortak import *
import sys
from leakage.sensitivity import nisan_duyarlilik

df = veri_yukle(); out = cikti_klasoru()
tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
d, ozet = nisan_duyarlilik(df, tid)
d.to_csv(out + f'duyarlilik_nisan2020_T{tid}.csv', index=False); ozet.to_csv(out + f'duyarlilik_nisan2020_ozet_T{tid}.csv', index=False)
print('\n', ozet.round(2).to_string(index=False)); print('yazıldı:', out)
