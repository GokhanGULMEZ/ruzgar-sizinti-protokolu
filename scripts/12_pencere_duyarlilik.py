"""ADIM 12 — Rolling VMD pencere uzunluğu duyarlılığı : 7 / 14 / 28 gün.
Kullanım:  python 12_pencere_duyarlilik.py 57            (varsayılan 57; tohum 42; 1 saat ufku)
           python 12_pencere_duyarlilik.py 57 336        (yalnız tek pencere; gece bölerek çalıştırmak için)
Süre   : Rolling VMD ön hesabı pencereyle orantılı: 336 -> ~30 dk, 1344 -> ~2 sa (Colab, 2 çekirdek); 672 mevcut önbellekten.
         Her pencere için 4 LSTM (B6, B7, B8, B8L). Yarıda kalırsa önbellek bloklarından devam eder.
Çıktı  : duyarlilik_pencere_T{tid}.csv (tek pencere verilirse duyarlilik_pencere_T{tid}_W{w}.csv)
Not    : 672 satırı ana koşuyla (MAE 116,11 / 64,71; LSI-VMD %44,3) örtüşmelidir (tohum 42, T57, 1 saat)."""
from _ortak import *
import sys
from leakage.sensitivity import pencere_duyarlilik

df = veri_yukle(); out = cikti_klasoru()
tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
ws = (int(sys.argv[2]),) if len(sys.argv) > 2 else (336, 672, 1344)
d = pencere_duyarlilik(df, tid, windows=ws)
ad = f'duyarlilik_pencere_T{tid}' + (f'_W{ws[0]}' if len(ws) == 1 else '') + '.csv'
d.to_csv(out + ad, index=False); print('\n', d.round(2).to_string(index=False)); print('yazıldı:', out + ad)
