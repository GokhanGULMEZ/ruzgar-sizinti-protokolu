"""ADIM 19 — Kayan VMD için 'geriye kaydırılmış nedensel' sınır-etkisi kontrolü (Bölüm 6.11 tamamlayıcısı).
Önkoşul: 02_turbin_kosusu.py 57 (vmd_full_T57.npy, vmd_roll_T57.npy).
Kullanım:  python 19_vmd_geri_kontrol.py 57 1 2 4      (türbin, ardından d değerleri; varsayılan 1 2 4)
Süre   : kayan VMD ön hesabı ~1 saat (tek çekirdek; tüm d aynı çözümden) + LSTM (3 tohum x (d sayısı + 2)) ≈ 20-30 dk.
         Bloklar diske yazılır; oturum kapanırsa AYNI komutu yeniden çalıştırın, kaldığı yerden sürer.
Çıktı  : vmd_geri_T57.csv, vmd_geri_ozet_T57.csv   (IYILESME > 0: pencere sonundan uzaklaşmak nedensel hatayı azaltıyor)"""
from _ortak import *
import sys
from leakage.vmd_geri import geri_egrisi

df = veri_yukle(); out = cikti_klasoru()
tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
ds = tuple(int(a) for a in sys.argv[2:]) or (1, 2, 4)
d, ozet = geri_egrisi(df, tid, ds=ds)
d.to_csv(out + f'vmd_geri_T{tid}.csv', index=False); ozet.to_csv(out + f'vmd_geri_ozet_T{tid}.csv', index=False)
print('\n', ozet.round(2).to_string(index=False)); print('yazıldı:', out)
