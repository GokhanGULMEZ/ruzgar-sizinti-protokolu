"""ADIM 17 — Kayan VMD için ileri-bakış doz–tepki deneyi (VMD sınır etkisini sızıntıdan ayırma; Bölüm 6.11).
Önkoşul: 02_turbin_kosusu.py 57 (vmd_full_T57.npy, vmd_roll_T57.npy önbellekleri).
Kullanım:  python 17_vmd_ileri_bakis.py 57 1 2 8     (türbin, ardından m değerleri; m çift/tek olabilir)
Süre   : HER m için bileşen üretimi ~1 saat (tek çekirdek) + LSTM eğitimi (3 tohum x (m sayısı + 2) model).
         Bloklar diske yazılır; oturum kapanırsa AYNI komutu yeniden çalıştırın, kaldığı yerden sürer.
Öneri  : önce  python 17_vmd_ileri_bakis.py 57 1   (en bilgilendirici: 30 dk ileri), sonra  ... 57 2 8.
Çıktı  : vmd_ileri_T57.csv, vmd_ileri_ozet_T57.csv"""
from _ortak import *
import sys
from leakage.vmd_ileri import vmd_ileri_egrisi

df = veri_yukle(); out = cikti_klasoru()
tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
ms = tuple(int(a) for a in sys.argv[2:]) or (1, 2, 8)
d, ozet = vmd_ileri_egrisi(df, tid, ms=ms)
sfx = '_m' + '-'.join(map(str, ms))
d.to_csv(out + f'vmd_ileri_T{tid}{sfx}.csv', index=False); ozet.to_csv(out + f'vmd_ileri_ozet_T{tid}{sfx}.csv', index=False)
print('\n', ozet.round(2).to_string(index=False)); print('yazıldı:', out)
