"""ADIM 14 — İleri-bakış doz–tepki eğrisi (LCSF): sızıntı, bileşenin kaç adım GELECEĞE baktığına göre nasıl artar?
Kullanım:  python 14_ileri_bakis_egrisi.py 57          (varsayılan 57, 1 saat, tohum 42/43/44)
Süre   : her m için bileşen üretimi ~1 dk; 3 tohum x 9 LSTM (7 m + nedensel + tam seri) ≈ 27 LSTM -> Colab'da ~1 saat.
Çıktı  : ileri_bakis_T{tid}.csv (tohum başına), ileri_bakis_ozet_T{tid}.csv (ortalama ± std)
Beklenen: m=0 satırında LSI ~0 (sıfır fazlı ama gelecek yok); m arttıkça LSI artar; m>=H (=2) hedefi içerir."""
from _ortak import *
import sys
from leakage.lookahead import ileri_bakis_egrisi

df = veri_yukle(); out = cikti_klasoru()
tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
d, ozet = ileri_bakis_egrisi(df, tid)
d.to_csv(out + f'ileri_bakis_T{tid}.csv', index=False); ozet.to_csv(out + f'ileri_bakis_ozet_T{tid}.csv', index=False)
print('\n', ozet.round(2).to_string(index=False)); print('yazıldı:', out)
