"""Deney 5 - placebo-gelecek kontrolü (pencere geometrisi sabit, gelecek bilgisi yok). Ayrıntı: leakage/placebo.py.
Önkoşul: 02_turbin_kosusu.py 57 çalışmış olmalı (vmd_full_T57.npy ve vmd_roll_T57.npy önbellekleri, OUT klasöründe).
Kullanım:
  python deney5_placebo_gelecek.py 57 lcsf 1 2 4 8      # dakikalar; ÖNCE bunu çalıştırın
  python deney5_placebo_gelecek.py 57 vmd 2             # her (m, kuyruk) için ~1 saat; bloklar diske yazılır, yarıda kalırsa aynı komut sürer
  python deney5_placebo_gelecek.py 57 vmd 8
Ortam: RT_VERI_CSV, RT_OUT (diğer betiklerle aynı). RT_SEEDS="42,43,44" (varsayılan). RT_MODLAR="ayna,artim" (varsayılan; 'tut' da eklenebilir).
Gerçek-gelecek kolu: LCSF'de aynı koşuda yeniden üretilir. VMD'de 17_vmd_ileri_bakis.py'nin ürettiği vmd_ileri_T57_m{m}.npy varsa eklenir.
Çıktı: placebo_{tur}_T{tid}.csv (tohum başına), placebo_{tur}_ozet_T{tid}.csv (ortalama ± std)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
from leakage import config
from leakage.placebo import placebo_egrisi

if not os.path.exists(config.VERI_CSV):
    sys.exit(f'Veri bulunamadı: {config.VERI_CSV} (RT_VERI_CSV ile yolu verin)')
df = pd.read_csv(config.VERI_CSV); out = config.OUT; os.makedirs(out, exist_ok=True)
tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
tur = sys.argv[2] if len(sys.argv) > 2 else 'lcsf'
ms = tuple(int(a) for a in sys.argv[3:]) or ((1, 2, 4, 8) if tur == 'lcsf' else (2,))
seeds = tuple(int(a) for a in os.environ.get('RT_SEEDS', '42,43,44').split(','))
modlar = tuple(os.environ.get('RT_MODLAR', 'ayna,artim').split(','))
d, ozet = placebo_egrisi(df, tid, tur, ms, modlar, seeds=seeds, out=out)
sfx = f'_{tur}_m' + '-'.join(map(str, ms))
d.to_csv(out + f'placebo{sfx}_T{tid}.csv', index=False); ozet.to_csv(out + f'placebo_ozet{sfx}_T{tid}.csv', index=False)
print('\n', ozet.round(2).to_string(index=False)); print('yazıldı:', out)
