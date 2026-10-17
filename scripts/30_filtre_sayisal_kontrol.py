"""ADIM 30 — Filtre sayısal kararlılık kontrolü: LCSF Butterworth (b,a) biçimi ile ikinci dereceden kesit (SOS) biçimi aynı mı?
Derece 3, bant sınırları 1e-4 civarına kırpılabildiği için (b,a) gösterimi kötü koşullu olabilir. Bu betik, her iki biçimin
bileşen çıktılarını (nedensel ve iki yönlü) aynı türbin için karşılaştırır ve nedensellik kontrolünü SOS için tekrarlar.
Eğitim GEREKTİRMEZ.  Kullanım: python 30_filtre_sayisal_kontrol.py [turbin=57]
Çıktı  : ekranda en büyük mutlak fark (kW) ve bileşen standart sapmasına oranı. Fark ihmal edilebilirse ana sonuçlar etkilenmez;
         değilse 'sos=True' ile yeniden koşmak gerekir (karar sizindir)."""
from _ortak import *
import sys, numpy as np
from leakage.data import TurbineData
from leakage.decomposition import LCSF

tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
df = veri_yukle(); td = TurbineData(df, tid); x = td.x_dec
lc = LCSF().fit(x[:len(td.tr)])
print(f'Türbin {tid}: merkez frekanslar (döngü/adım) = {lc.fc}')
print(f'bant sınırları = {lc.edges}')
for ad, nedensel in (('nedensel (lfilter)', True), ('iki yönlü (filtfilt)', False)):
    A = lc.transform(x, nedensel, sos=False); B = lc.transform(x, nedensel, sos=True)
    d = np.abs(A - B).max(0); sd = A.std(0)
    print(f'{ad:22s} bileşen başına en büyük |fark| (kW) = {np.round(d, 6)} | / std = {np.round(d / sd, 8)}')
tau = len(x) // 2; rng = np.random.default_rng(0); x2 = x.copy(); x2[tau + 1:] += rng.normal(0, 200, len(x) - tau - 1)
f = float(np.abs(lc.transform(x, True, sos=True)[tau] - lc.transform(x2, True, sos=True)[tau]).max())
print(f'SOS nedensel, gelecek bozuldu -> t anındaki fark = {f}   [beklenen 0]')
