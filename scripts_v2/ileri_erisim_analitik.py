"""Kac adim ileri bakiyor? -- iki yonlu (sizdiran) LCSF icin ANALITIK erisim.
Veri gerekmez. Nedensel olmayan filtre = filtfilt; bir bilesenin t anindaki degeri, gelecekteki hangi ornek
konumlarina ne kadar agirlikla bagli? Bunu birim darbe cevabi (impulse response) verir.

Yontem: uzun bir sifir dizinin ortasina birim darbe koy, paketteki gibi filtfilt(b, a) uygula. Cikti h[k] =
'k adim sonraki ornegin t anindaki bilesene agirligi' (sifir fazli oldugundan h[k] = h[-k]).
Raporlanan: (i) genlik erisimi: |h[k]| > %1 * max|h| olan en buyuk k; (ii) enerji erisimi: gelecek tarafta kalan
enerji payi <= eps olan en kucuk k. Adim = 30 dk.
Bant kenarlari (cycle/step): v1 paketindeki T57 degerleri: 0.0032, 0.0285, 0.0629.
Calistir: python ileri_erisim_analitik.py   (scipy, numpy, pandas gerekir; saniyeler)"""
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, lfilter
ORDER = 3; EDGES = [0.0032, 0.0285, 0.0629]; N = 40001; C = N // 2; DT_H = 0.5
def bands():
    out = [('K1 low-pass', butter(ORDER, EDGES[0], 'low', fs=1.0))]
    for i in range(1, 3):
        out.append((f'K{i+1} band-pass', butter(ORDER, [EDGES[i-1], EDGES[i]], 'band', fs=1.0)))
    out.append(('K4 high-pass', butter(ORDER, EDGES[2], 'high', fs=1.0)))
    return out
rows = []
for ad, (b, a) in bands():
    x = np.zeros(N); x[C] = 1.0
    h2 = filtfilt(b, a, x)                    # iki yonlu (sizdiran)
    h1 = lfilter(b, a, x)                     # tek yonlu (nedensel)
    fut = h2[C + 1:]; e = fut ** 2; tot = (h2 ** 2).sum()
    amp = np.abs(h2).max()
    k_amp = int(np.where(np.abs(fut) > 0.01 * amp)[0].max() + 1)
    tail = np.cumsum(e[::-1])[::-1] / tot     # tail[k-1]: k. adimdan sonrasi enerji payi (gelecek taraf)
    r = {'bilesen': ad, 'genlik_erisimi_adim(%1)': k_amp, 'saat': k_amp * DT_H}
    for eps in (1e-2, 1e-3):
        k = int(np.where(tail > eps)[0].max() + 2) if (tail > eps).any() else 1
        r[f'enerji_erisimi_adim(eps={eps:g})'] = k; r[f'saat(eps={eps:g})'] = k * DT_H
    cum = np.cumsum(e) / e.sum()
    for m in (1, 2, 4, 8, 16, 32):
        r[f'gelecek_enerjinin_%_{m}adimda'] = round(100 * cum[m - 1], 1)
    r['nedensel_h[k>0]_maks_mutlak'] = float(np.abs(h1[:C]).max())   # nedensel filtre: gecmis tarafi sifir olmali
    rows.append(r)
df = pd.DataFrame(rows).set_index('bilesen')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df.T.to_string())
df.to_csv('ileri_erisim_analitik.csv')
