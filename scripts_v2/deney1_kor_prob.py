"""Deney 1 - Kor (blind) prob testi. Egitim yok, paket gerekmez (numpy, scipy, pandas).
Amac : Probun, sizintisi ONCEDEN bilinmeyen ve bazilari 'nedensel gorunen' operatorleri ayirt edip edemedigini sinamak.
Yontem: 8 operator rastgele sirayla ve kodlarla (O1..O8) probdan gecirilir; karar yalniz probdan gelir. Etiketler en sonda acilir.
Prob : Test 1 gelecek bozma D(tau;delta), Test 2 onek degismezligi. Karar: D>tol veya onek farki>tol ise 'SIZDIRIYOR'.
Girdi: python deney1_kor_prob.py [x57.npy]   (npy verilmezse sentetik seri kullanilir; gercek seri icin Colab'da x_dec kaydedilir)
Cikti: deney1_kor_prob.csv"""
import sys, numpy as np, pandas as pd
from scipy.signal import butter, lfilter, filtfilt

def seri(csv=None, tid=57, n=6000):
    if csv and csv.endswith('.npy'):
        return np.load(csv)[:n].astype(float)
    if csv:
        df = pd.read_csv(csv); d = df[df['TurbID'] == tid].sort_values(['Day', 'Tmstamp']) if 'TurbID' in df else df
        x = d['Patv'].ffill().fillna(0).values.astype(float)
        return x[:n]
    r = np.random.default_rng(0); t = np.arange(n)
    x = 700 + 300*np.sin(2*np.pi*t/48) + 150*np.sin(2*np.pi*t/336)
    e = r.normal(0, 60, n); a = np.zeros(n)
    for i in range(1, n): a[i] = 0.9*a[i-1] + e[i]
    return np.clip(x + a, 0, 1500)

EDGES = [0.0032, 0.0285, 0.0629]
def banka(kind):
    B = [butter(3, EDGES[0], 'low')]
    B += [butter(3, [EDGES[0], EDGES[1]], 'band'), butter(3, [EDGES[1], EDGES[2]], 'band')]
    B += [butter(3, EDGES[2], 'high')]
    return B
def apply_bank(x, B, two_sided=False):
    return np.column_stack([(filtfilt(b, a, x) if two_sided else lfilter(b, a, x)) for b, a in B])

def ema_bank(x):                       # nedensel
    out = []
    for al in (0.5, 0.1, 0.02):
        y = np.empty_like(x); y[0] = x[0]
        for i in range(1, len(x)): y[i] = al*x[i] + (1-al)*y[i-1]
        out.append(y)
    return np.column_stack(out)
def O_lcsf_causal(x): return apply_bank(x, banka(0), False)
def O_pencere_filtfilt(x, W=240, ileri=0):   # pencere t+ileri'de biter; ileri=0 nedensel (bilgi acisindan)
    N = len(x); out = np.full((N, 4), np.nan); B = banka(0)
    for t in range(W, N):
        e = min(t + ileri + 1, N); w = x[e-W:e]
        z = apply_bank(w, B, True)
        out[t] = z[t - (e - W)]
    out[:W] = out[W]
    return out
def O_merkez_ortalama(x, h=1):               # merkezli hareketli ortalama, h adim ileri
    k = 2*h + 1; c = np.convolve(np.pad(x, h, mode='edge'), np.ones(k)/k, mode='valid')
    return np.column_stack([c, x - c])
def O_global_z(x):                           # global istatistikle olcekle, sonra nedensel EMA (gorunurde nedensel)
    z = (x - x.mean())/x.std(); return ema_bank(z)
def O_bant_tum_seri(x):                      # bant kenarlari TUM serinin periyodogramindan; sonra nedensel suzme
    f = np.fft.rfftfreq(len(x)); p = np.abs(np.fft.rfft(x - x.mean()))**2
    ks = np.argsort(p[1:])[-3:] + 1; fc = np.sort(f[ks]); c = 0.5*(fc[:-1] + fc[1:]) if len(fc) > 1 else fc
    e1 = max(0.002, min(0.45, c[0])); e2 = max(e1*1.5, min(0.45, c[-1]))
    B = [butter(3, e1, 'low'), butter(3, [e1, e2], 'band'), butter(3, e2, 'high')]
    return apply_bank(x, B, False)
def O_lcsf_iki_yonlu(x): return apply_bank(x, banka(0), True)
def O_pencere_ileri1(x): return O_pencere_filtfilt(x, ileri=1)

OPS = {'lcsf_nedensel': (O_lcsf_causal, False), 'ema_nedensel': (ema_bank, False),
       'pencere_filtfilt_t': (O_pencere_filtfilt, False), 'merkez_ort_h1': (O_merkez_ortalama, True),
       'global_z_ema': (O_global_z, True), 'bant_tum_seri': (O_bant_tum_seri, True),
       'lcsf_iki_yonlu': (O_lcsf_iki_yonlu, True), 'pencere_filtfilt_t+1': (O_pencere_ileri1, True)}

def prob(f, x, delta=200.0, taus=(0, 1, 2, 4, 8, 16, 32), n_ep=12, n_cut=12, seed=3):
    r = np.random.default_rng(seed); N = len(x); Z0 = f(x)
    t0s = r.integers(400, N - 40, n_ep); D = np.zeros((n_ep, len(taus)))
    for i, t0 in enumerate(t0s):
        x2 = x.copy(); x2[t0+1:] += r.normal(0, delta, N - t0 - 1); Z2 = f(x2)
        for j, tau in enumerate(taus):
            D[i, j] = np.nanmax(np.abs(Z0[t0 - tau] - Z2[t0 - tau]))
    cuts = r.integers(400, N - 40, n_cut); pf = 0.0
    for t in cuts:
        pf = max(pf, float(np.nanmax(np.abs(f(x[:t+1])[-1] - Z0[t]))))
    return D.max(0), pf

if __name__ == '__main__':
    a = sys.argv[1:]; x = seri(a[0] if a else None, int(a[1]) if len(a) > 1 else 57)
    names = list(OPS); order = np.random.default_rng(1).permutation(len(names))
    kod = {f'O{i+1}': names[j] for i, j in enumerate(order)}
    rows = []
    for k, ad in kod.items():
        f, gercek_sizinti = OPS[ad]; D, pf = prob(f, x)
        karar = 'SIZDIRIYOR' if (D.max() > 1e-6 or pf > 1e-6) else 'sifir'
        rows.append(dict(kod=k, D_tau0=D[0], D_tau8=D[4], D_tau32=D[6], onek_maks_fark=pf, karar=karar, _ad=ad, _gercek='sizdiran' if gercek_sizinti else 'nedensel'))
        print(k, 'D(0)=%.4g D(8)=%.4g D(32)=%.4g onek=%.4g ->' % (D[0], D[4], D[6], pf), karar, flush=True)
    df = pd.DataFrame(rows); df['dogru'] = (df.karar == 'SIZDIRIYOR') == (df._gercek == 'sizdiran')
    print('\n=== ETIKETLER ACILDI ==='); print(df[['kod', '_ad', '_gercek', 'karar', 'dogru']].to_string(index=False))
    print('dogru siniflanan: %d / %d' % (df.dogru.sum(), len(df)))
    df.to_csv('deney1_kor_prob.csv', index=False)
