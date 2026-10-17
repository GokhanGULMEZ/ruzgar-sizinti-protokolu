"""ADIM 29 — Nedensellik sertifikası (Türbin 57): gelecek-pertürbasyon profili ve önek değişmezliği testi.
İşleçler: nedensel LCSF, kayan VMD (nedensel); iki yönlü LCSF, tam seri VMD (sızdıran). Eğitim GEREKTİRMEZ.
Profil : t0 anından SONRAKİ gözlemlere N(0, delta) gürültü eklenir; t0-tau anındaki bileşenin değişimi (kW) ölçülür.
         Nedensel işleç için tüm tau >= 0'da tam 0 beklenir. 'Erişim' = fark > 0 olan en büyük tau (izgara sınırına eşitse erişim en az o kadardır).
Önek   : Phi(x[:t+1])_t == Phi(x)_t ? rastgele kesim noktalarında.
Kullanım: python 29_nedensellik_sertifikasi.py [t0_sayisi=20] [t0_sayisi_tam_vmd=5] [kesim_sayisi=50] [kesim_sayisi_tam_vmd=10] [turbin=57]
Süre   : tam seri VMD her çağrıda ~1 dk: varsayılan ayarlarla toplam ~1-2 saat; diğer işleçler dakikalar.
Çıktı  : sertifika_profil_T{tid}.csv, sertifika_onek_T{tid}.csv. Sonuç, kullanılan t0 sayısı, gecikme ve büyüklük kümesiyle
         birlikte raporlanmalıdır (sertifika gerekli koşuldur, kanıt değil)."""
from _ortak import *
import sys, numpy as np, pandas as pd
from leakage.config import W
from leakage.data import TurbineData
from leakage.decomposition import LCSF, last_mode, full_vmd
from leakage.certificate import Operator, pertürbasyon_profili, erisim, onek_degismezligi

a = sys.argv[1:]
n_t0 = int(a[0]) if len(a) > 0 else 20; n_t0_vmd = int(a[1]) if len(a) > 1 else 5
n_kes = int(a[2]) if len(a) > 2 else 50; n_kes_vmd = int(a[3]) if len(a) > 3 else 10
tid = int(a[4]) if len(a) > 4 else 57
TAULAR = [0, 1, 2, 4, 8, 16, 32]; DELTALAR = [50, 100, 200, 400]     # adım ; kW
df = veri_yukle(); out = cikti_klasoru(); td = TurbineData(df, tid)
x, N = td.x_dec, td.N
lc = LCSF().fit(x[:len(td.tr)])
OPS = [
    (Operator('LCSF nedensel', at=lambda s, t: lc.transform(s, True)[t]), n_t0, n_kes),
    (Operator('Kayan VMD (nedensel)', at=lambda s, t: last_mode(s[t - W + 1:t + 1])), n_t0, n_kes),
    (Operator('LCSF iki yönlü (sızdıran)', at=lambda s, t: lc.transform(s, False)[t]), n_t0, n_kes),
    (Operator('Tam seri VMD (sızdıran)', full=lambda s: full_vmd(s, len(s))), n_t0_vmd, n_kes_vmd),
]
rng = np.random.default_rng(7)
profil_satir, onek_satir = [], []
for op, nt0, nk in OPS:
    t0s = sorted(rng.integers(W + max(TAULAR) + 1, N - 2, nt0).tolist())
    S = pertürbasyon_profili(op, x, t0s, DELTALAR, TAULAR, tohum=11)
    for i, dl in enumerate(DELTALAR):
        profil_satir.append(dict(islec=op.ad, delta_kW=dl, t0_sayisi=nt0, erisim_tau=erisim(S[i], TAULAR), erisim_izgara_siniri=bool(erisim(S[i], TAULAR) == max(TAULAR)),
                                 **{f'tau={t}': round(float(S[i, j]), 4) for j, t in enumerate(TAULAR)}))
    kes = sorted((rng.integers(W + 1, N - 2, nk) | 1).tolist())     # tek t -> önek uzunluğu çift (vmdpy tek uzunlukta son örneği keser)
    m, ihlal, n = onek_degismezligi(op, x, kes)
    onek_satir.append(dict(islec=op.ad, kesim_sayisi=n, en_buyuk_fark=m, toleransi_asan=ihlal, tolerans=0.0, dtype=str(x.dtype)))
    print(f'{op.ad:28s} | profil tamam | önek: en büyük fark={m:.3g}, ihlal={ihlal}/{n}', flush=True)
pf = pd.DataFrame(profil_satir); pf.to_csv(out + f'sertifika_profil_T{tid}.csv', index=False)
pd.DataFrame(onek_satir).to_csv(out + f'sertifika_onek_T{tid}.csv', index=False)
print(pf.to_string(index=False))
