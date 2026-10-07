"""ADIM 10 — Diebold–Mariano testinin KARE KAYIPLA tekrarı (MAE'ye bağımlı olmayan sağlamlık kontrolü).
Girdi : adim2_T57_tahminler.pkl (OUT klasöründe; model eğitilmez, ~1 dakika).
Çıktı : dm_kare_kayip_T57.csv (30 karşılaştırma, Holm 30 üzerinde; mutlak ve kare kayıp yan yana)
        dm_persistence_T57.csv (nedensel modeller vs persistence; her iki kayıp; Holm kendi ailesinde)
Aile 1 = Tablo 8 / Ek A ile aynı 30 karşılaştırma. Aile 2 = ek: nedensel modellerin persistence'a karşı sınanması."""
from _ortak import *
import pickle, numpy as np, pandas as pd
from leakage.evaluation import dm_test, holm

OUT = cikti_klasoru()
res = pickle.load(open(OUT + 'adim2_T57_tahminler.pkl', 'rb'))
HS = [1, 2, 4, 6, 8, 12]

def kos(pairs, aile):
    rows = []
    for m1, m2, ad in pairs:
        for H in HS:
            if (m1, H) in res and (m2, H) in res:
                r1, r2 = res[(m1, H)], res[(m2, H)]
                assert np.array_equal(r1['t'], r2['t']); y = r1['y']
                a = dm_test(y, r1['p'], r2['p'], h=H, loss='abs'); s = dm_test(y, r1['p'], r2['p'], h=H, loss='sq')
                rows.append(dict(karsilastirma=ad, H_saat=H / 2,
                                 MAE_1=np.mean(np.abs(y - r1['p'])), MAE_2=np.mean(np.abs(y - r2['p'])),
                                 RMSE_1=np.sqrt(np.mean((y - r1['p']) ** 2)), RMSE_2=np.sqrt(np.mean((y - r2['p']) ** 2)),
                                 DM_abs=a['DM'], p_abs=a['p_raw'], DM_sq=s['DM'], p_sq=s['p_raw']))
    d = pd.DataFrame(rows)
    d['holm_abs'] = holm(d['p_abs'].values); d['holm_sq'] = holm(d['p_sq'].values)
    d['anlamli_abs'] = d['holm_abs'] < 0.05; d['anlamli_sq'] = d['holm_sq'] < 0.05
    d['yon_abs'] = np.sign(d['DM_abs']); d['yon_sq'] = np.sign(d['DM_sq'])
    d['sonuc_ayni'] = (d['anlamli_abs'] == d['anlamli_sq']) & ((~d['anlamli_abs']) | (d['yon_abs'] == d['yon_sq']))
    print(f'\n== {aile}: {len(d)} karşılaştırma | sonuç (anlamlılık+yön) her iki kayıpta aynı: {int(d.sonuc_ayni.sum())}/{len(d)}')
    return d

P1 = [('B6', 'B7', 'Sızıntı: Full VMD vs Rolling VMD'), ('B8L', 'B8', 'Sızıntı: LCSF çift yönlü vs nedensel'),
      ('B9', 'B1', 'Önerilen vs Persistence'), ('B9', 'B8', 'OOF katkısı (B9 vs B8)'),
      ('B9', 'B3', 'B9 vs CatBoost Direct'), ('B8', 'B7', 'LCSF vs Rolling VMD (nedensel)'),
      ('B7', 'B2', 'Rolling VMD vs Raw LSTM'), ('B8', 'B2', 'LCSF vs Raw LSTM')]
P2 = [(k, 'B1', ad + ' vs Persistence') for k, ad in
      (('B2', 'Raw LSTM'), ('B3', 'CatBoost Direct'), ('B7', 'Rolling VMD'), ('B8', 'LCSF (nedensel)'), ('B9', 'B9'))]

d1 = kos(P1, 'Aile 1 (Tablo 8 ile aynı)'); d2 = kos(P2, 'Aile 2 (nedensel modeller vs persistence)')
pd.set_option('display.width', 250)
c = ['karsilastirma', 'H_saat', 'DM_abs', 'holm_abs', 'DM_sq', 'holm_sq', 'sonuc_ayni']
print(d1[c].round(4).to_string(index=False))
print('\nSonucu DEĞİŞENLER (aile 1):'); print(d1[~d1.sonuc_ayni][c].round(4).to_string(index=False) or '  yok')
print('\nAile 2:'); print(d2[c].round(4).to_string(index=False))
d1.to_csv(OUT + 'dm_kare_kayip_T57.csv', index=False); d2.to_csv(OUT + 'dm_persistence_T57.csv', index=False)
print('\nyazıldı:', OUT)
