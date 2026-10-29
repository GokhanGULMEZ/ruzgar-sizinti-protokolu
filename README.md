> **v2 (manuscript submitted to Renewable Energy):** new experiments, result tables and the second wind farm (Kelmarsh) are described in [README_v2.md](README_v2.md); that file starts with an English summary.
> **v2 (Renewable Energy gönderimi):** yeni deneyler, sonuç tabloları ve ikinci çiftlik (Kelmarsh) için bkz. [README_v2.md](README_v2.md).

# Rüzgâr gücü tahmininde zamansal sızıntının ölçülmesi — kod paketi

Makale: *Ayrıştırma tabanlı rüzgâr gücü tahmininde zamansal (ileri-bakış) sızıntının nicelleştirilmesi*
(Gökhan Gülmez, Kadir Sarıkaya — Tokat Gaziosmanpaşa Üniversitesi). Veri: SDWPF (Zhou vd., *Sci Data* 11:649, 2024).

Bu paket makaledeki **tüm sayıları** üretir. Makaledeki hiçbir sayı elle yazılmamalı; ilgili betik çıktısından alınmalıdır.

## Ne var?

| Makale bölümü | Konu | Dosya |
|---|---|---|
| 3 Veri | yükleme, 30 dk ızgara, 37 öznitelik, kronolojik bölme | `leakage/data.py`, `scripts/00_veri_kontrol.py` |
| 4.1 Protokol | pencere (L), ufuk (H), purge, eğitimden ölçekleme | `leakage/data.py`, `leakage/config.py`, `scripts/01_L_secimi.py` |
| 4.2–4.4 Ayrıştırma | Full VMD (sızdıran), Rolling VMD (nedensel), LCSF (nedensel / çift yönlü) | `leakage/decomposition.py` |
| 4.5 Nedensellik denetimi | "geleceği boz" testi | `scripts/05_nedensellik_denetimi.py` |
| 4.6 Modeller | LSTM, CatBoost, OOF kalıntı modeli | `leakage/models.py` |
| 5 Değerlendirme | MAE, Skill, **LSI**, Diebold–Mariano (HLN), Holm | `leakage/evaluation.py` |
| 6.1, 6.5, 6.6 | T57 çok ufuk, DM+Holm, B9, ablasyon | `scripts/04_t57_cok_ufuk.py` |
| 6.2 Altı türbin | ana koşu | `scripts/02_turbin_kosusu.py` |
| 6.3 Tohum tekrarı | 3 tohum, ortalama ± std | `scripts/03_seed_tekrar.py` |
| 6.4 Maliyet | LCSF ve Rolling VMD gecikmesi | `scripts/06_maliyet_olcumu.py` |
| Tablo/Şekiller | kayıtlı sonuçlardan (tüm ölçütler dahil: RMSE, nMAE, nRMSE, R², Skill; makale Ek B bu dosyadan, `tablo_tum_metrikler.csv`) | `scripts/07_tablolar_sekiller.py` |
| Ek C | CatBoost öznitelik önemi, sızdıran vs nedensel bileşenler; CatBoost ailesinde LSI | `scripts/09_ek_c_oznitelik_onemi.py` |
| Doğrulama | paket ile özgün defter sonuçlarının eşleşmesi | `scripts/08_eslesme_kontrolu.py` |
| Kaynak kayıt | makale sonuçlarını üreten özgün Colab defterleri | `notebooks/orijinal_defterler/` |

Model kısaltmaları: **B1** Persistence · **B2** Raw LSTM · **B3** CatBoost Direct · **B6** Full VMD–LSTM (sızdıran) ·
**B7** Rolling VMD–LSTM (nedensel) · **B8** LCSF–LSTM (nedensel) · **B8L** LCSF çift yönlü (sızdıran) ·
**B9** B8 + OOF kalıntı CatBoost · **B8I** B8 + eğitim-içi kalıntı (ablasyon) · **B2O** Raw LSTM + OOF (ablasyon).
**LSI (%) = (E_nedensel − E_sızdıran) / E_nedensel × 100** (aynı model ailesinde).

## Kurulum

```bash
pip install -r requirements.txt
```
Test edilen sürümler: Python 3.11 (sentetik testler; numpy 2.4, pandas 3.0, scipy 1.17, catboost 1.2, tensorflow 2.21 CPU) ve Colab Python 3.13 (gerçek veri, T10 eşleşmesi).
Colab'da `catboost` ve `vmdpy` dışındaki paketler hazır gelir. Makale sonuçlarının üretildiği kesin paket sürümleri `requirements_kilit.txt` dosyasındadır (Colab'da `pip freeze` çıktısından, yalnızca paketin kullandığı kütüphaneler).

## Veri

SDWPF veri dosyası pakete dahil **değildir** (kendi lisansıyla dağıtılır). Tek CSV beklenir; sütunlar:
`TurbID, Tmstamp, Patv, Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, T2m, Sp, RelH, Wspd_w, Wdir_w, Tp`.
Yollar ortam değişkeniyle verilir (varsayılanlar Colab/Drive içindir):

```bash
export RT_VERI_CSV=/yol/sdwpf_2001_2112_full.csv
export RT_OUT=/yol/sonuclar/          # önbellek ve çıktılar; sonunda / olsun
```

## Çalıştırma sırası

| # | Komut | Süre (CPU) | Çıktı |
|---|---|---|---|
| – | `python tests/test_sentetik.py` | ~2 dk | testler geçmeli (sentetik veri) |
| 0 | `python scripts/00_veri_kontrol.py` | saniyeler | N=35.089 (731 gün × 48 + 1); 24.562/5.251/5.252; türbin seçim kuralı; zaman adımı teşhisi |
| 1 | `python scripts/01_L_secimi.py` | 10–20 dk | L=4 (2 saat) |
| 2 | `python scripts/02_turbin_kosusu.py 10 83 120 61 84` | ~70 dk / türbin | `adim6_T*.pkl`, `ozet_T*.csv` |
| 3 | `python scripts/03_seed_tekrar.py 10 83 120 61 84` | ~20–30 dk / türbin | `seed_T*.pkl` |
| 4 | `python scripts/04_t57_cok_ufuk.py` | birkaç saat | `adim2_T57_tahminler.pkl`, `adim5_dm_holm_T57.csv`, `ablation_T57.csv` |
| 5 | `python scripts/05_nedensellik_denetimi.py` | ~2 dk | 0,0 / 0,0 / 33,79 / 137,08 |
| 6 | `python scripts/06_maliyet_olcumu.py` | ~2 dk | ms / µs / oran |
| 7 | `python scripts/07_tablolar_sekiller.py` | saniyeler | tablolar ve şekiller |
| 8 | `python scripts/08_eslesme_kontrolu.py yedekle 10` → `02_turbin_kosusu.py 10` → `08_eslesme_kontrolu.py karsilastir 10` | 10–20 dk | özgün sonuçlarla eşleşme |
| 9 | `pip install shap` → `python scripts/09_ek_c_oznitelik_onemi.py 57` | 5–10 dk (Rolling VMD önbelleği varsa) | `ekc_ozet_T57.csv`, `ekc_onem_T57.csv` (permütasyon + SHAP), `sekil_ekc_T57.png`, `sekil_ekc_shap_*_T57.png` (SHAP beeswarm) |
| 10 | `python scripts/10_dm_kare_kayip.py` | ~1 dk (model eğitilmez) | `dm_kare_kayip_T57.csv`, `dm_persistence_T57.csv` (DM testinin kare kayıpla tekrarı) |
| 11 | `python scripts/11_nisan2020_duyarlilik.py 57` | 30–45 dk (18 LSTM) | `duyarlilik_nisan2020_T57.csv`, `duyarlilik_nisan2020_ozet_T57.csv` (Nisan 2020 boşluğu: 3 doldurma yöntemi x 3 tohum, LCSF) |
| 12 | `python scripts/12_pencere_duyarlilik.py 57` | ~3-4 sa (Rolling VMD 336 ve 1344 ön hesabı) | `duyarlilik_pencere_T57.csv` (Rolling VMD penceresi 7/14/28 gün: MAE, LSI-VMD, LSI-LCSF) |
| 13 | `python scripts/13_tablo_tutarlilik.py` | < 1 dk | `tutarlilik_raporu.csv` (makaledeki MAE/LSI değerlerinin ham sonuç dosyalarından yeniden hesaplanıp denetlenmesi) |
| 14 | `python scripts/14_ileri_bakis_egrisi.py 57` | ~1 sa (27 LSTM) | `ileri_bakis_T57.csv`, `ileri_bakis_ozet_T57.csv` (LCSF ileri-bakış doz–tepki: LSI, bileşenin kaç adım geleceğe baktığına göre) |
| 15 | `python scripts/15_bootstrap_guven_araligi.py` | < 1 dk | `lsi_bootstrap_aralik.csv` (altı türbin, LSI için blok bootstrap %95 aralığı) |
| 16 | `python scripts/16_maliyet_dogrulama.py 57` | 5-10 dk | (ekran çıktısı) Rolling VMD maliyetinin 200 konumda ölçümü ve paralel/ardışık karşılaştırma |
| 17 | `python scripts/17_vmd_ileri_bakis.py 57 1 2 8` | ~1 sa her m için (kayan VMD ön hesabı) + LSTM | `vmd_ileri_T57_m*.csv`, `vmd_ileri_ozet_T57_m*.csv` (kayan VMD için ileri-bakış doz–tepki: VMD sınır etkisini gelecek bilgisinden ayırma) |
| 18 | `python scripts/18_ek_tohumlar.py 57` | ~1,5–2 sa (≈20 LSTM) | `seed_ek_T57.pkl` + ekranda 8 tohumluk LSI özeti (ort., std, min–maks, %95 aralık) |
| 19 | `python scripts/19_vmd_geri_kontrol.py 57 1 2 4` | ~1,5 sa (kayan VMD ön hesabı + 15 LSTM) | `vmd_geri_T57.csv`, `vmd_geri_ozet_T57.csv` (geriye kaydırılmış nedensel bileşen: VMD sınır etkisi kontrolü) |
| 20 | `python scripts/20_vmd_uc_nokta_tani.py` | birkaç dk (LSTM eğitilmez) | VMD ileri-bakış eğrisinin monoton olmayışı için hızlı tanı (17. adım bileşenleri gerekir) |
| 21 | `python scripts/21_cok_ufuk_alti_turbin.py` | birkaç saat | diğer beş türbinde 3 ve 6 saatlik ufuk (tek tohum) |
| 22 | `python scripts/22_vmd_dogrusal_tani.py 57` | dakikalar | tohumsuz Ridge tanısı |
| 23 | `RT_ROLL=1 python scripts/23_cozunurluk_10dk.py 57` | birkaç saat (kayan VMD önbellekli) | 30 dk ve yerel 10 dk ızgara karşılaştırması |
| 24 | `python scripts/24_dm_tohum_bazinda.py` | ~1 sa (model eğitilmez) | `dm_tohum_bazinda.csv`, `dm_tohum_ozet.csv` (6 türbin x 8 tohum DM+HLN+Holm) |
| 25 | `python scripts/25_era5_interpolasyon_kontrol.py [10dk]` | dakikalar | ERA5 doldurma yönteminin etkisi (yalnız ölçüm) |
| 26 | `python scripts/26_mimari_duyarlilik.py gru 57` ve `... tcn 57` | ~1–3 sa her mimari (15 eğitim) | `mimari_gru_T57.pkl`, `mimari_tcn_T57.pkl` + ekranda LSI ve DM özeti (aynı protokol GRU / nedensel TCN ile) |
| 27 | `python scripts/27_kayip_duyarlilik.py mae 57` (veya `huber 57`) | ~30–60 dk (15 eğitim) | `kayip_mae_T57.pkl` + ekranda MAE, LSI ve DM özeti (aynı LSTM, MSE yerine MAE kaybıyla) |
| 28 | `python scripts/28_referans_modeller.py` | dakikalar (eğitim yok) | `referans_modeller.csv`: persistence, iklimsel ortalama ve Nielsen vd. (1998) referansının MAE/RMSE değerleri (altı türbin, altı ufuk) ve eldeki tahminlerin Nielsen'e göre skill'i |
| 29 | `python scripts/29_nedensellik_sertifikasi.py` | ~1–2 sa (çoğu tam seri VMD) | `sertifika_profil_T57.csv`, `sertifika_onek_T57.csv`: gelecek-pertürbasyon profili ve önek değişmezliği testi (dört işleç) |
| 30 | `python scripts/30_filtre_sayisal_kontrol.py 57` | dakikalar | LCSF Butterworth (b,a) biçimi ile SOS biçiminin bileşen çıktılarının karşılaştırması |

Rolling VMD en pahalı adımdır (tek türbinde 35 bin VMD çağrısı); bloklar `RT_OUT` içine önbelleklenir ve
yarıda kalırsa aynı komut kaldığı yerden devam eder. Colab için hazır defter: `notebooks/Colab_calistir.ipynb`.

## Doğrulama (bu paketin orijinal Colab koduyla aynı olduğu nasıl biliniyor?)

Paket, makale sonuçlarını üreten Colab defterlerinden (`notebooks/orijinal_defterler/`) yeniden düzenlenmiştir
(global değişkenler `TurbineData` sınıfına taşındı; hesaplar aynı). İki düzeyde doğrulandı:

1. **Sentetik veri:** orijinal defter kodu ile paket kodu karşılaştırıldı; LCSF bantları ve çıktıları, pencereler, ölçekleme,
   Raw LSTM, CatBoost ve OOF-stack tahminleri bit düzeyinde aynı (maksimum fark 0,0).
2. **Gerçek veri (SDWPF, Türbin 10, 5 Ekim 2026, Colab, Python 3.13):** `02_turbin_kosusu.py 10` çıktısı, özgün
   `adim6_T10.pkl` ile B1, B2, B3, B6, B7, B8, B8L ve B9 için karşılaştırıldı: test kümeleri aynı, MAE farkı %0,00,
   maksimum tahmin farkı 0,000 (`scripts/08_eslesme_kontrolu.py`).

## Veri notları (makale Bölüm 3 ile tutarlı)

- Ham CSV zamana göre **sıralı değildir**; zaman adımı farkları sıralamadan alınırsa yanlış sonuç (30 dk) verir. `load_turbine` ve `00_veri_kontrol.py` sıralar.
- Dosya sürümünde 2020 kayıtları 15 dk, 2021 kayıtları 10 dk aralıklıdır (T57: 32.224 ve 52.559 kayıt) ve Nisan 2020'de ~30 günlük boşluk vardır; bu yüzden :10 ve :40 anları alınarak 30 dk ortak ızgara kurulur.
- Türbin seçimi: ham Patv eksik oranına göre en düşük, %25, %50, %75, en yüksek + T57 → T10, T57, T61, T83, T84, T120.
- Anma gücü 1500 kW (SL1500/82); en yüksek Patv T57/61/84/120'de ≈1521, T10/83'te ≈1561 kW (kırpılmadı).

## Bilinen sınırlar

- Tek veri seti (SDWPF), tek rüzgâr çiftliği; test dönemi 13 Eylül–31 Aralık 2021 (sonbahar–kış).
- Nielsen referansı MSE bakımından eniyidir; MAE'de persistence'tan kötü olabilir, bu yüzden her iki ölçütle raporlanmalıdır (betik 28 ikisini de verir).
- `certificate.py` sertifikası, denenen kesim anları, gecikmeler ve büyüklükler için "bağımlılık saptanmadı" der; nedenselliğin kanıtı değildir.
- `evaluation.lambda_orani`: λ [0,1] aralığında kalmayabilir; oracle'a duyarlıdır (makale Bölüm 6.14).
- Ana deneylerde tek LSTM mimarisi; GRU ve nedensel TCN ile küçük bir tekrar için `scripts/26_mimari_duyarlilik.py`.
- Tohum tekrarı (42–49) yalnızca 1 saatlik ufukta ve LSTM tabanlı modellerde yapılmıştır; 3 ve 6 saatlik ufuklar altı türbinde tek tohumludur.
- Maliyet ölçümü makineye bağlıdır (Python, tek çekirdek); donanım makalede belirtilmiştir.
- Nedensel modellerin hata farkları tohum gürültüsü içinde ayırt edilemez; bu bir bulgu olarak raporlanır, "yeni en iyi model" iddiası yoktur.

## Lisans / atıf

Kod MIT lisansı altındadır (`LICENSE`). Atıf için `CITATION.cff` dosyasına bakınız; Zenodo DOI is added to the manuscript and to this file when the archived release is created / Zenodo DOI, arşivlenmiş sürüm oluşturulduğunda makaleye ve bu dosyaya eklenir.


## Nisan 2020 boşluğu duyarlılık sonucu

`scripts/11_nisan2020_duyarlilik.py` (Türbin 57, 1 saat, yalnızca LCSF, tohum 42/43/44) uzun eksik veri boşluklarının (48 adımdan uzun Patv NaN dizisi) üç yöntemle ele alınmasını karşılaştırır: A ileri doldurma (ana deneyler), B eğitim ortalaması, C bantların boşluksuz eğitim kesitinden öğrenilmesi.

| Yöntem | LSI % (42/43/44) | LSI % ort. ± std | MAE nedensel | MAE sızdıran |
|---|---|---|---|---|
| A | 50,7 / 51,6 / 37,8 | 46,7 ± 7,7 | 115,94 | 61,73 |
| B | 50,8 / 51,0 / 37,4 | 46,4 ± 7,8 | 115,91 | 62,05 |
| C | 50,3 / 51,9 / 37,7 | 46,7 ± 7,8 | 115,87 | 61,74 |

Yöntemler arası LSI farkı (ortalamada en çok 0,3 puan; aynı tohumda en çok 0,9 puan) tohum değişkenliğinin (std ≈ 7,7 puan) çok altındadır; yani boşluğun doldurulma biçiminin sızıntı ölçümünü tohum değişkenliğinin ötesinde etkilediği gözlenmemiştir. Kapsam: tek türbin, üç tohum, yalnızca LCSF; kayan ve tam seri VMD yeniden hesaplanmamıştır.

## Pencere uzunluğu duyarlılık sonucu

`scripts/12_pencere_duyarlilik.py` (Türbin 57, 1 saat, tohum 42): kayan VMD penceresi 7 / 14 / 28 gün.

| W (adım) | Rolling VMD nedensel | Full VMD sızdıran | LCSF nedensel | LCSF çift yön | LSI-VMD % | LSI-LCSF % |
|---|---|---|---|---|---|---|
| 336 | 119,85 | 68,90 | 120,17 | 57,67 | 42,5 | 52,0 |
| 672 | 116,11 | 64,71 | 116,29 | 57,34 | 44,3 | 50,7 |
| 1344 | 116,60 | 63,94 | 117,06 | 59,24 | 45,2 | 49,4 |

Her pencerede sızıntı kayda değerdir (LSI-VMD %42,5–45,2). LSI değerleri yaklaşık 2,7 puan aralığında kayar; bu, türbinler arası tohum std ile aynı büyüklüktedir. Tek türbin ve tek tohum; yön kesin kabul edilmemelidir. W=672 satırı ana koşuyla birebir örtüşür.

## İleri-bakış doz–tepki sonucu

`scripts/14_ileri_bakis_egrisi.py` (Türbin 57, 1 saat, LCSF, tohum 42/43/44): bileşen her t anında yalnız x[..t+m] üzerinde sıfır fazlı süzülür.

| m (adım) | MAE (kW) | LSI-LCSF % (ort ± std) |
|---|---|---|
| nedensel (tek yönlü) | 115,94 | — |
| 0 (sıfır fazlı, gelecek yok) | 113,27 | 2,3 ± 1,4 |
| 1 | 85,86 | 26,0 ± 1,8 |
| 2 (= H, hedef bileşende) | 11,05 | 90,5 ± 1,0 |
| 4 | 29,10 | 74,9 ± 0,6 |
| 8 | 59,43 | 48,7 ± 6,3 |
| 16 | 63,50 | 45,2 ± 2,7 |
| 48 | 61,60 | 46,8 ± 5,1 |
| ∞ (tam seri) | 61,73 | 46,7 ± 7,7 |

Sıfır fazlı ama geleceğe bakmayan süzme LSI'yi yalnızca ~%2 yapar; sızıntı bileşenin geleceğe bakmasıyla ortaya çıkar ve m≥8'de ~%45–49 platosuna oturur (tam seri ile aynı düzey). m=2 ve 4 mekanizmanın üst sınırıdır (hedef bileşende). Tek türbin, tek ufuk, yalnız LCSF.

## Kayıtlar
`notebooks/dogrulama_kayitlari/` paketin gerçek veride çalıştırıldığını gösteren Colab kaydını (maliyet ölçümü ve T10 eşleşme kontrolü) içerir.
