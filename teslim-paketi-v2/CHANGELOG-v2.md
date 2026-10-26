# CHANGELOG — v2 (Renewable Energy sürümü)

**Tarih:** 2026-10-26 · **Hedef dergi:** Renewable Energy (Elsevier, `elsarticle`)
**Önceki:** v1 (2026-10-08, dergi-bağımsız yapısal sürüm)
**Girdi:** Gökhan'ın `Manuscript-EN-draft1_1_v2.docx` (19.671 kelime, 33 tablo, 50 kaynak)

---

## Dosyalar

```
teslim-paketi-v2/
├── CHANGELOG-v2.md                       bu dosya
└── manuscript/
    ├── en/RE_manuscript_v2.tex           ESAS — elsarticle, Renewable Energy
    └── tr/RE_makale_v2.tex               Türkçe paralel (Gökhan için)
```

---

## 1. Dergi biçimi

- `\documentclass[review,3p,times]{elsarticle}` — Elsevier inceleme formu
  (çift satır aralığı + satır numarası).
- `\begin{highlights}` — **5 madde, her biri 85 karakter sınırının altında**
  (karakter sayıları kaynak dosyada yorum olarak işaretli). Gönderim sistemi
  sınırı zorunlu kılıyor; orada tekrar kontrol edin.
- Özet 248 kelime.
- Elsevier'in istediği beyanlar ayrı başlıklar olarak kuruldu: CRediT, çıkar
  çatışması, üretken yapay zekâ, veri erişilebilirliği.
- ⬜ **Doğrulanmadı:** derginin güncel yazar kılavuzundaki özet kelime sınırı ve
  üretken-YZ beyanının öngörülen tam ifadesi. Kılavuzdan teyit edilmeli.

## 2. Çerçeve değişikliği — maliyet ekseni

v1'de makale "ölçme çalışması" olarak konumlanmıştı. v2'de **ölçme + çözüm**:

| | v1 | v2 |
|---|---|---|
| Başlık | *Measuring the Evaluation Protocol…* | *Leakage-free short-term wind power forecasting at turbine level: a low-latency causal filter bank, a training-free causality certificate, and the cost of honest evaluation* |
| Açılış | sızıntı büyüklüğü | **yedek boyutlandırma** → dürüst protokolün bedeli |
| LCSF | kontrol operatörü | **1. katkı** — dürüst protokolü uygulanabilir kılıyor |
| Katkı sırası | λ → sertifika → faktöriyel | LCSF/maliyet → sertifika → λ → türbin düzeyi ölçüm |

**Gerekçe.** Jeon vd. (2026), kendi kontrol listelerinin 1. maddesi için kayan
yeniden-ayrıştırmanın *"gerçek bir hesaplama maliyeti"* taşıdığını ve bunun
*"konuşlandırmayı yansıtan bir değerlendirmenin bedeli"* olduğunu yazıp çözüm
sunmuyor. Ölçülmüş çözüm bizde: 69 µs'ye karşı 89,9 ms. §1.2'de bu boşluk
açıkça adlandırıldı.

Renewable Energy kapsamı da bunu gerektiriyor: dergi yalnızca **güç üretimiyle
ilgili** makaleleri kapsıyor. §1.1 bu yüzden yedek tedariki, kısıtlama ve gün
içi ticaretle açılıyor.

## 3. Jeon vd. karşısında konumlandırma

§1.3 ve §2.2 yeniden yazıldı. Ölçümün kendisi **artık yeni olarak sunulmuyor**;
üç açık boşluk adlandırılıyor:

1. Çözümün maliyeti adlandırılmış ama ele alınmamış → §4.1, §6.5
2. Operatörün nedenselliği için uygulanabilir test yok → §5.4, §6.3
3. Büyüklük yalnız ham fark olarak raporlanmış → §5.3, §6.4

Ayrıca kurulum farkı: öncül ölçümlerin tamamı saatlik toplam/ulusal serilerde;
bu çalışma 10--30 dk türbin düzeyi SCADA'da.

## 4. Eklenen kaynaklar (bu oturumda doğrulandı)

| Kaynak | Tür | Nerede |
|---|---|---|
| **Jeon vd. (2026)**, *Int. J. Energy Res.*, 10.1155/er/4433079 | Rakip — en yakın | §1.3, §2.2, §7 |
| **Zhang vd. (2025)**, 10.1117/12.3045386 | Rakip — mod düzeyi atama | §1.3, §2.2 |
| **Zhou vd. (2024)**, *Energy Sources A*, 10.1080/15567036.2024.2318485 | Rakip — PMAE %40–55 | §1.3, §2.2 |
| **Valdivieso vd. (2026)**, SSRN | Rakip (preprint, işaretli) | §1.3 |
| **Wang J. vd. (2024)**, *Energy* 312 | Rakip | §1.3, §2.3 |
| **Lee, Draxl & Berg (2022)**, *Renew. Energy* 200:457–475 | Destek — **tür öncülü** | §2.4 |
| **González Sopeña vd. (2020)** *RSER* 110515; **(2022)** *SETA* | Destek + rakip | §2.4 |
| **Hong vd. (2020)** | Destek — otorite alıntısı | §2.4 |
| **Deignan vd. (2025)**, *Sci Rep* | Destek — K≈4, Kelmarsh/Penmanshiel | §2.4 |
| **Nielsen vd. (1998)**; **Madsen vd. (2005)**; **Messner vd. (2020)** | Referans/ölçüt | §5.2 |
| **Jia vd.** STVMD; **Dan vd.** RSVMD; **Guo vd.** GVMD | Nedensel ayrıştırma rakipleri | §2.5, §4.1 |
| **Künsch (1989)** | Blok bootstrap | §5.1 |

`[VERIFY]` işaretli alanlar (cilt, sayfa, DOI) gönderimden önce tamamlanacak.

## 5. Tablolar

Ana metin **10 gösterime** indirildi; geri kalan 23 tablo eklere taşındı.
Ana metindeki tablolar: çok ufuklu MAE+LSI; altı türbin (v2'nin Tablo 4+5'i
tek tabloda birleştirildi); 8 tohum LSI; doz–tepki; sertifika; λ; gecikme;
referans modeller; kayıp duyarlılığı.

**Tüm sayılar `.docx`'ten programatik olarak çıkarıldı, elle yeniden
yazılmadı.** Çıkarma betiği: `tools/extract.py` (oturum çalışma alanında).

## 6. Açık kalemler

| # | Kalem | Durum |
|---|---|---|
| 1 | **Depo özel** — makaledeki URL okur için 404 | **Gönderim engeli** |
| 2 | Depoda kayıtlı sonuç tablosu yok, yalnız kod | Yüksek |
| 3 | Zenodo DOI yer tutucusu | Yüksek |
| 4 | §6.14–6.18 sayıları `verify_claims` tarzı bir denetimle koda bağlanmadı (λ hariç — `tests/test_sentetik.py` onu sabitliyor) | Yüksek |
| 5 | Kelmarsh ikinci çiftlik | Hakem isteyecek |
| 6 | STVMD + RSVMD baseline — LCSF katkıysa zorunlu | Hakem isteyecek |
| 7 | Rolling-origin ya da aylık alt-dönem | Orta |
| 8 | λ'nın kalan hücreleri | Orta |
| 9 | Yazar sırası (makale / `CITATION.cff` / Zenodo çelişiyor) | Orta |
| 10 | `32_dm_hac_duyarlilik.py` sonuçları taslakta görünmüyor | Kontrol |
| 11 | SDWPF örnekleme aralığı figshare orijinaliyle doğrulanmadı | Orta |
| 12 | §3, §4, §6, §7, §8'in `[TAŞI]` metinleri `.docx`'ten aktarılacak | Yazım |

## 7. Kod paketi

`main` = `810cbdf` (19 Ekim) arşivi v1 paketinin `code/` klasörüne eklendi:
`ruzgar-sizinti-protokolu-main-810cbdf.zip`. Taslak v2'nin §6.14–6.18
sayılarını üreten betikler bu sürümde:
`27_kayip_duyarlilik.py`, `28_referans_modeller.py`,
`29_nedensellik_sertifikasi.py`, `30_filtre_sayisal_kontrol.py`,
`31_esit_egitim_ham.py`, `32_dm_hac_duyarlilik.py`,
`leakage/reference_models.py`, `leakage/evaluation.py::lambda_orani`.

Etiket `v1.0-gonderim` = `cd1443f` (5 Ekim); v1 paketindeki eski zip ayrı bir
geçmişten (`61247ff`) geldiği için SHA'sı depoyla çapraz doğrulanamıyor, arşiv
olarak duruyor.
