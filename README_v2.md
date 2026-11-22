# v2 ek paketi — makale taslağı v2 (Renewable Energy) için

**English summary.** This folder adds the code behind the v2 manuscript ("low-latency causal filter bank, training-free causality probe,
low-latency causal filter bank") to the v1 package. It does **not** modify any v1 file (`leakage/`, `scripts/`, `tests/`). Every number in v2 is produced
by a v1 script (`scripts/00`–`32`) or by one of the scripts below; the table at the end maps each manuscript item to its script.

Bu klasör, v1 paketine (`ruzgar_sizinti_paketi`, etiket `v1.0-gonderim`) **eklenir**; v1 dosyalarının hiçbirini değiştirmez.
`leakage/` paketi aynen kullanılır; Kelmarsh betiği sabitleri (anma gücü, ölçüm dakikaları) çalışma anında kendi içinde değiştirir.

## Yerleşim

```
README_v2.md                  bu dosya
ek/                           v1 sonuçlarını OKUYAN ek analizler (eğitim yok, ~1 dk)
  01_dm_3sa_6sa/              diğer 5 türbinde 3 ve 6 sa: DM + HLN + Holm, LSI aralığı
  02_birim_bagimlilik/        türbin = birim (n=6), ortak-zaman blok bootstrap
  08_esdegerlik/              LCSF ile rolling VMD farkı: güven aralığı ve eşdeğerlik (TOST), ~1 dk
  04_istatistik_protokol/     DM varyansı, HLN, Holm aileleri, bağımlılık: Ek metni
  ortak/                      ortak yardımcılar
scripts_v2/                   yeni deneyler
  sertifika_genis_tau.py      nedensellik sondası, gecikme ızgarası 2048 adıma kadar (Şekil 3, Tablo S4.2)
  ileri_erisim_analitik.py    iki yönlü LCSF'nin ileri erişimi, birim darbe cevabından (veri gerekmez, Tablo S4.1)
  ek04_stvmd_kiyas.py         sıcak başlatmalı hızlı nedensel VMD: süre ve doğruluk (Tablo S5.5)
  deney1_kor_prob.py          kodlanmış sekiz işleçte körleme sonda testi (Tablo S4.3)
  deney3_lambda_turbinler.py  oracle-normalleştirilmiş oran, altı türbin (Tablo S5.13, keşfedici)
  deney4_ucuz_bankalar.py     VMD'siz iki nedensel filtre bankası (Tablo S5.10)
  deney5_placebo_gelecek.py   placebo-gelecek kontrolü, pencere geometrisi sabit, gelecek bilgisi yok (Tablo S5.12; leakage/placebo.py kullanır)
  ek05_rolling_origin.py      rolling-origin: seri 70/80/90/100% kesilir (Tablo E.6)
  ek06_ornekleme_kontrol.py   SDWPF örnekleme aralığı kontrolü (bizim dosya / figshare orijinali)
  ek07_kelmarsh.py            ikinci çiftlik: Kelmarsh
  sekil_uret.py               ana metin Şekil 1–3'ü (tablolardaki sayılardan)
sonuclar_v2/                  özet CSV'ler (bkz. OKU_BENI.md)
sekiller_v2/                  ek SHAP şekilleri: Full-series VMD (B6) ve rolling VMD (B7), T57, 1 sa. Makaledeki Şekil C.1–C.2 (LCSF, B8/B8L) v1 `scripts/09` ile üretilir
```

## Kurulum ve ortam

v1'in `requirements.txt` dosyası geçerlidir; ek olarak `pip install vmdpy catboost` (Colab'da diğerleri hazır). Yollar ortam değişkeniyle verilir:

```python
import os
os.environ['RT_PKG']        = '/content/drive/MyDrive/RuzgarTahmin/ruzgar_sizinti_paketi'   # 'leakage' klasörünün olduğu dizin
os.environ['RT_OUT']        = '/content/drive/MyDrive/RuzgarTahmin/yeni/'                   # v1 sonuçları: adim6_T*.pkl, seed_T*.pkl, vmd_roll_T57.npy ...
os.environ['RT_EK_OUT']     = '/content/drive/MyDrive/RuzgarTahmin/ek_cikti/'               # bu paketin çıktıları
os.environ['RT_VERI_CSV']   = '/content/drive/MyDrive/RuzgarTahmin/veri/sdwpf_2001_2112_full.csv'
os.environ['RT_KEL_DIR']    = '/content/drive/MyDrive/RuzgarTahmin/veri/kelmarsh'           # yalnız ek07: açılmamış Kelmarsh zip'leri
```

## Yeni betiklerin çalıştırılması

| Betik | Girdi | Süre | Çıktı |
|---|---|---|---|
| `python scripts_v2/ileri_erisim_analitik.py` | yok (analitik) | saniyeler | `ileri_erisim_analitik.csv` |
| `python scripts_v2/sertifika_genis_tau.py` | veri, `leakage` | LCSF saniyeler; tam seri VMD ~1 sa | `sertifika_genis_T57.csv` |
| `python scripts_v2/deney5_placebo_gelecek.py 57 lcsf 2` | veri, `vmd_roll_T57.npy`, `vmd_full_T57.npy` (v1 betik 02) | LCSF: eğitim süresi belirler, m=2 için 3 seed ~45 dk (GPU) | `placebo_lcsf_m2_T57.csv`, `placebo_ozet_lcsf_m2_T57.csv` |
| `python scripts_v2/deney5_placebo_gelecek.py 57 vmd 2` | yukarıdakiler ve `vmd_ileri_T57_m2.npy` (v1 betik 17) | kayan VMD ile aynı mertebe, bloklar diske yazılır | `placebo_vmd_m2_T57.csv`, `placebo_ozet_vmd_m2_T57.csv` |
| `python scripts_v2/ek04_stvmd_kiyas.py` | `vmd_roll_T57.npy`, `adim6_T57.pkl`, `seed_T57.pkl` | ~15 dk (CPU) | `ek04_zaman_bilesen.csv`, `ek04_tahmin.csv` |
| `python scripts_v2/ek05_rolling_origin.py` | `vmd_roll_T57.npy`, veri | ~1 sa (CPU) | `ek05_rolling_origin.csv` |
| `python scripts_v2/ek06_ornekleme_kontrol.py <csv> [türbin]` | bizim CSV ve figshare orijinali | saniyeler | konsol |
| `python scripts_v2/ek07_kelmarsh.py` | Kelmarsh SCADA 2020, 2021 zip'leri | türbin başına ~1 sa (rolling VMD) | `ek07_kelmarsh.csv` |
| `python scripts_v2/sekil_uret.py` | yok | saniyeler | `sekiller_v2/fig_*.png` |
| `python ek/01_dm_3sa_6sa/ek01_dm_3sa_6sa.py` | `cok_ufuk_T*.pkl` (v1 betik 21) | ~1 dk | `ek01_dm_3sa_6sa.csv` |
| `python ek/02_birim_bagimlilik/ek02_birim_ve_bagimlilik.py` | `adim6`, `seed`, `seed_ek` pkl (v1 betik 02, 03, 18) | ~1 dk | `ek02_*.csv` |
| `python ek/08_esdegerlik/ek08_esdegerlik.py` | `adim6`, `seed`, `seed_ek` pkl | ~1 dk | `ek08_esdegerlik.csv` |

## Dürüstlük notları (makaleyle aynı)

- `ek04`: **bu bir STVMD veya RSVMD uygulaması değildir.** Rolling VMD'nin her pencereyi önceki pencerenin merkez frekanslarından başlatıp iterasyonu
  sınırlayan hâlidir. 500 iterasyonda `vmdpy` ile 3×10⁻¹¹ kW farkla aynıdır (betik başında doğrulanır). Bileşenleri rolling VMD'den farklıdır.
- `ek05`: yalnız T57, 1 sa, 2 seed. Sızdıran Full VMD her kesim için o kesimin serisinde yeniden hesaplanır.
- `ek07`: Kelmarsh'ta ERA5 sütunları yoktur; altı ERA5 özniteliği sabit 0'dır (37 öznitelikten 31'i bilgili). Anma gücü 2050 kW. 30 dk ızgara :00/:30 anlarıdır.
  Veri: Kelmarsh (Cubico Sustainable Investments), Zenodo kaydı 16807551, CC-BY-4.0; kullanırken kaydı ve lisansı anın.
- `deney5_placebo_gelecek.py`: yalnız T57, m=2, üç seed; LCSF için ayna ve rastgele-artım kuyrukları, VMD için yalnız ayna kuyruğu. VMD'de rastgele-artım kuyruğu ve m=8 koşulmadı. `tests/test_placebo.py` sahte kuyrukların nedensel olduğunu doğrular.
- `sertifika_genis_tau.py`: tam seri VMD değerleri 6 pertürbasyon döneminden gelir, gösterge niteliğindedir.

## Makale öğesi → betik

Numaralar makalenin ana metnine (Tablo 1–6, Şekil 1–3) ve Ek'e (S1–S8) aittir. Ek, ayrıntılı eşlemeyi S8'de de verir.

| Makale öğesi | Betik | Not |
|---|---|---|
| Veri, bölme, L seçimi | v1 `scripts/00`, `01` | |
| Şekil 1, Tablo S5.1 | v1 `04`; şekil: `scripts_v2/sekil_uret.py` | T57, 0,5–6 sa, seed 42 |
| Tablo 3 | v1 `02` | seed 42 |
| Tablo 4 | v1 `03` (43, 44) ve `18` (45–49); `ek/02` | sekiz seed |
| Tablo S2.6, 3 ve 6 sa | v1 `21`; DM ve bootstrap: `ek/01` | diğer beş türbin, tek seed |
| Şekil 2, Tablo S5.2, S5.8 | v1 `14` (m eğrisi), `17`, `19`, `20`, `22` (VMD) | |
| Tablo S5.12 | `scripts_v2/deney5_placebo_gelecek.py` | placebo-gelecek, T57, m=2 |
| Tablo 6, Şekil 3, Tablo S4.2 | v1 `29`; `scripts_v2/sertifika_genis_tau.py` | τ ≤ 32 ve τ ≤ 2048 |
| Tablo S4.1 | `scripts_v2/ileri_erisim_analitik.py` | |
| Tablo S4.3 | `scripts_v2/deney1_kor_prob.py` | sekiz işleç, etiketler gizli |
| Tablo S5.13 (λ) | `scripts_v2/deney3_lambda_turbinler.py`; v1 `14` (m = H oracle) | keşfedici, ana metin buna dayanmaz |
| Tablo 5 | v1 `16` (06 değil: 06 VMD süresini düşük gösterir) | |
| Tablo S5.5 | `scripts_v2/ek04_stvmd_kiyas.py` | aynı oturumda ölçüm |
| Tablo S5.3, S5.9 | v1 `28` | |
| Tablo S5.4 | v1 `27` | |
| Tablo S5.6 | `scripts_v2/ek05_rolling_origin.py` | |
| Tablo S5.7 | `ek/08_esdegerlik/ek08_esdegerlik.py` | |
| Tablo S5.10 | `scripts_v2/deney4_ucuz_bankalar.py` | LogButter ve EMA farkı |
| Kelmarsh, S6 | `scripts_v2/ek07_kelmarsh.py` | |
| GRU / TCN | v1 `26` | |
| Pencere 7/14/28 gün | v1 `12` | |
| Nisan 2020 boşluğu | v1 `11` | |
| 10 dk ızgara | v1 `23` | |
| Eşit eğitim kümesi | v1 `31` | |
| ERA5 interpolasyon | v1 `25` | |
| Filtre sayısal (SOS) | v1 `30` | |
