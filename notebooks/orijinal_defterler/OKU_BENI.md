# Özgün Colab defterleri (kullanıcının çalıştırdığı, makale sonuçlarını üreten kayıtlar)

Bu defterler paketin KAYNAĞIDIR; hücre çıktıları (sonuçlar) içlerinde durur. Paket kodu (leakage/, scripts/) bunlardan
yeniden düzenlenmiştir ve T10'da bu defterlerin ürettiği tahminlerle birebir aynıdır (scripts/08_eslesme_kontrolu.py).

| Defter | İçerik | Paket karşılığı |
|---|---|---|
| Makale_2_T57_taban_adimlari | rt_core, veri/bölme, L seçimi, B1-B3, Full/Rolling VMD, B6/B7, nedensellik kanıtı, türbin seçim kuralı (Hücre 3) | leakage/data.py, scripts/00, 01, 04 |
| Makale_3_2_LCSF_B9_ablasyon_T10_T83_T120 | LCSF bantları, B8/B8L, B9 OOF, DM+Holm, maliyet (ESKİ ölçüm), ablasyon, run_turbine, T10/T83/T120 | leakage/decomposition.py, models.py, pipeline.py, scripts/02, 04, 06 |
| Makale_3_kopya_T61_T84_seed | T61/T84 koşuları, seed tekrarı (Hücre 21) | scripts/02, 03 |

Dahil EDİLMEYEN: ilk protokolle (10 dk, kapasite 2000 kW) yazılmış "Makael_1" defteri. Sonuçları geçersizdir; makale sayıları için kullanılmaz.
Defterdeki maliyet hücresi (46,3 ms / 112 µs / 413x) eskidir; makalede scripts/06 ile alınan üç tekrarın ortancası kullanılır.

Dahil EDİLMEYEN ek dosyalar (5 Ekim): `Makael_1` ve `Makael-1` (ilk protokol: kapasite 1521/2000 kW, purge yok; geçersiz), `ADIM1_protokol` (10 dk adımlı ilk protokol taslağı, 38 öznitelik; 30 dk protokolü ile değiştirildi), `Makale_3_1` (Makale_3_2 ile aynı kod), `Makale_3` (Makale_3_2'nin yarım kalmış kopyası: T83'te kesilmiş).
