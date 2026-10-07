# Doğrulama kayıtları (Colab çıktıları)

`kontrol1_maliyet_ve_eslesme.ipynb` — paket kodunun gerçek veride çalıştırıldığının kaydı (hücre çıktıları içinde):
1. `scripts/06_maliyet_olcumu.py` çıktısı ve donanım (Intel Xeon 2,20 GHz, 2 çekirdek, Python 3.13).
2. `scripts/02_turbin_kosusu.py 10` ile özgün defter sonuçlarının eşleşme kontrolü: sekiz modelin hepsinde MAE farkı +%0,00, en büyük tahmin farkı 0,000.
3. Türbin başına en yüksek Patv değerleri ve ham zaman damgası aralıkları (sıralanmamış dosya notu için README, "Veri notları").

Not: Buradaki maliyet satırı (526x) üç tekrardan biridir; makalede kullanılan değer üç tekrarın ortancasıdır (bkz. README).
