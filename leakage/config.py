"""Tüm sabitler tek yerde. Makaledeki değerlerle birebir aynıdır (Bölüm 3-4)."""
import os

# --- Veri (Bölüm 3) ---------------------------------------------------------
KAPASITE = 1500.0        # kW. SDWPF: Sinovel SL1500/82 -> anma gücü 1,5 MW (Zhou vd. 2024)
STEP_MIN = 30            # ortak çözünürlük (dk). 2020: 15 dk, 2021: 10 dk ham; :10 ve :40 anları alınır
MEASURE_MINUTES = (10, 40)

# --- Protokol (Bölüm 4.1) ---------------------------------------------------
H_LIST = [1, 2, 4, 6, 8, 12]   # ufuk (30 dk adım): 0,5 / 1 / 2 / 3 / 4 / 6 saat
H4 = [2, 6, 12]                # LCSF/B9/ablasyon için kullanılan ufuklar: 1 / 3 / 6 saat
SEQ_OPTS = [4, 8, 12]          # aday geçmiş pencereleri: 2 / 4 / 6 saat (doğrulamada seçilir)
L_BEST = 4                     # doğrulamada seçilen: 2 saat (01_L_secimi.py ile yeniden doğrulanır)
PURGE = 12                     # bölmeler arası boşluk (adım) = 6 saat
SPLIT = (0.70, 0.15, 0.15)     # kronolojik eğitim/doğrulama/test
SEED = 42                      # ana tohum
SEEDS_EK = (43, 44)            # 3 tohumlu tekrarın ek tohumları (03). Sekiz tohum = 42 + SEEDS_EK + (45..49) [18 ve 24. betikler]

# --- Ayrıştırma (Bölüm 4.2-4.4) ---------------------------------------------
ALPHA, TAU, K, DC, INIT, TOL = 2000, 0.0, 4, 0, 1, 1e-7   # VMD
W = 672                        # Rolling VMD penceresi: 14 gün (30 dk adımda)
BUTTER_ORDER = 3               # LCSF Butterworth derecesi

# --- Modeller (Bölüm 4.6) ---------------------------------------------------
LSTM_UNITS = (64, 32)
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_EPOCHS = 50
LSTM_BATCH = 256
LSTM_PATIENCE = 5
CB_PARAMS = dict(iterations=500, learning_rate=0.05, depth=6, loss_function='MAE',
                 early_stopping_rounds=50, verbose=0, allow_writing_files=False)
OOF_BLOCKS = 4

# --- Yollar -----------------------------------------------------------------
# Ortam değişkenleriyle değiştirilebilir (Colab'da Drive yolları varsayılandır).
VERI_CSV = os.environ.get('RT_VERI_CSV', '/content/drive/MyDrive/RuzgarTahmin/veri/sdwpf_2001_2112_full.csv')
OUT = os.environ.get('RT_OUT', '/content/drive/MyDrive/RuzgarTahmin/yeni/')
