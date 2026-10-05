"""Betiklerin ortak yardımcıları: paket yolu ve veri yükleme."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pandas as pd
from leakage import config


def veri_yukle():
    if not os.path.exists(config.VERI_CSV):
        sys.exit(f'Veri bulunamadı: {config.VERI_CSV}\n'
                 'RT_VERI_CSV ortam değişkeniyle yolu verin (README, "Veri" bölümü).')
    return pd.read_csv(config.VERI_CSV)


def cikti_klasoru():
    os.makedirs(config.OUT, exist_ok=True)
    return config.OUT
