"""Rüzgâr gücü tahmininde zamansal (ileri-bakış) sızıntının ölçülmesi — kod paketi.

Modül haritası (makale bölümleriyle):
  config.py         Sabitler ve hiperparametreler             (Bölüm 3, 4)
  data.py           Veri yükleme, öznitelik, bölme, pencere   (Bölüm 3, 4.1)
  decomposition.py  Full VMD, Rolling VMD, LCSF               (Bölüm 4.2-4.4)
  models.py         LSTM, CatBoost, OOF kalıntı modeli        (Bölüm 4.6)
  evaluation.py     MAE/Skill/LSI, Diebold-Mariano, Holm      (Bölüm 5)
  pipeline.py       Türbin/ufuk koşuları, seed tekrarı        (Bölüm 6)
"""
