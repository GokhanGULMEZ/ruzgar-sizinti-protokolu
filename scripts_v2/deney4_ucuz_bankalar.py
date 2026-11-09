"""Deney 4 - LCSF, VMD'den turetilmemis ucuz nedensel bankalarla karsilastirilir.
Soru: LCSF'nin VMD'den ogrenilen bant kenarlari gerekli mi, yoksa herhangi bir nedensel filtre bankasi ayni dogrulugu verir mi?
Kollar (hepsi nedensel, ayni 37 oznitelik, ayni egitim kumesi skip=W-1, ayni LSTM ve tohumlar 42,43,44):
  lcsf_kontrol : LCSF (dogrulama; T57 icin beklenen 115.94 kW, 3 tohum ortalamasi)
  log_butter   : sabit logaritmik aralikli Butterworth bankasi (VMD kullanilmaz). DIKKAT: butter() fs verilmeden cagrildi, bu yuzden
               0.0039, 0.0156, 0.0625 Nyquist'e gore oranlardir; gercek kenarlar 0.00195, 0.0078, 0.03125 cycle/adim (periyot 512, 128, 32 adim)
  ema_fark     : uc ussel ortalama (0.5, 0.1, 0.02) ve fark bankasi; bilesenler e3, e2-e3, e1-e2, x-e1
Karsilastirma icin Drive'daki onceki tahminlerden: ham LSTM (B2), rolling VMD (B7), LCSF (B8), 3 tohum ortalamasi.
Ortam: RT_PKG, RT_OUT, RT_EK_OUT. Ayar: ALT_TURB="57,10,83,120,61,84". Cikti: deney4_ucuz_bankalar.csv (her turbin bitince yazilir)."""
import os, sys, time, pickle
sys.path.insert(0, os.environ.get('RT_PKG', '/content/paket'))
import numpy as np, pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from leakage import config
from leakage.config import W, L_BEST, OUT
from leakage.data import TurbineData
from leakage.decomposition import LCSF
from leakage.models import fit_lstm, pred_lstm
from leakage.evaluation import mae

EK = os.environ.get('RT_EK_OUT', '/content/drive/MyDrive/RuzgarTahmin/ek_cikti/'); os.makedirs(EK, exist_ok=True)
H = 2; SEEDS = (42, 43, 44)
TURB = [int(x) for x in os.environ.get('ALT_TURB', '57,10,83,120,61,84').split(',') if x]
LOG_EDGES = [0.0039, 0.0156, 0.0625]   # Nyquist'e gore oran (fs verilmedi): gercek 0.00195, 0.0078, 0.03125 cycle/adim


def bank_log(x):
    B = [butter(3, LOG_EDGES[0], 'low'), butter(3, [LOG_EDGES[0], LOG_EDGES[1]], 'band'),
         butter(3, [LOG_EDGES[1], LOG_EDGES[2]], 'band'), butter(3, LOG_EDGES[2], 'high')]
    return np.column_stack([lfilter(b, a, x, zi=lfilter_zi(b, a) * x[0])[0] for b, a in B])


def ema(x, al):
    b, a = [al], [1.0, -(1.0 - al)]
    return lfilter(b, a, x, zi=lfilter_zi(b, a) * x[0])[0]


def bank_ema(x):
    e1, e2, e3 = ema(x, 0.5), ema(x, 0.1), ema(x, 0.02)
    return np.column_stack([e3, e2 - e3, e1 - e2, x - e1])


def evaluate(td, U, tag):
    Ut = td.scale_comp(U); D = td.get_xy_ext(L_BEST, H, Ut, skip=W - 1); out = []
    for sd in SEEDS:
        m = fit_lstm(D, L_BEST, sd, None); p = pred_lstm(m, D['te']['X']); out.append(mae(D['te']['y'], p))
    print(f'  {tag}: seed MAE =', np.round(out, 2), flush=True); return float(np.mean(out)), out


def prev(tid, key):
    try:
        b = pickle.load(open(os.path.join(OUT, f'adim6_T{tid}.pkl'), 'rb')); s = pickle.load(open(os.path.join(OUT, f'seed_T{tid}.pkl'), 'rb'))
        v = [mae(b[(key, H)]['y'], b[(key, H)]['p'])] + [mae(s[(key, sd)]['y'], s[(key, sd)]['p']) for sd in (43, 44)]
        return float(np.mean(v))
    except Exception as e:
        print('  (onceki sonuc yok:', key, repr(e)[:60], ')'); return float('nan')


def main():
    df = pd.read_csv(config.VERI_CSV); rows = []
    for tid in TURB:
        print(f'T{tid}', flush=True); t0 = time.time()
        td = TurbineData(df, tid); x = td.x_dec.astype(float)
        r = dict(turbin=tid, ham_B2=prev(tid, 'B2'), rolling_VMD_B7=prev(tid, 'B7'), LCSF_B8=prev(tid, 'B8'))
        if tid == 57:
            lc = LCSF().fit(x[:len(td.tr)]); r['lcsf_kontrol'], _ = evaluate(td, lc.transform(x, True), 'lcsf_kontrol (beklenen 115.94)')
        r['log_butter'], _ = evaluate(td, bank_log(x), 'log_butter')
        r['ema_fark'], _ = evaluate(td, bank_ema(x), 'ema_fark')
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, f'({(time.time()-t0)/60:.1f} dk)', flush=True)
        rows.append(r); pd.DataFrame(rows).to_csv(os.path.join(EK, 'deney4_ucuz_bankalar.csv'), index=False)
    d = pd.DataFrame(rows); print(d.round(2).to_string(index=False))
    print('turbinler arasi ortalama:'); print(d.drop(columns='turbin').mean().round(2).to_string())


main()
