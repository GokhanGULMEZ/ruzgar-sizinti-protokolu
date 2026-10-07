"""ADIM 23 — Çözünürlük duyarlılığı: 30 dakikalık ortak ızgara yerine yerel 10 dakikalık ızgara (Bölüm 7 sınırlılık #3).
Neden yalnız 2021: ham veri 2020'de 15 dk, 2021'de 10 dk aralıklıdır; 10 dakikalık çözünürlük yalnız 2021'de mümkündür.
Tasarım (çözünürlük etkisini veri uzunluğundan ve pencereden ayırmak için İKİ koşul aynı kodla çalıştırılır):
  '30dk' : 2021 verisi, 30 dk ızgara (:10 ve :40 anları), kayan VMD penceresi 224 adım  (= 4,67 gün)
  '10dk' : 2021 verisi, 10 dk ızgara,                    kayan VMD penceresi 672 adım  (= 4,67 gün)
İki koşulda da: aynı tarih sınırları (eğitim/doğrulama/test bölme tarihleri ana deneydekiyle aynı; eğitim yalnız 2021'in ilk kısmıdır),
aynı fiziksel pencere, geçmiş penceresi 2 saat (L=4 / 12 adım), ufuk 1 saat (H=2 / 6 adım), aynı model ailesi.
Modeller: B1 persistence, B2 ham LSTM, B6 tam seri VMD (sızdıran), B7 kayan VMD (nedensel), B8 nedensel LCSF, B8L iki yönlü LCSF (sızdıran).
LSI-VMD = (B7 − B6)/B7, LSI-LCSF = (B8 − B8L)/B8. Tohumlar: RT_SEEDS (varsayılan 42,43,44). CatBoost yoktur.
Önkoşul: veri CSV'si. Önbellek ana çıktı klasörünün altındaki 'cozunurluk_30dk/' ve 'cozunurluk_10dk/' klasörlerine yazılır (ana önbellek bozulmaz).
Kullanım:
  python 23_cozunurluk_10dk.py [türbin=57]
Ortam değişkenleri:
  RT_ROLL=0     kayan VMD'yi ATLA (yalnız LSI-LCSF ve tam seri VMD; hızlıdır, ~15-30 dk). Kayan VMD için sonra RT_ROLL=1 ile tekrar çalıştırın
                (kayan VMD 10 dk'da ~1-2 saat sürer; bloklar diske yazılır, oturum kapanırsa aynı komutla sürer).
  RT_SEEDS=42,43,44
  RT_SMOKE=1    hızlı duman testi (verinin ilk 60 günü, küçük pencere, 2 epoch; sonuçlar anlamsızdır, yalnız kodun çalıştığını gösterir).
Çıktı: cozunurluk_ozet_T57.csv (tohum başına), cozunurluk_ozet_ort_T57.csv (ortalama ± std)."""
from _ortak import *
import sys, json, time, numpy as np, pandas as pd
from leakage.config import OUT, K, L_BEST, STEP_MIN, PURGE
from leakage.data import TurbineData, FEATURES, LAG_H, make_windows
from leakage.decomposition import LCSF, full_vmd, rolling_vmd
from leakage.models import fit_lstm, pred_lstm
from leakage.evaluation import mae, lsi

SMOKE = os.environ.get('RT_SMOKE') == '1'
ROLL = os.environ.get('RT_ROLL', '1') != '0'
SEEDS = tuple(int(a) for a in os.environ.get('RT_SEEDS', '42' if SMOKE else '42,43,44').split(','))
EPOCHS = 2 if SMOKE else None
PURGE_SAAT = PURGE * STEP_MIN / 60                  # ana deneydeki boşluk: 12 adım x 30 dk = 6 saat
ERA5 = ['T2m', 'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp']
# Eğitim yalnız 2021'in ilk ~5 ayıdır; test Eylül-Aralık'tır. Yıllık takvim öznitelikleri eğitimde görülmeyen değerlere taşar (ölçek dışı),
# bu yüzden bu deneyde çıkarılır (iki çözünürlükte de aynı). Saat ve haftanın günü her dönemde görüldüğü için kalır.
TAKVIM_YIL = ['ay', 'yilin_gunu', 'ay_sin', 'ay_cos']
FEAT = [f for f in FEATURES if f not in TAKVIM_YIL]


def ozellik(d, sph):
    """data.add_features'ın saatte `sph` adımlık ızgara için genellemesi (gecikmeler ve kayan ortalamalar yalnız geçmişe bakar)."""
    d = d.copy()
    d[ERA5] = d[ERA5].ffill(limit=sph)          # yalnız geçmişe bakan doldurma; 30 dk'da limit=2 (1 saat) ile aynı fiziksel süre (bkz. data.py)
    d['is_missing'] = d['Patv'].isna().astype(int); d['Patv'] = d['Patv'].clip(lower=0)
    idx = d.index
    d['saat'] = idx.hour; d['haftanin_gunu'] = idx.dayofweek; d['ay'] = idx.month; d['yilin_gunu'] = idx.dayofyear
    d['saat_sin'] = np.sin(2 * np.pi * d['saat'] / 24); d['saat_cos'] = np.cos(2 * np.pi * d['saat'] / 24)
    d['ay_sin'] = np.sin(2 * np.pi * d['ay'] / 12);     d['ay_cos'] = np.cos(2 * np.pi * d['ay'] / 12)
    d['wdir_sin'] = np.sin(np.deg2rad(d['Wdir']));      d['wdir_cos'] = np.cos(np.deg2rad(d['Wdir']))
    for h in LAG_H: d[f'Patv_lag{h}h'] = d['Patv'].shift(h * sph)
    d['Patv_rol1h'] = d['Patv'].shift(1).rolling(sph).mean()
    d['Patv_rol4h'] = d['Patv'].shift(1).rolling(4 * sph).mean()
    d['Wspd_rol1h'] = d['Wspd'].shift(1).rolling(sph).mean()
    return d


class TD(TurbineData):
    """TurbineData'nın, tarih sınırlarıyla bölünen ve ızgara adımı serbest sürümü (yalnız 2021)."""
    def sc(self, x):
        return (x[FEAT].values - self.mn) / self.rng

    def __init__(self, df_all, tid, step_min, sinir):
        self.tid = tid
        d = df_all[df_all['TurbID'] == tid].copy()
        d['Tmstamp'] = pd.to_datetime(d['Tmstamp'])
        d = d.sort_values('Tmstamp').drop_duplicates('Tmstamp').set_index('Tmstamp')
        d = d[d.index >= sinir['bas']]
        if SMOKE: d = d[d.index < sinir['bas'] + pd.Timedelta(days=60)]
        if step_min == 30: d = d[d.index.minute.isin([10, 40])]
        else:
            fark = d.index.to_series().diff().dropna().dt.total_seconds().median() / 60
            assert abs(fark - step_min) < 1e-6, f'Ham veri {step_min} dk aralıklı değil (medyan {fark} dk); başlangıç tarihini kontrol edin.'
        full = pd.date_range(d.index.min(), d.index.max(), freq=f'{step_min}min')
        d = d.reindex(full); d.index.name = 'Tmstamp'
        self.d = ozellik(d, 60 // step_min)
        pur = pd.Timedelta(hours=PURGE_SAAT)
        if SMOKE:
            n = len(self.d); i1, i2 = int(n * .7), int(n * .85); p = int(PURGE_SAAT * 60 / step_min)
            self.tr, self.va, self.te = self.d.iloc[:i1], self.d.iloc[i1 + p:i2], self.d.iloc[i2 + p:]
        else:
            self.tr = self.d[self.d.index < sinir['va_bas'] - pur]
            self.va = self.d[(self.d.index >= sinir['va_bas']) & (self.d.index < sinir['te_bas'] - pur)]
            self.te = self.d[self.d.index >= sinir['te_bas']]
        self.mn = self.tr[FEAT].min().values; mx = self.tr[FEAT].max().values
        self.rng = np.where(mx - self.mn == 0, 1, mx - self.mn)
        self.x_dec = self.d['Patv'].ffill().fillna(0).values.astype(float); self.N = len(self.x_dec)


def bilesenler(td, out, pencere, roll):
    """B6 (tam seri VMD), B8/B8L (LCSF) ve isteğe bağlı B7 (kayan VMD). Önbellek `out` altındadır."""
    os.makedirs(out, exist_ok=True); tag = f'T{td.tid}'
    lcsf = LCSF().fit(td.x_dec[:len(td.tr)])
    fa = os.path.join(out, f'vmd_full_{tag}.npy')
    if os.path.exists(fa): uA = np.load(fa)
    else: uA = full_vmd(td.x_dec, td.N); np.save(fa, uA)
    uA = uA.copy(); uA[:pencere - 1] = np.nan
    cC = lcsf.transform(td.x_dec, True); cL = lcsf.transform(td.x_dec, False); cC[:pencere - 1] = np.nan; cL[:pencere - 1] = np.nan
    Ut = {'B6': td.scale_comp(uA), 'B8': td.scale_comp(cC), 'B8L': td.scale_comp(cL)}
    if roll: Ut['B7'] = td.scale_comp(rolling_vmd(td.x_dec, tag, out_dir=out, window=pencere))
    return Ut


def ridge_kontrol(Dk):
    """Tohumsuz doğrusal kontrol (LSTM eğitim sorunlarını veri/tasarım sorunlarından ayırır). Betik 22'deki Ridge ile aynıdır."""
    from sklearn.linear_model import Ridge
    from leakage.config import KAPASITE
    sonuc = {}
    for k, D in Dk.items():
        f = lambda a: D[a]['X'].reshape(len(D[a]['X']), -1)
        best = None
        for al in (0.01, 0.1, 1, 10, 100):
            r = Ridge(alpha=al).fit(f('tr'), D['tr']['y'])
            e = mae(D['va']['y'], np.clip(r.predict(f('va')), 0, KAPASITE))
            if best is None or e < best[0]: best = (e, r)
        sonuc[k] = float(mae(D['te']['y'], np.clip(best[1].predict(f('te')), 0, KAPASITE)))
    return sonuc


def kosu(df, tid, ad, step_min, pencere, L, H, sinir):
    out = OUT + ('smoke_' if SMOKE else '') + f'cozunurluk_{ad}/'; os.makedirs(out, exist_ok=True)
    t0 = time.time(); td = TD(df, tid, step_min, sinir)
    print(f'[{ad}] N={td.N:,} | eğitim/doğrulama/test = {len(td.tr):,}/{len(td.va):,}/{len(td.te):,} | '
          f'eğitim {td.tr.index[0].date()}→{td.tr.index[-1].date()} | test {td.te.index[0].date()}→{td.te.index[-1].date()}', flush=True)
    Ut = bilesenler(td, out, pencere, ROLL)
    D0 = td.get_xy(L, H); t_ref = D0['te']['t']
    Dk = {'B2': D0}
    for k, U in Ut.items():
        D = td.get_xy_ext(L, H, U); assert np.array_equal(D['te']['t'], t_ref), f'test kümesi farklı ({k})'; Dk[k] = D
    print(f'[{ad}] hazırlık {(time.time()-t0)/60:.1f} dk | test örneği: {len(t_ref):,}', flush=True)
    rg = ridge_kontrol(Dk)
    print(f'[{ad}] Ridge (tohumsuz) MAE: ' + ' | '.join(f'{k} {v:.1f}' for k, v in rg.items()) + ' | persistence ' +
          f'{mae(D0["te"]["y"], D0["te"]["yc"]):.1f}', flush=True)
    rows = []
    for sd in SEEDS:
        fn = out + f'sonuc_T{tid}_tohum{sd}.json'
        m = json.load(open(fn)) if os.path.exists(fn) else {}
        m.setdefault('B1', float(mae(D0['te']['y'], D0['te']['yc'])))
        for k, D in Dk.items():                                      # yalnız eksik modeller eğitilir (RT_ROLL=1 ile devam edilebilir)
            if k in m: continue
            mod = fit_lstm(D, L, sd, EPOCHS); vl = mod.history.history['val_loss']
            m[k] = float(mae(D['te']['y'], pred_lstm(mod, D['te']['X'])))
            print(f'[{ad}]   {k}: {len(vl)} epoch, en iyi doğrulama kaybı {min(vl):.5f} (epoch {int(np.argmin(vl))+1})', flush=True)
            json.dump(m, open(fn, 'w'))
        r = dict(cozunurluk=ad, tohum=sd, **m)
        r['LSI_VMD'] = lsi(m['B7'], m['B6']) if 'B7' in m else np.nan
        r['LSI_LCSF'] = lsi(m['B8'], m['B8L'])
        r['LSI_LCSF_ridge'] = lsi(rg['B8'], rg['B8L']); r['LSI_VMD_ridge'] = lsi(rg['B7'], rg['B6']) if 'B7' in rg else np.nan
        rows.append(r); print(f'[{ad}] tohum {sd}: ' + ' | '.join(f'{k} {v:.1f}' for k, v in m.items()) +
                              f' | LSI-VMD {r["LSI_VMD"]:.1f} LSI-LCSF {r["LSI_LCSF"]:.1f}', flush=True)
    return rows


if __name__ == '__main__':
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 57
    df = veri_yukle()
    if SMOKE: sinir = dict(bas=pd.Timestamp('2021-01-01'))
    else:                                                              # bölme tarihleri ana deneyden (30 dk, 2020-2021) alınır
        ana = TurbineData(df, tid)
        sinir = dict(bas=pd.Timestamp('2021-01-01'), va_bas=ana.va.index[0], te_bas=ana.te.index[0])
        print('Bölme tarihleri (ana deney): doğrulama başı', sinir['va_bas'], '| test başı', sinir['te_bas'])
    W30, W10 = (20, 60) if SMOKE else (224, 672)                      # her ikisi de 4,67 gün (smoke: 10 saat)
    rows = kosu(df, tid, '30dk', 30, W30, 4, 2, sinir) + kosu(df, tid, '10dk', 10, W10, 12, 6, sinir)
    R = pd.DataFrame(rows); pre = 'smoke_' if SMOKE else ''
    R.to_csv(OUT + f'{pre}cozunurluk_ozet_T{tid}.csv', index=False)
    S = R.groupby('cozunurluk', sort=False).agg(['mean', 'std']).drop(columns='tohum', errors='ignore')
    S.columns = [f'{a}_{b}' for a, b in S.columns]; S = S.round(2).reset_index()
    S.to_csv(OUT + f'{pre}cozunurluk_ozet_ort_T{tid}.csv', index=False)
    print('\n', S.T.to_string(header=False)); print('yazıldı:', OUT)
