"""Placebo-gelecek kontrolü: ileri-bakış deneyindeki pencere GEOMETRİSİNİ sabit tutup gelecek BİLGİSİNİ çıkarır.

Mevcut doz-yanıt deneyleri (lookahead.py, vmd_ileri.py) bileşeni t anında [.., t+m] penceresinden üretir; x(t+1..t+m) gerçek
gelecektir. m değişince iki şey birlikte değişir: (i) pencerenin sonuna göre t'nin konumu (uç etkisi, geometri) ve
(ii) pencereye giren gelecek bilgisi. Burada pencere yine [.., t+m] biter, ama son m örnek gelecek bilgisi taşımayan
bir kuyrukla değiştirilir:

  gercek : x(t+1..t+m) — mevcut deney (kontrol, aynı koşuda yeniden üretilir)
  tut    : x(t) sabit tutulur              (yalnız geçmişten, düz uzatma)
  ayna   : x(t+j) := x(t-j)                (yalnız geçmişten, çift yansıma; sınır etkisi için standart)
  artim  : x(t) + eğitim kesitinden rastgele bir bloğun artımlarının birikimi (gelecekle ilgisiz, t'ye bağlanmış)

Üç sahte kuyruk da t anındaki bileşeni YALNIZ x(..t) ve eğitim verisine bağlar; yani bileşen nedenseldir. tests/test_placebo.py
bunu doğrular (x(t+1..) değiştirilince bileşen değişmez; 'gercek' kuyrukta değişir).
Yorum: LSI_m(placebo) küçükse (m=0 düzeyine yakın) gerçek deneydeki artış geometriden değil bilgiden gelir.
Placebo LSI'nın yüksek kalan kısmı pencere geometrisine ve uç etkisine atfedilir.
Sınır: pencere uzunluğu farkı (tam seri VMD, ~35 bin örnek, kayan pencereye karşı 672) bu deneyle ayrışmaz."""
import os, time
import numpy as np, pandas as pd
from scipy.signal import filtfilt
from joblib import Parallel, delayed
from .config import W, K, OUT, L_BEST
from .decomposition import vmd_sorted, blok_temizle

MODLAR = ('gercek', 'tut', 'ayna', 'artim')
GERI = 720          # LCSF filtre geçişi için geriye bakılan adım (lookahead.py ile aynı)


def kuyruk(x, t, n, mod, ntr, tohum=0):
    """t'den sonraki n adım için değerler. mod='gercek' dışında gelecek bilgisi içermez."""
    if n <= 0: return x[:0]
    if mod == 'gercek': return x[t + 1:t + 1 + n]
    if mod == 'tut': return np.full(n, x[t], dtype=float)
    if mod == 'ayna': return x[t - np.arange(1, n + 1)]
    if mod == 'artim':
        rng = np.random.default_rng(tohum * 1_000_003 + t)
        for _ in range(50):                           # t'ye yakın olmayan rastgele bir eğitim bloğu (olanaksızsa son çekiliş)
            s = int(rng.integers(0, ntr - n - 1))
            if abs(s - t) > 1000: break
        return x[t] + np.cumsum(np.diff(x[s:s + n + 1]))
    raise ValueError(mod)


def lcsf_placebo(lc, x, m, mod, ntr, bas=W - 1, geri=GERI):
    """t >= bas için bileşen(t) = filtfilt(x[t-geri+1 .. t] + kuyruk(m))[t]. mod='gercek' lookahead.lcsf_ileri_bakis ile aynıdır."""
    N = len(x); out = np.full((N, K), np.nan)
    for t in range(bas, N):
        a = max(0, t - geri + 1); n = min(N, t + m + 1) - (t + 1)
        seg = np.concatenate([x[a:t + 1], kuyruk(x, t, n, mod, ntr)])
        for k, (bb, aa) in enumerate(lc.bands):
            out[t, k] = filtfilt(bb, aa, seg)[t - a]
    return out


def _vmd_t(x, t, m, window, mod, ntr):
    N = len(x); bas = t - window + 1; n = min(N, t + m + 1) - (t + 1)
    seg = np.concatenate([x[bas:t + 1], kuyruk(x, t, n, mod, ntr)])
    if len(seg) % 2: seg = seg[1:]; bas += 1          # vmdpy: tek uzunlukta son örnek atılır; biz ilk örneği atarız (vmd_ileri.py ile aynı)
    return vmd_sorted(seg)[:, t - bas]


def vmd_placebo(x, tag, m, mod, ntr, out_dir=OUT, window=W, block=2000, n_jobs=-1):
    """VMD(x[t-window+1 .. t] + kuyruk(m)) içinde t anındaki mod değerleri. Bloklar diske yazılır; yarıda kalırsa aynı komut sürer.
    Süre: her (m, mod) için kayan VMD ile aynı mertebe (türbin başına yaklaşık bir saat)."""
    N = len(x); fn_all = os.path.join(out_dir, f'vmd_placebo_{tag}_m{m}_{mod}.npy')
    if os.path.exists(fn_all): return np.load(fn_all)
    U = np.full((N, K), np.nan)
    for s in range(window - 1, N, block):
        e = min(s + block, N); fn = os.path.join(out_dir, f'vmdplacebo_{tag}_m{m}_{mod}_blk_{s}.npy')
        if os.path.exists(fn): U[s:e] = np.load(fn); continue
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(_vmd_t)(x, t, m, window, mod, ntr) for t in range(s, e))
        blk = blok_temizle(res); np.save(fn, blk); U[s:e] = blk
        print(f'   VMD placebo m={m} {mod} {tag}: {e}/{N} | {time.time()-t0:.0f} sn', flush=True)
    np.save(fn_all, U); return U


def placebo_egrisi(df_all, tid=57, tur='lcsf', ms=(2,), modlar=('ayna', 'artim'), H=2, seeds=(42, 43, 44),
                   L=L_BEST, window=W, out=OUT, epochs=None, gercek=True):
    """tur='lcsf' (dakikalar) veya 'vmd' (saatler). Referans: nedensel kol (LCSF için B8, VMD için B7) ve sızdıran kol (B8L / B6).
    Her (m, mod) için LSTM eğitilir. LSI = (MAE_nedensel − MAE_kol) / MAE_nedensel × 100 (ileri-bakış deneyindeki tanımla aynı).
    gercek=True: aynı m için gerçek gelecek kolu da eklenir (LCSF'de yeniden üretilir; VMD'de vmd_ileri_T{tid}_m{m}.npy önbelleği varsa)."""
    from .data import TurbineData
    from .decomposition import LCSF
    from .models import fit_lstm, pred_lstm
    from .evaluation import mae, lsi
    from .pipeline import build_components
    td = TurbineData(df_all, tid); x = td.x_dec; ntr = len(td.tr); tag = f'T{tid}'
    Ut, lc = build_components(td, out, window)
    ref, sizd = ('B8', 'B8L') if tur == 'lcsf' else ('B7', 'B6')
    girdi = {'nedensel': td.get_xy_ext(L, H, Ut[ref]), 'sizdiran (tam seri)': td.get_xy_ext(L, H, Ut[sizd])}
    meta = {'nedensel': (None, None), 'sizdiran (tam seri)': (None, None)}
    for m in ms:
        for mod in (('gercek',) if gercek else ()) + tuple(modlar):
            if tur == 'lcsf':
                U = lcsf_placebo(lc, x, m, mod, ntr, window - 1)
            elif mod == 'gercek':
                fn = os.path.join(out, f'vmd_ileri_{tag}_m{m}.npy')
                if not os.path.exists(fn): print(f'   (VMD gerçek-gelecek önbelleği yok, atlandı: {fn})'); continue
                U = np.load(fn); U[:window - 1] = np.nan
            else:
                U = vmd_placebo(x, tag, m, mod, ntr, out, window); U[:window - 1] = np.nan
            ad = f'm={m} {mod}'; girdi[ad] = td.get_xy_ext(L, H, td.scale_comp(U)); meta[ad] = (m, mod)
            print(f'   {ad} bileşenleri hazır', flush=True)
    t_ref = girdi['nedensel']['te']['t']
    for ad, D in girdi.items():
        assert np.array_equal(D['te']['t'], t_ref), f'test kümesi farklı ({ad})'
    rows = []
    for sd in seeds:
        M = {ad: mae(D['te']['y'], pred_lstm(fit_lstm(D, L, sd, epochs), D['te']['X'])) for ad, D in girdi.items()}
        for ad in girdi:
            rows.append(dict(tohum=sd, kosul=ad, m=meta[ad][0], mod=meta[ad][1], MAE=M[ad],
                             LSI=(0.0 if ad == 'nedensel' else lsi(M['nedensel'], M[ad]))))
        print(f'  tohum {sd}: ' + ' | '.join(f'{ad} {M[ad]:.1f}' for ad in girdi), flush=True)
    d = pd.DataFrame(rows)
    ozet = d.groupby('kosul', sort=False).agg(m=('m', 'first'), mod=('mod', 'first'), MAE=('MAE', 'mean'), LSI_ort=('LSI', 'mean'),
                                              LSI_std=('LSI', lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)).reset_index()
    return d, ozet
