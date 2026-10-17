"""Değerlendirme ölçütleri ve anlamlılık testleri (makale Bölüm 5)."""
import numpy as np
from scipy import stats
from .config import KAPASITE


def metrics(y, p, y_cur, cap=KAPASITE):
    """MAE, RMSE, nMAE, nRMSE, R2 ve persistence'a göre Skill (%).
    y_cur: tahmin anındaki gerçek güç P(t) = persistence tahmini."""
    mae = np.mean(np.abs(y - p)); rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    mae_pers = np.mean(np.abs(y - y_cur))
    return dict(MAE=mae, RMSE=rmse, nMAE=100 * mae / cap, nRMSE=100 * rmse / cap, R2=r2,
                Skill=100 * (mae_pers - mae) / mae_pers)


def lsi(e_causal, e_leaky):
    """Leakage Sensitivity Index (%) = (E_nedensel - E_sızdıran) / E_nedensel * 100.
    Aynı model ailesinde sızdıran ve nedensel hata kıyaslanır."""
    return 100 * (e_causal - e_leaky) / e_causal


def lambda_orani(e_nedensel, e_sizdiran, e_oracle):
    """Oracle-normalize sızıntı oranı: lambda = (E_nedensel - E_sızdıran) / (E_nedensel - E_oracle).
    Payda <= 0 ise tanımsızdır (NaN). lambda [0,1] dışına çıkabilir (oracle sızdırandan kötüyse > 1):
    aralık garanti değildir, sonuç mutlak değerlerle birlikte raporlanmalıdır. Döner: (lambda, uyarı metni veya '')."""
    pay = e_nedensel - e_sizdiran; payda = e_nedensel - e_oracle
    if payda <= 0:
        return float('nan'), 'payda <= 0: oracle nedenselden iyi değil, lambda tanımsız'
    lam = pay / payda
    return float(lam), ('' if 0 <= lam <= 1 else 'lambda [0,1] dışında (oracle sızdırandan kötü veya sızdıran nedenselden kötü)')


def dm_test(y, p1, p2, h=1, loss='abs'):
    """Diebold-Mariano testi + Harvey-Leybourne-Newbold (1997) küçük örneklem düzeltmesi.
    DM < 0: ilk model (p1) daha iyi. h: tahmin ufku (adım) -> Newey-West benzeri uzun dönem varyans."""
    e1, e2 = y - p1, y - p2
    d = (np.abs(e1) - np.abs(e2)) if loss == 'abs' else (e1 ** 2 - e2 ** 2)
    T = len(d); dbar = d.mean(); dc = d - dbar
    lrv = np.dot(dc, dc) / T
    for k in range(1, max(h, 1)):
        lrv += 2 * (1 - k / h) * np.dot(dc[k:], dc[:-k]) / T
    lrv = max(lrv, 1e-12)
    dm = dbar / np.sqrt(lrv / T)
    dm_c = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * (1 - stats.t.cdf(abs(dm_c), df=T - 1))
    return dict(DM=dm_c, p_raw=p, T=T, loss=loss)


def holm(pvals):
    """Holm (1979) basamaklı düzeltme: düzeltilmiş p-değerleri (aile-bazlı hata oranı)."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
