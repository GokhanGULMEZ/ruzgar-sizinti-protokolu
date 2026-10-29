# İstatistik protokolü (makale Ek'ine yapıştırılabilir metin, İngilizce) + Türkçe not

Kaynak: `leakage/evaluation.py` (v1 paketi, değiştirilmedi). Aşağıdaki her cümle o koddan çıkarılmıştır.

## English text for the appendix

**Loss differential and test statistic.** For two forecasts p1 and p2 of the same target y over the same T test points, the loss differential is d_t = |y_t − p1_t| − |y_t − p2_t| (absolute loss) or d_t = (y_t − p1_t)² − (y_t − p2_t)² (squared loss). The Diebold–Mariano statistic is DM = d̄ / sqrt(V̂ / T), where d̄ is the mean of d_t. A negative DM means that the first model has the lower loss.

**Long-run variance.** V̂ is the Bartlett-weighted estimate γ̂₀ + 2 Σ_{k=1}^{ℓ} (1 − k/(ℓ+1)) γ̂_k, with sample autocovariances γ̂_k computed with divisor T. The default number of lags is ℓ = h − 1, where h is the forecast horizon in steps (30 min per step); hence ℓ = 1 at 1 h, ℓ = 5 at 3 h, ℓ = 11 at 6 h. V̂ is floored at 10⁻¹². A lag sensitivity analysis (ℓ = 1, 6, 24, 48) is reported separately.

**Small-sample correction and p-value.** DM is multiplied by sqrt((T + 1 − 2h + h(h−1)/T) / T) (Harvey, Leybourne and Newbold, 1997). The two-sided p-value comes from a Student t distribution with T − 1 degrees of freedom.

**Multiple testing.** Holm's step-down procedure is applied within each family; a result is called significant only if the Holm-adjusted p < 0.05 and DM < 0 (the leaky model is better). The families are fixed by the analysis they belong to (e.g., the 30 comparisons of Appendix A; the 24 comparisons against persistence; the 16 per-seed tests of one turbine = 8 seeds × 2 comparisons) and are listed next to each table. [**Check against Section 5 of the manuscript before pasting.**]

**Dependence.** The tests are unconditional tests of equal average accuracy; they do not test conditional predictive ability. All six turbines belong to one farm, share the weather and the test period, and the seeds of one turbine share the same data. Therefore turbine–seed pairs are not treated as independent units. Two additional analyses are reported: (i) seeds are averaged within turbine and the turbine is the unit (n = 6; mean LSI, t-interval with 5 degrees of freedom, number of turbines with positive LSI); (ii) a common-time moving-block bootstrap (block = 96 steps = 2 days, 2,000 resamples) that resamples the same time blocks for all turbines jointly and reports the interval of the cross-turbine mean LSI. These intervals reflect test-sampling uncertainty; training (seed) variability is handled in (i).

**LSI.** LSI = (E_causal − E_leaky) / E_causal × 100, computed within the same model family on the same test points. It is a relative performance gap; it is not a direct measure of leaked information.

## Türkçe not (kısa)

- Bu metin yeni bir analiz yapmaz, mevcut kodun ne yaptığını açık yazar. "DM varyansı, HLN ve Holm aileleri nasıl tanımlandı?" sorusunun cevabıdır.
- Köşeli parantezli uyarıyı yapıştırmadan önce makalenin 5. bölümüyle karşılaştır; aile sayılarını (30, 24, 16) makaledeki gibi bırak.
- Bağımlılık paragrafındaki iki analiz `02_birim_bagimlilik/` betiğiyle üretilir.
