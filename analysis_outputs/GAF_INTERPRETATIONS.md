# Multi-Asset GAF Interpretation Digest

## S&P 500 (^GSPC)
### Daily (1d)
The daily sweep covers 24,558 closes from 1927-12-30 through 2025-10-07 and yields 1,504 overlapping 512-day windows rendered as six-channel 512×512 Gramian images. Sign-based labels skew bullish—1,154 windows (76.7%) finish positive versus 350 bearish observations—while box-counting dimensions cluster near 1.19/1.09/1.20 for the symmetric channels and 1.82–1.87 for the anti-symmetric ones, all with standard deviations below 0.16. A 16-dimensional PCA embedding stores 256 representative windows with a centroid norm of 6.36×10⁻⁶, keeping cosine geometry numerically stable.

### Weekly (1wk)
Weekly aggregation spans the full 1927–2025 history with 5,103 observations that condense into 607 windows rendered at 256×256. Directional skew intensifies: 488 windows (80.4%) close higher, only 118 (19.4%) close lower, and a single window is neutral. Average box-counting scores rise to roughly 1.25 on the leading symmetric slice while anti-symmetric layers remain near 1.81–1.89 with tightened dispersion. Another 256 embeddings populate the same 16-D basis with a centroid norm of 3.03×10⁻⁶.

### Monthly (1mo)
Monthly sampling begins in 1985, producing 105 windows of ~15.3 years each at 256×256 resolution. Every window is labelled bullish under the ±10⁻⁶ neutrality band, reflecting the secular market drift. Box-counting means climb toward 1.30 on the symmetric channels and ~1.83 on the anti-symmetric trio while standard deviations fall into the 0.02–0.14 range. All 105 windows occupy the shared 16-D embedding with a centroid norm of 5.00×10⁻⁶.

### Cross-scale cosine alignment
Centroid similarities confirm a regime break: daily–weekly embeddings align at 0.94 with a permutation null mean of 0.39 (σ≈0.17) and two-sided p≈0.005, while both daily–monthly and weekly–monthly comparisons flip negative near −0.76 with p-values between 0.010 and 0.035. Monthly textures therefore inhabit an opposing orientation relative to the higher-frequency cubes despite sharing the same label skew.

### Fractality takeaways
Daily and weekly cubes preserve fractal motifs—volatility clustering, alternating angular gradients, and shared embedding orientation—across adjacent scales. Monthly cubes smooth those textures, saturate bullish labels, and rotate the embedding into a different quadrant, signalling that long-horizon evidence of fractality requires alternative labelling or detrending before cross-scale comparisons.

## Bitcoin (BTC-USD)
### Daily (1d)
The daily run spans 4,044 sessions from 2014-09-17 to 2025-10-12, delivering 222 overlapping 512-day windows at 512×512 resolution. Despite high volatility, 177 windows (79.7%) close higher. Box-counting dimensions average 1.18/1.07/1.15 for the symmetric channels and 1.81–1.86 for the anti-symmetric trio, all with dispersion below 0.14. A 16-D PCA embedding houses all 222 windows with a centroid norm of 1.04×10⁻⁵ and image diagnostics confirm a one-to-one mapping between PAA resolution and canvas size.

### Weekly (1wk)
Weekly sampling compresses the same span into 579 observations and 41 windows rendered at 256×256. Every window is labelled bullish, revealing that five-year slices smooth out daily drawdowns. Box-counting means jump toward 1.30 on the symmetric channels and 1.90 on anti-symmetric layers while dispersion tightens below 0.13. All 41 windows remain in the shared 16-D embedding with a centroid norm of 9.60×10⁻⁶.

### Monthly (1mo)
Only 135 monthly observations are available—short of the 180 bars required for a single 512-day window—so no monthly cube is generated. The summary captures the shortfall alongside the dormant image configuration to guide future reruns with shorter windows.

### Cross-scale cosine alignment
Daily and weekly centroids register a cosine similarity of −0.18. The permutation null averages +0.10 with σ≈0.30, yielding a two-sided p≈0.19 across 200 shuffles. Weekly aggregation therefore rotates Bitcoin’s embedding away from the daily orientation, although the evidence stops short of statistical significance.

### Fractality takeaways
Bitcoin displays scale-sensitive behaviour. Daily cubes preserve a mix of bullish bias and high-frequency roughness, whereas weekly cubes suppress label diversity yet amplify symmetric roughness. The absence of monthly imagery highlights the asset’s short history; cross-scale fractality tests should concentrate on daily versus weekly windows and consider adaptive window lengths to unlock macro-horizon comparisons.

## Apple (AAPL)
### Daily (1d)
Apple’s daily history spans 11,299 closes from 1980-12-12 through 2025-10-10, producing 676 windows rendered at 512×512. Bullish labels dominate (523 windows, 77.4%), while box-counting averages sit near 1.18/1.08/1.17 for symmetric channels and 1.81–1.85 for anti-symmetric layers. A 16-D embedding stores 256 windows with a centroid norm of 1.04×10⁻⁵.

### Weekly (1wk)
Weekly aggregation yields 2,341 observations and 261 windows at 256×256. Bullish saturation rises to 88.9% (232 windows), and box-counting means lift to ~1.25 on symmetric channels with anti-symmetric scores near 1.88. The weekly embedding contributes 256 samples with a centroid norm of 5.03×10⁻⁶.

### Monthly (1mo)
Monthly sampling leaves 105 windows (~15.3 years each) rendered at 256×256. Labels reach 96.2% bullishness, box-counting means climb to 1.35/1.26/1.30 on the symmetric trio and ~1.83 on anti-symmetric slices, and dispersion contracts below 0.14. All windows remain in the 16-D basis with a centroid norm of 4.95×10⁻⁶.

### Cross-scale cosine alignment
Daily and weekly centroids align at 0.992 with p≈0.005, indicating a unified regime. Daily–monthly cosine drops to −0.150 (p≈0.005) and weekly–monthly to −0.077 (p≈0.10), showing that the trend-dominated monthly cubes rotate away from the higher-frequency embedding.

### Fractality takeaways
Apple retains fractal persistence at daily and weekly horizons, where label skews, box-counting statistics, and embeddings move in lockstep. Monthly cubes, however, smooth the textures and pivot the embedding, so long-horizon studies need detrending or alternative labelling before claiming cross-scale self-similarity.

## EUR/USD (EURUSD=X)
### Daily (1d)
Daily FX fixes from 2003-01-02 to 2025-10-10 generate 521 windows at 512×512 with a nearly balanced label split (260 bullish, 261 bearish). Box-counting means hover near 1.18/1.07/1.16 for symmetric channels and 1.82–1.89 for anti-symmetric layers with standard deviations between 0.12 and 0.20. The embedding stores 256 windows with a centroid norm of 7.16×10⁻⁶ and passes the image-size audit.

### Weekly (1wk)
Weekly aggregation covers the same window with 1,140 observations and 111 windows at 256×256. Labels turn bearish (70 negative, 41 positive), box-counting means rise modestly to ~1.23/1.90 while dispersion tightens below 0.10, and all 111 windows remain in the shared 16-D embedding (centroid norm 3.61×10⁻⁶).

### Monthly (1mo)
Monthly resampling leaves 264 fixes and 29 windows (~15.3 years each). Labels saturate bearish (28 negative, 1 positive). Symmetric box-counting means fall toward 1.06 while anti-symmetric means stay near 1.84 with tiny dispersion (<0.034). The embedding retains 29 windows with a centroid norm of 1.38×10⁻⁵ and clean image diagnostics.

### Cross-scale cosine alignment
Daily–weekly cosine is −0.027 with p≈0.21, suggesting no strong relationship. Daily–monthly cosine reaches 0.43 with p≈0.055, hinting that extreme monthly regimes share orientation with daily textures. Weekly–monthly cosine is 0.051 yet attains a near-zero p-value because the permutation null is strongly negative (mean −0.94), underscoring that monthly embeddings diverge from weekly ones despite similar trends.

### Fractality takeaways
EUR/USD exhibits path-dependent fractality: daily cubes alternate between bullish and bearish regimes with high variance, weekly cubes capture persistent downtrends with homogeneous textures, and monthly cubes deliver uniformly bearish, ultra-consistent patterns that detach from the weekly embedding while partially echoing the daily geometry.

## Long-Duration Bonds (TLT)
### Daily (1d)
The daily sweep spans 5,839 closes from 2002-07-30 to 2025-10-10, yielding 334 windows at 512×512. Bullish windows account for 71.9% (240 samples). Box-counting means sit near 1.17/1.06/1.15 for symmetric channels and 1.80–1.86 for anti-symmetric layers with dispersion between 0.03 and 0.20. The embedding retains 256 windows with a centroid norm of 7.25×10⁻⁶ and clean image diagnostics.

### Weekly (1wk)
Weekly aggregation draws 1,209 observations and 120 windows rendered at 256×256. Label skew cools—60 bullish versus 60 bearish—while box-counting means climb to ~1.21/1.88 with reduced variance (<0.12). All windows remain in the shared 16-D embedding with a centroid norm of 7.06×10⁻⁶.

### Monthly (1mo)
Monthly sampling provides 279 observations and 34 windows. Labels are balanced (17 bullish, 17 bearish). Symmetric box-counting means drop toward 1.03 while anti-symmetric means hover around 1.82 with tight dispersion (<0.05). The embedding retains all 34 windows with a centroid norm of 9.73×10⁻⁶ and passes image diagnostics.

### Cross-scale cosine alignment
Daily–weekly cosine registers 0.27 with p≈0.045, revealing moderate alignment between short- and medium-term textures. Daily–monthly cosine is −0.32 (p≈0.035) and weekly–monthly is −0.15 (p≈0.25), indicating that monthly embeddings diverge from higher-frequency regimes despite balanced labels.

### Fractality takeaways
Bond-market cubes highlight how interest-rate regimes reshape textures. Daily and weekly embeddings share modest alignment, consistent with volatility clustering across short horizons. Monthly cubes introduce smoother symmetric surfaces and rotate the embedding away from the higher-frequency orientation, implying that structural shifts in yields disrupt self-similarity beyond the seven-year window.
