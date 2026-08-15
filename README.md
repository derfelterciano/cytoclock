# CytoClock, the Circadian Cell Painting Analyzer

This is data pipeline that processes files mainly from Harmony/Signals
and tries to detect oscillations from features

---

## Methodology

These are the steps used to analyze all the data in this pipeline.

### Cell Imaging

Cells were imaged using Revvity's Opera Phenix system for high-content screening. Revvitty's SignalsImageArtists then calculated various phenotypic measurements to be used in further down-stream analysis. In this experiement, we calculated roughly around ~4700-5000 phenotypic measurements for each experiment. This results in a dataset that is roughly ~20GB

### Data Preprocessing

Outputted data from the imaging analysis resulted in a tab-delimited file that consists of phenotypic measurements (or features) for _every_ cell detected in all of the wells. Since this is a high-throughput experiment, we want to look at the measurements of the population of the cells rather than each individual cell. Furthermore, in a cytological profiling experiment, we look at the features in an un-biased matter therefore, we must include all ~4700-5000 features in our calculations. With that said, we calculated several statistics that describe the cell population within a well for each timepoint. These statistics are the following:

- mean
- median
- standard deviation
- 10% quantile
- 25% quantile
- 50% quantile
- 75% quantilee
- 90% quantile
- 100% quantile

We chose these metrics as they are typically the most robust in terms of summarizing the population's activity.

### Denoising/Smoothing & Detrending

We applied a two-pass moving average with a 24-hour window on both passes to extract the true noise from each feature's time-series data. This step can be through of as the average of the moving average. We then subtracted the noise from the raw data to give us the clean oscillating data.

A 24-hour window was chosen since it matches with the expected cell circadian period. Anything oscillating at the ~24 hour mark or faster gets averaged out in the rolling mean, so when you subtract the noise from the raw data, the oscillation is preserved in the residuals. Anything slower than 24 hours (due to confluency, cytoxicity, illumination difference, etc. ) stays in the rolling mean and gets removed.

A 2-pass method was used because the first window doesn't perfectly remove the noisy signal at the window frequency. Applying the same rolling average on the first rolling average's result gives us more of a cleaner cutoff which smoothens and separates the circadian-frequency oscillations from the low-frequency trend. The equivalent, but harder to explain, alternative to using the 2-pass rolling average method is called the Gaussian filter. This method was suggested to us by Devons from the Kimmey Lab using a series of Fourier Transforms to prove the effctiveness of th 2-pass rolling average method.

### Circadian detection

#### Models

##### Cosinor

For this experiment, oscillation detection was performed using cosinor analysis, which fits a cosine function to each feature's denoised/detrended time series with the period set to 24-hours. The model uses the following cosinor function:

$$Y\left(t\right) = M + A \cdot \cos\left(\frac{2\pi t}{\tau} - \Phi \right)$$

Here, $M$ is is the baseline (typically the midline estimating statistic of rhythm, or MESOR), $A$ is the amplitude, $\tau$ is the period _(which is set to 24-hours for this expriment,)_ and $\Phi$ is the acrophase (which is th phase of the peak.) Parameters for the described cosinor model were estimated using non-linear least squares (`scipy.optimize.curve_fit`).

##### Damped Cosinor

Some experimental conditions (particularly drug-treated wells) exhibited oscillations whose amplitude decayed over the course of the time-series rather than remaining constant. For these cases, a damped cosinor model was implemented, adding an exponential decay term to the standard cosinor equation:

$$Y\left(t\right) = M + A \cdot e^{-\lambda t} \cdot \cos\left(\frac{2\pi t}{\tau} - \Phi \right) + \epsilon$$

Here, $\lambda$ is the damping coefficient, which describes how quickly the oscillation's amplitude decays over time. Rather than reporting $\lambda$ directly, we convert it to the time constant $\tau_{decay} = 1/\lambda$ (in hours), since $\tau_{decay}$ has a more direct biological interpretation: it is the time required for the oscillation's amplitude to decay to ($1/e$) of its initial value.

Because the damping term assumes decay begins at $t=0$, timepoints were anchored to the first observation (i.e. $t_{i} - t_{0}$) before fitting, rather than using raw experiment time. This ensures the damping estimate reflects decay from the start of the recorded signal rather than decay that may have already partially occurred before the first measurement.

Parameters were again estimated via non-linear least squares (`scipy.optimize.curve_fit`), with bounds enforcing $A \geq 0$ and $\lambda \geq 0$, since negative damping would imply amplitude growth rather than decay, which is not physically meaningful for this model.

##### JTK_CYCLE and Empirical JTK_CYCLE (eJTK_CYCLE)

Cosinor-based methods assume a fixed period and a purely sinusoidal waveform, and are sensitive to outliers since they are fit via least-squares. Several experiments in this study (particularly gene expression data collected via EnVision) exhibited period drift over the course of the recording, which violates the fixed-period assumption underlying cosinor analysis. To address this, we implemented JTK_CYCLE (Hughes, Hogenesch, & Kornacker, 2010), a non-parametric, rank-based method for rhythm detection.

JTK_CYCLE evaluates a time series against a family of reference cosine waves spanning a range of candidate periods and phases. Rather than comparing raw values, it uses Kendall's rank correlation ($\tau$) between the observed data's pairwise rank ordering and each reference waveform's rank ordering, making it robust to outliers and to non-sinusoidal (but still monotonically rhythmic) waveforms. The best-fitting (period, phase) combination is selected by maximum Kendall $\tau$, and amplitude is estimated using the Hodges-Lehmann estimator (a robust, median-of-pairwise-averages statistic) rather than a least-squares amplitude, consistent with JTK_CYCLE v3.1.

Significance under standard JTK_CYCLE is assessed via a Bonferroni-adjusted p-value, computed from either the exact null distribution of the Kendall $\tau$ statistic (via the Harding combinatorial algorithm) or a normal approximation when the exact calculation becomes numerically infeasible for larger sample sizes.

Because the Bonferroni correction does not account for JTK_CYCLE's internal selection-of-best-lag procedure, it tends to be conservative. We additionally implemented **empirical JTK_CYCLE (eJTK_CYCLE)**, which computes p-values by permutation: the observed time series is randomly permuted many times (default $n=500$), JTK_CYCLE is re-run on each permutation, and the empirical p-value is the fraction of permutations whose best Kendall $\tau$ meets or exceeds the observed $\tau$. This produces p-values that more accurately reflect the true null distribution for this specific selection procedure, at the cost of additional computation.

For both JTK_CYCLE variants, once the best-fitting period is identified, we additionally report a fixed-period cosinor refit: a linear least-squares fit of $M + A\cos(\omega t) + B\sin(\omega t)$ with $\omega$ fixed to JTK's detected period. This is reported separately from JTK's own rank-based amplitude/phase estimate, since JTK's parameters are derived from rank concordance rather than being optimized against the raw signal values, and are therefore not guaranteed to minimize residual error the way a least-squares fit is. The fixed-period refit gives an $R^2$ goodness-of-fit statistic that the rank-based JTK output does not provide.

#### Statistical testing

The Significance of the oscillation was assessed using an F-test comparing the cosinor model (which has 3 parameters: baseline, amplitude, and phase) against a null model consisting of a flat-line at the mean (1 parameter). The F-statistic quantifies whether the variance from the cosinor fit is signfiicnatly greate than expected from adding two free parametrs to the null model.

**Testing correction**

P-values were adjusted within every replicate group (n=3 wells per group) across all features using the Benjamini-Hochberg method to adjust the false discovery rate at $\alpha=0.05$.

### Gene Expression Pipeline

Gene expression data (collected via a PerkinElmer EnVision plate reader, sampled every 30 minutes) was processed through a parallel pipeline sharing the same detrending and oscillation-detection framework used for cell painting data, with two additions specific to this data type.

#### Discrete Cosine Transform (DCT) for period pre-detection

Prior to detrending, we applied a Discrete Cosine Transform to each well's raw time series to obtain an initial estimate of the dominant oscillatory period, independent of any assumed circadian ($\sim$24hr) period. This step was necessary because several treatment conditions produced periods that deviated substantially from 24 hours (e.g. $\sim$30 hours), which would not be correctly captured by a cosinor/JTK model constrained to a 24-hour period. DCT-derived period estimates were computed per replicate well and summarized (mean, median, standard deviation) across replicates within a treatment group, then used to parameterize the detrending window and, where applicable, the period used for downstream cosinor fitting on a per-group basis.

#### Wavelet analysis

Because gene expression data more frequently exhibited both period drift and amplitude damping over the course of the recording, we additionally applied continuous wavelet transform analysis (via the `pyBoat` package) to selected wells/features. Unlike cosinor or JTK, which return a single period/phase/amplitude estimate per time series, wavelet analysis produces a time-resolved estimate of instantaneous period, amplitude, and phase throughout the recording, visualized as a wavelet power spectrum (period vs. time, with power indicated by color) alongside the dominant ridge (the instantaneous period with maximal power at each timepoint). This allowed us to directly visualize when an oscillation was present, how its period changed over time, and when/whether it damped out, rather than assuming these properties were constant across the full recording as cosinor/JTK do. A time-averaged Fourier spectrum was computed alongside the wavelet spectrum as a complementary, non-time-resolved summary of the dominant period.

### Network / module analysis (WGCNA)

To identify groups of features that co-vary together over time — and to reduce the dimensionality of the $\sim$4700-5000 feature cell painting dataset into a smaller number of biologically interpretable units — we implemented a Weighted Gene Co-expression Network Analysis (WGCNA)-style pipeline, adapted from the standard WGCNA methodology (Langfelder & Horvath, 2008) for our specific data structure.

Standard WGCNA assumes many independent samples (e.g. individual RNA-seq libraries) contributing to a single co-expression network. In this dataset, however, the natural unit of replication is the timepoint within a well, and biological replicates take the form of multiple wells under the same treatment condition rather than independent individuals. We therefore treat each treatment group's pooled timepoints (z-scored per well, then stacked across replicate wells) as the observation set for network construction, producing one network per treatment condition. Where no replicate/grouping structure is supplied, each well is instead processed independently.

**Correlation and adjacency.** A feature-by-feature correlation matrix was computed for each group (biweight midcorrelation by default, following WGCNA convention, with Pearson and Spearman available as alternatives). This was transformed into a weighted adjacency matrix via a soft-thresholding power $\beta$:

$$a_{ij} = \left(\frac{1+r_{ij}}{2}\right)^{\beta} \quad \text{(signed network)}$$

A signed network formulation was used rather than unsigned, since for circadian data, features that are anti-phase (e.g. one peaking near CT0, another near CT12) are biologically distinct rather than co-regulated, and an unsigned network would incorrectly merge them into the same module. The soft-thresholding power $\beta$ was chosen per the standard WGCNA criterion — the lowest power at which the network's connectivity distribution approximates a scale-free topology (assessed via the fit of $\log_{10}(p(k))$ vs. $\log_{10}(k)$, where $k$ is each feature's total connectivity) — selected by visual inspection of the scale-free fit index across a range of candidate powers.

**Topological overlap and module detection.** The adjacency matrix was converted to a Topological Overlap Matrix (TOM), which weights the similarity between two features by both their direct connection strength and the extent to which they share the same network neighbors, improving robustness to spurious pairwise correlations. Modules were identified via average-linkage hierarchical clustering on TOM-based dissimilarity ($1-TOM$), with a minimum module size enforced (features in clusters below this size were left unassigned). Highly similar modules (module eigengene correlation above a set threshold) were subsequently merged.

**Module eigengenes.** For each module, a representative "eigengene" time series was computed as the first principal component of its member features' (z-scored) values, sign-corrected so that the eigengene positively correlates with its member features. Because module structure was defined on pooled, replicate-stacked data, eigengene scores were subsequently projected back onto each individual well's own timepoints, yielding a per-well, per-timepoint eigengene value directly comparable in structure to any single feature's time series. Module eigengenes were then used as inputs to the same cosinor/JTK_CYCLE/eJTK_CYCLE rhythm-detection framework applied to individual features, allowing rhythm detection at the module level rather than only the individual-feature level. Module membership (feature-to-eigengene correlation, $kME$) was retained per feature to identify hub features best representing each module's overall behavior.

## Future directions and implementations

- Preprocess data for unidentifiable/irregular periods (such as 30 hour)

# CURRENT OBJECTIVES (TODO)

- [x] Implement harmony formatter
- [x] Implement **damped cosinor equation**
  - This should be relatively the same as the preivous cosinor base but with a
    few minor tweaks
  - The dampened cosinor can be seen [here](https://www.biorxiv.org/content/10.1101/2022.07.04.498691v3) at equation `2.2`
- ~~Implement `Lomb-Scargle` for period detection. We can only do this AFTER detrending.~~
- [x] Implement `eJTK_CYCLE` since our data has period drift we need to use this
      to everything more robust
- [x] Implement network analysis of features (WGCNA)
  - Use the following [link](https://gds-yazarlab.bilkent.edu.tr/wp-content/uploads/YeastTutorialHorvath.pdf) for time series data on WGCNA

- [x] Implement parallel genexpression analysis pipeline to cell painting pipline
  - [x] Implement **fft's discrete cosinor transform**
    - This is just a method to detect the periods in the experiment.
    - We still need to implement this as a package
    - This needs to be a CLI method
  - [x] Implement EnVision formatting pipeline
  - [x] Implement wavelet analysis from `pyBoat`
    - [x] Implement wavelet visualization
