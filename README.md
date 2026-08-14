# Circadian Cell Painting Analyzer

This is data pipeline that processes files mainly from Harmony/Signals
and tris to detect oscillations from features

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

For this experiment, oscillation detection was performed using cosinor analysis, which fits a cosine function to each feature's denoised/detrended time series with the period set to 24-hours. The model uses the following cosinor function:

$$Y\left(t\right) = M + A \cdot \cos\left(\frac{2\pi t}{\tau} - \Phi \right)$$

Here, $M$ is is the baseline (typically the midline estimating statistic of rhythm, or MESOR), $A$ is the amplitude, $\tau$ is the period _(which is set to 24-hours for this expriment,)_ and $\Phi$ is the acrophase (which is th phase of the peak.) Parameters for the described cosinor model were estimated using non-linear least squares through scipy's `curve_fit` function.

#### Statistical testing

The Significance of the oscillation was assessed using an F-test comparing the cosinor model (which has 3 parameters: baseline, amplitude, and phase) against a null model consisting of a flat-line at the mean (1 parameter). The F-statistic quantifies whether the variance from the cosinor fit is signfiicnatly greate than expected from adding two free parametrs to the null model.

**Testing correction**

P-values were adjusted within every replicate group (n=3 wells per group) across all features using the Benjamini-Hochberg method to adjust the false discovery rate at $\alpha=0.05$.

## Future directions and implementations

- Implement JTK_CYCLE into python package
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
