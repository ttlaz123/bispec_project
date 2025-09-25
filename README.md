# Bispectrum fNL Amplitude Analysis

This project calculates the non-Gaussianity amplitude parameter, $f_{\text{NL}}$, from bispectrum data. It's designed to be used for the BICEP/Keck experiment but can be adapted for other data sets.

## Features

* **Calculates fNL:** Computes amplitudes using both diagonal and full covariance matrices.

* **Caches Data:** Saves calculated amplitudes to a CSV file to avoid re-computation.

* **Generates Plots:** Creates histograms of simulated $f_{\text{NL}}$ values with the observed value overlaid.

## How to Run

To run the analysis, use the following command with your desired parameters:

```bash
python calc_fnl.py --bn 10 --bk B33y --freqs 100 --simn 499
