# LuminousBandsInERing

Tools, pipelines and analysis notebooks for detecting and characterising inclined
periodic luminous-band structures in Saturn's E ring using Cassini ISS data: an
automated survey of ISS NAC images that detects the bands in CLEAR-subtracted
residual images with a zero-padded Fourier angular spectrum and a
phase-randomised null ensemble, and characterises their orientation, contrast and
viewing geometry.

- Paper: *[reference and DOI to be added on acceptance]*
- Code archive (this repository): Zenodo DOI *[to be added]*
- Output data (image list, classifications, figure data): Zenodo DOI *[to be added]*
- Observation-set tool: [PyISS](https://github.com/ArnaudMath/pyiss) (GitHub only;
  unrelated to the `pyiss` package on PyPI)

Author: Arnaud Mathieu. License: MIT.

## Installation

Python 3.11 or newer (the paper run used Python 3.13.5).

```bash
git clone https://github.com/ArnaudMath/LuminousBandsInERing
cd LuminousBandsInERing
pip install -r requirements-lock.txt   # exact versions of the paper run
# or: pip install .                    # compatible versions (see pyproject.toml)
# interactive notebooks: pip install ".[notebooks]"
```

PyISS is installed from GitHub at the exact commit used for the paper.
SPICE kernels are not included; step 03 checks and downloads the attitude (CK)
kernels from NAIF.

## Overview

```
OPUS query ──> observation sets ──> CLEAR/science pairs ──> calibrated images
   01               02 (PyISS)              02                    04
                                                                   │
residual images <── CLEAR subtraction <────────────────────────────┘
   05  ──> quality review (06) ──> detection (07) ──> visual review (09)
                                                          │
                     post-pipeline analysis and figures <─┘
```

## Pipeline (steps 00–09, repository root)

Run steps 01–06 with `python 00_run_pipeline.py` (query parameters at the top of
the file), or each step on its own. All outputs go to `data/`.

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 00 | `00_run_pipeline.py` | Runs steps 01–06 in sequence | – |
| 01 | `01_query_images.py` | Queries OPUS for ISS CLEAR images of Enceladus in the forward-scattering geometry (Table 1 of the paper) | `data/query_results/query_results.parquet`, `query_manifest.csv` |
| 02 | `02_infer_sets.py` | Reconstructs observation sets with PyISS and forms (CLEAR, science-filter) pairs | `data/inferred_sets/pairs.parquet`, `sets_raw.parquet`, `no_match.csv` |
| 03 | `03_check_kernels.py` | Checks SPICE CK coverage for every pair and downloads missing kernels | `data/kernels/`, updated metakernel |
| 04 | `04_download_images.py` | Downloads the calibrated `_CALIB.IMG/.LBL` files from OPUS (dry run without `--download`) | `data/raw_images/` |
| 05 | `05_clear_subtraction.py` | Residual = science − CLEAR (both CISSCAL I/F), 3σ outlier filter | `data/cisscal_output/<id>/residual.npz`, `grayscale.png`, `redblue.png` |
| 06 | `06_flag_review.py` | Interactive quality review of the residuals | `06_flagged_pairs.json` |
| 07 | `07_detect_bands.py` | Preprocessing, FFT and zero-padded angular spectra, null ensemble, smoothing (9 then 64 bins), peak and FWHM | `data/pipeline_output/<id>/bundle_A.png`, `spectra.npz`, `metrics.json`; `summary.parquet` |
| 08 | `08_obs_sequences.py` | Retrieves the flyby / orbit tag of every image | `data/pipeline_output/obs_sequences.csv` |
| 09 | `09_review_detections.py` | Interactive visual review: positive / doubtful / unlabelled | `09_detection_labels.json` |

Result of the paper run: 631 pairs in the main reconstruction, of which 52 were
excluded in the quality review; together with 45 pairs from a second
reconstruction run, 624 pairs went through detection and visual review, giving
62 positive and 26 doubtful classifications. The per-pair accounting is in the
data deposit (`image_pairs.csv`).

## Post-pipeline analysis (`post_pipeline/`)

| Script | What it does | Output |
|---|---|---|
| `11_review_bundle_A.py` | Final review labels (62 positive, 26 doubtful) | `11_bundle_A_review_labels.json` |
| `12_selected_bundle_A_geometry_overview.py` | Cassini and Enceladus positions for every pair (single reference epoch) | `12_selected_bundle_A_geometry_overview.json` |
| `12b_geometry_per_epoch.py` | Same, with the Sun direction at each image epoch (used for the ansa assignment) | `12b_geometry_per_epoch.json` |
| `13_profile_slices.ipynb` | Interactive multi-slice profile extraction and peak/trough picks | `13_profile_slices/<id>/picks.json` |
| `14_contrast_normalised_residual.py` | Band amplitude δ and SNR from the picks | `14_contrast_table.csv` |
| `14b_contrast_michelson.py` | Normalised contrast C = δ / (I_max + I_min) used in the paper | `14b_contrast_table.csv` |
| `15_contrast_inspection.ipynb` | Inspection of the contrast measurements | – |
| `17_vims_band_spectra.ipynb`, `18_vims_ansa_comparison.ipynb` | VIMS background spectra per ansa (**under revision**, see below) | – |
| `20_build_data_release.py` | Builds the Zenodo data deposit: every image pair with metadata, geometry, selection flags and results, and the figure tables | `release/zenodo_data/` |

## Article figures (`figure_scripts/`)

| Figure | File | Script |
|---|---|---|
| Observation set mosaic | `obs_set_mosaic_163EN.png` | `make_obs_set_mosaic.py` |
| Synthetic band diagnostic | `synthetic_band_diagnostic_A.png` | `make_synthetic_band_diagnostic.py` |
| FFT pipeline overview | `fft_pipeline_overview.png` | `make_fft_pipeline_overview.py` |
| Zero-padded spectrum | `zp_threepanel.png` | `make_zp_threepanel.py` |
| Detection diagnostics | `bundle_A_positive1.png`, `bundle_A_null.png` | `make_article_bundle.py` |
| Orientation vs scattering geometry | `band_orientation_agreement.png` | `wp2_image_plane_alignment.py`, then `make_band_orientation_agreement.py` |
| Profile extraction | `profile_slice.png` | `make_profile_example.py` |
| Contrast vs geometry | `contrast_trends_14b.png` | `make_contrast_trends_14b.py` |
| Viewing geometry, ansa coverage | `geometry_positives.png`, `ansa_coverage_histogram.png` | `recompute_geometry_spice.py` |
| Appendix B table | – | `make_classified_images_table.py` |

## Known limitations of this release

- Several post-pipeline and figure scripts still contain absolute data paths
  (`/mnt/storage/...`, `/home/arnaudm/...`) from the original machine. Adjust the
  path constants at the top of each script to your data location.
- The residual example and Enceladus disc mask figures were produced
  interactively and have no standalone script.
- The VIMS comparison (notebooks 17 and 18) is being redone and should not be
  used in its current form.
