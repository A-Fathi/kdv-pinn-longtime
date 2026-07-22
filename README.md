# Long-Time KdV Soliton Propagation Using Co-Moving Conservation-Regularized Physics-Informed Neural Networks

This repository contains the Google Colab/Python workflow associated with the manuscript
**“Long-Time KdV Soliton Propagation Using Co-Moving Conservation-Regularized Physics-Informed Neural Networks.”**

The repository provides two distinct routes:

1. **Fast publication-figure reproduction** from deposited numerical archives. This is the recommended reviewer route and does not retrain the neural networks.
2. **Optional full retraining** of the matched laboratory-frame and co-moving-frame Stage 1/Stage 2 PINN configurations.

The work uses a standard co-moving coordinate transformation, a PDE-dominant Stage 1, and conservation-aware Stage 2 refinement. The co-moving transformation reduces the transport/phase burden, while Stage 2 primarily improves invariant drift after co-moving stabilization. This repository does not claim a new neural-network architecture.

## Repository contents

```text
kdv-pinn-longtime/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── config/
│   └── paper_config.json
├── notebooks/
│   └── KdV_longtime_publication.ipynb
├── data/
│   └── README.md
├── release/
│   ├── environment.json
│   └── summary_metrics.csv
└── scripts/
    └── verify_dataset.py
```

Large numerical archives and PyTorch model files are deposited separately on Zenodo and are intentionally excluded from GitHub.

## Fast reproduction in Google Colab

### 1. Create the project directory in Google Drive

Create this exact folder:

```text
My Drive/KdV_longtime_publication/
```

The publication notebook is deliberately configured to use:

```text
/content/drive/MyDrive/KdV_longtime_publication
```

### 2. Download the associated Zenodo Dataset

Dataset DOI:

```text
REPLACE_WITH_RESERVED_DATASET_DOI
```

From the dataset record, download:

- `processed_numerical_archives.zip`;
- `trained_models.zip`;
- `figure_source_data.zip`.

Upload the three ZIP files to `My Drive/KdV_longtime_publication/` and extract them there. Each ZIP already contains the correct `data/...` directory hierarchy.

The result should include:

```text
KdV_longtime_publication/
└── data/
    ├── processed/
    │   ├── LAB_E1_core_archive.npz
    │   ├── COMOV_E1_core_archive.npz
    │   └── COMOV_E1_residual_diag.npz
    ├── models/
    │   ├── LAB_E1_model_bundle.npz
    │   ├── COMOV_E1_model_bundle.npz
    │   ├── LAB_E1_net1_stage1.pt
    │   ├── LAB_E1_net2_stage2.pt
    │   ├── COMOV_E1_net1_stage1.pt
    │   └── COMOV_E1_net2_stage2.pt
    └── source_data/
        └── nine source-data files for the main and supplementary figures
```

### 3. Open the notebook in Colab

Open `notebooks/KdV_longtime_publication.ipynb` in Google Colab. Run only the following sections for the fast route:

1. Setup;
2. Frozen paper configuration;
3. Data inventory check;
4. Figure 1;
5. Figures 2–6 and Supplementary Figures S1–S3;
6. Release manifest and headline-metric verification.

**Do not use “Run all”** unless full retraining is intended, because the notebook also contains two computationally expensive optional training cells.

## Expected headline values

The deposited archives reproduce the following principal values:

| Quantity | Value |
|---|---:|
| Co-moving Stage 2 relative $L^2$ error at $\tau=80$ | $1.5003\times10^{-3}$ |
| Co-moving Stage 2 maximum crest drift | $2.1446\times10^{-3}$ |
| Co-moving Stage 2 maximum invariant envelope | $5.6914\times10^{-4}$ |
| Maximum space–time absolute field error | $4.2966\times10^{-4}$ |
| Stage 1 to Stage 2 co-moving field-error improvement factor | $3.70$ |
| Stage 1 to Stage 2 co-moving invariant-envelope improvement factor | $13.5$ |

The complete machine-readable metric table is in `release/summary_metrics.csv`.

## Tested environment

The final figure-reproduction run recorded:

```text
Python       3.12.13
NumPy        2.0.2
SciPy        1.16.3
Matplotlib   3.10.0
PyTorch      2.11.0+cu128
CUDA         12.8
GPU          NVIDIA Tesla T4
```

The code uses a fixed seed of 1234 for the controlled comparison. GPU retraining is not guaranteed to be bitwise identical across different PyTorch, CUDA, driver, and hardware versions. The supplied numerical archives and model weights support figure reproduction without retraining.

## Reproducibility scope

- The ETDRK4 output is described as a **numerical reference solution**, not an exact or formally converged numerical solution.
- Each configuration corresponds to one controlled fixed-seed run; no statistical multi-seed robustness claim is made.
- The public release supports exact reproduction of the deposited plots and metrics from the supplied archives.
- In `Figure6_source_data.npz`, the local PDE contributions are stored as magnitudes under `term_t_magnitude`, `term_nl_magnitude`, and `term_disp_magnitude`.
- In `Supplementary_Figure_S1_source_data.npz`, `Rho_masked` intentionally contains `NaN` values where the field is below the stated masking threshold.

## Verify a downloaded dataset

After extracting the Zenodo data into the project directory, run:

```bash
python scripts/verify_dataset.py --root /path/to/KdV_longtime_publication
```

The script verifies required files, archive schemas, model state dictionaries, and principal published metrics.

## Citation

Citation metadata is provided in `CITATION.cff`. The Zenodo software DOI will be added after GitHub release `v1.0.0` is archived.

## License

The source code is released under the MIT License. The associated dataset is released separately under CC BY 4.0.
