# CNN Image Denoising with a Convolutional Autoencoder

A research and demonstration project comparing a compact CNN autoencoder with classical image filters for additive white Gaussian noise (AWGN) removal. The Streamlit app includes single-image inference, batch comparison, and a Set12 research analytics dashboard.

## Features

- CNN autoencoder inference and three classical denoising filters: Gaussian, Median, and Bilateral.
- Single-image and batch image workflows, with original/noisy/denoised comparisons.
- PSNR, SSIM, and latency metrics.
- Set12 benchmark analytics across noise levels sigma 15, 25, and 50.
- CSV downloads for batch metrics and filtered benchmark rows.
- CPU-friendly configuration and repeatable AWGN seed (42).

## Benchmark results

The table below reports the saved Set12 experiment means across all 12 images and all three noise levels. Latency is hardware-dependent and should be treated as a reference for the machine used during evaluation.

| Method | Mean PSNR (dB) | Mean SSIM | Mean latency (ms) |
|---|---:|---:|---:|
| CNN Autoencoder | 28.49 | 0.7770 | 133.86 |
| Gaussian | 25.84 | 0.6824 | 11.04 |
| Median | 24.78 | 0.5858 | 10.90 |
| Bilateral | 24.50 | 0.5709 | 11.33 |

Results by noise level are in [`results/experiments/stride16_epochs30/tables/evaluation_results.md`](results/experiments/stride16_epochs30/tables/evaluation_results.md); per-image values are in the matching CSV. In this run, the CNN has the strongest average PSNR and SSIM, while the classical filters have lower CPU latency.

These are synthetic-noise benchmark results: each clean test image is given AWGN at the selected sigma before denoising. They are not measurements on real noisy photographs. Latency depends on hardware and evaluation conditions.

## Dataset and experimental setup

- **Training:** 35 curated natural images from the BSDS collection, extracted into 64×64 patches using stride 16 for the reported experiment.
- **Validation:** 10 separate natural images.
- **Test:** the 12 standard Set12 images, kept separate from training and validation.
- **Noise:** AWGN at sigma 15, 25, and 50, with seed 42.
- **CNN:** `CompactCNNAutoencoder`, configured for three input and output channels with 32 base channels and skip connections.
- **Filters:** Gaussian (5×5, sigma 1.0), Median (3×3), and Bilateral (diameter 7, color sigma 50, spatial sigma 50).
- **Reported experiment config:** `configs/config_stride16_ep30.yaml`.

Dataset split details are in [`data/DATASET_SPEC.md`](data/DATASET_SPEC.md). The download script retrieves images from the [DnCNN-PyTorch data repository](https://github.com/SaoYan/DnCNN-PyTorch) and has a scikit-image fallback. The fallback is useful for trying the pipeline offline; its sample images are not the Set12 benchmark and will not reproduce the published Set12 result table.

## Setup (Windows PowerShell)

Run these commands from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_dataset.py
```

If the project already has its training and benchmark images, skip the download step. To use the offline sample images instead, run:

```powershell
python scripts/download_dataset.py --offline-only
```

For a real Set12 comparison, ensure `data/test_benchmark/` contains the 12 Set12 PNG files (`01.png` through `12.png`).

## Run the app

From the project folder:

```powershell
streamlit run app/streamlit_app.py
```

Use the **Workflow** selector in the sidebar:

- **Single Image Demo:** upload an image or select a benchmark image, then compare denoising results.
- **Batch Image Processing Studio:** upload PNG/JPG images, process them together, browse the comparison gallery, and download outputs or metrics.
- **Research Analytics:** explore saved Set12 results by noise level, compare method averages, and download filtered CSV rows.

The analytics dashboard reads the first available file from these paths:

1. `results/experiments/stride16_epochs30/tables/evaluation_results.csv`
2. `results/tables/evaluation_results.csv`

## Model checkpoint

The app expects the trained checkpoint at:

```text
results/experiments/stride16_epochs30/checkpoints/best_model.pth
```

Model weight files (`.pth`, `.pt`, and `.onnx`) are excluded by `.gitignore`, so cloning the repository alone will not provide the CNN checkpoint. To distribute a runnable CNN demo, attach the checkpoint to a GitHub Release or provide a trusted download location and explain where to place it. Do not commit a checkpoint until its size, license, and repository hosting limits have been reviewed.

The single-image workflow can still run the classical filters without the checkpoint. Batch processing requires a checkpoint when **CNN Autoencoder** is selected.

## Re-run evaluation

With the Set12 images and checkpoint in place, regenerate the saved experiment results with:

```powershell
python src/evaluate.py `
  --config configs/config_stride16_ep30.yaml `
  --checkpoint results/experiments/stride16_epochs30/checkpoints/best_model.pth `
  --test-dir data/test_benchmark `
  --output-dir results/experiments/stride16_epochs30/tables
```

This evaluates the CNN and classical filters and writes CSV and Markdown tables. It does not retrain the model.

## Tests

Pytest is not currently listed in `requirements.txt`. Install it in the active environment, then run:

```powershell
python -m pip install pytest
pytest -q tests/test_batch_studio.py
pytest -q -p no:cacheprovider
```

## Project layout

```text
app/                      Streamlit application
configs/                  Baseline and stride-16 experiment settings
data/DATASET_SPEC.md      Dataset splits and protocol
data/raw/                 Downloaded training and validation images (ignored)
data/test_benchmark/      Set12 images (ignored)
scripts/                  Dataset download and demo utilities
src/                      Training, model, metrics, filters, and evaluation
results/experiments/      Saved experiment tables, logs, and checkpoints
tests/                    Automated tests
requirements.txt          Runtime dependencies
```

## Reproducibility notes

- The benchmark uses a fixed noise seed and the same noise realization per image and sigma across denoising methods.
- The benchmark dashboard visualizes saved CSV results; it does not run inference when opened.
- Training is not required to launch the app if the checkpoint is already available.
- The default `configs/config.yaml` describes the baseline training setup. The reported 30-epoch, stride-16 experiment uses `configs/config_stride16_ep30.yaml`.
