# Dataset Specification & Experimental Protocol

This document defines the exact dataset split, provenance, and specifications used in the project to eliminate ambiguity between training, validation, and benchmarking.

---

## 1. Dataset Architecture & Split

| Split | Source / Benchmark | Image Count | Resolution / Format | Role in Project | Storage Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | BSDS Curated Natural Subset | 35 images (`test_001.png`–`test_035.png`) | $481 \times 321$ / PNG | Decomposed into $64 \times 64$ patches for CNN training | `data/raw/train/` |
| **Validation** | BSDS Curated Natural Subset | 10 images (`test_036.png`–`test_045.png`) | $481 \times 321$ / PNG | Epoch-level validation loss and PSNR checkpoint tracking | `data/raw/val/` |
| **Test** | **Set12 Standard Benchmark** | 12 images (`01.png`–`12.png`) | $256 \times 256$ to $512 \times 512$ / PNG | Full-resolution comparative evaluation against classical filters | `data/test_benchmark/` |

---

## 2. Benchmark Test Set (Set12 Details)

Set12 is a universally recognized standard test benchmark in digital image denoising literature (used by BM3D, DnCNN, FFDNet, etc.):

1. `01.png`: Cameraman ($256 \times 256$) - Sharp silhouettes, smooth sky.
2. `02.png`: House ($256 \times 256$) - Geometric edges, brick textures.
3. `03.png`: Peppers ($256 \times 256$) - Smooth gradients, rounded contours.
4. `04.png`: Starfish ($256 \times 256$) - High-frequency organic texture.
5. `05.png`: Monarch ($256 \times 256$) - High contrast, fine boundary lines.
6. `06.png`: Airplane ($256 \times 256$) - Large homogeneous regions with sharp edges.
7. `07.png`: Parrot ($256 \times 256$) - Detailed feather textures, varying color depth.
8. `08.png`: Lena ($512 \times 512$) - Mixed facial, cloth, and feather frequency details.
9. `09.png`: Barbara ($512 \times 512$) - High-frequency regular patterns (stripe textures).
10. `10.png`: Boat ($512 \times 512$) - Rigging lines, water textures, metal hull surfaces.
11. `11.png`: Man ($512 \times 512$) - Natural facial features, fabric weave.
12. `12.png`: Couple ($512 \times 512$) - Low lighting, subtle structural transitions.

---

## 3. Disjoint Partition Guarantee
- The training split and validation split are mutually exclusive subsets of BSDS.
- The test benchmark (Set12) is completely independent and never exposed to the CNN autoencoder during training or validation hyperparameter tuning.
- Total download footprint is under 2.5 MB, ensuring lightweight execution and fast setup.
