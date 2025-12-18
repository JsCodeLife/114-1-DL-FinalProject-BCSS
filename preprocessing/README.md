# BCSS Data Preprocessing Pipeline

This repository contains the preprocessing pipeline for the BCSS histology dataset. It provides tools for Stain Normalization and Contrast Enhancement to prepare images for deep learning models.

## Features
1.  **Macenko Stain Normalization:** Standardizes color distribution across images using a reference slide (`BCSS/train` first image) to reduce batch effects.
2.  **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances local contrast in the LAB color space (L-channel) to improve texture details.
3.  **Dataset Statistics:** Automatically computes and saves RGB mean/std for the processed training set.

## Installation

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Batch Processing (Recommended)
Use `batch_process.py` to process the entire dataset structure.

```bash
python batch_process.py --data_root /path/to/dataset --workers 4
```

*   `--data_root`: Path to the directory containing `BCSS` and `BCSS_512` folders.
*   `--out_dir`: (Optional) Name of the output directory (default: `preprocess_data`).
*   `--workers`: (Optional) Number of parallel processes (default: 4).

### Directory Structure Expectation
The script expects the following source structure under `--data_root`:
```
data_root/
├── BCSS/
│   ├── train/
│   ├── val/
│   └── test/
└── BCSS_512/
    ├── train_512/
    └── val_512/
```

Processed data will be saved to: `data_root/preprocess_data/`.

## Output & Integration

### Output Files
*   **Processed Images:** Saved in the output directory preserving the split structure.
*   **Statistics:** `bcss_stats.json` and `bcss_512_stats.json` containing `mean_rgb` and `std_rgb`.

### PyTorch Integration
Use the generated measurement statistics (`mean_rgb` and `std_rgb` from the JSON files) in your training transform pipeline:

```python
import json
import torchvision.transforms as transforms

# Load stats (example)
with open("path/to/bcss_stats.json", "r") as f:
    stats = json.load(f)

mean = stats["mean_rgb"]
std = stats["std_rgb"]

# Training Transform
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Converts to [0, 1]
    transforms.Normalize(mean=mean, std=std) # Normalizes using calculated stats
])
```

## Methodology Breakdown

### 1. Macenko Stain Normalization
**Why:**
*   **Eliminates Batch Effects:** Removes color variations caused by different staining protocols and scanners.
*   **Standardization:** Ensures consistent feature distribution for stable model training.

**How:**
*   **Decomposition:** Decomposes RGB signals into Hematoxylin & Eosin stain vectors using SVD.
*   **Reconstruction:** Projects all images to the color space of a single reference image to enforce global consistency.

### 2. CLAHE
**Why:**
*   **Enhances Details:** Improves local contrast to highlight cell boundaries and fine structures without amplifying background noise.

**How:**
*   **LAB Space Processing:** Applied only to the **L-channel (Lightness)** to preserve color fidelity.
*   **Adaptive Grid:** Computed on 8x8 tiles with a Clip Limit of 2.0 to prevent over-saturation.
