# Cross-Domain Cereal Crop Classification in Algeria
## Using Fourier Domain Adaptation and Swin Transformers

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper:** Cross-Domain Cereal Crop Classification in Algeria using Fourier Domain Adaptation and Swin Transformers

A data-efficient deep learning framework for crop classification in Algeria using Sentinel-2 satellite imagery. We leverage the label-rich **PASTIS dataset** (France) and transfer knowledge to North Africa using **Fourier Domain Adaptation (FDA)** combined with **Swin Transformers**.

## 📌 Key Results

| Method | F1-Score | Notes |
|--------|----------|-------|
| Target Only | 39.2% | Severe overfitting |
| Direct Transfer | 58.7% | Domain shift failure |
| Mixed Batch (No FDA) | 88.3% | Limited by spectral shift |
| MT-ADA (Iftene & Larabi, 2024) | 80.0% | Feature-level alignment |
| **Ours (FDA+Mix)** | **93.2% ± 1.9%** | **State-of-the-Art** |

**Potato Classification (Generalization Test):** 85.6% ± 2.4% F1-Score

---

## 🔬 Methodology Overview

### The Problem
1. **Label Scarcity:** Large labeled datasets are scarce in Algeria compared to Europe
2. **Spectral Domain Shift:** Different atmospheric conditions, sun angles, and soil types
3. **Phenological/Temporal Shift:** Different growing seasons between Europe and North Africa

### Our Solution

![Architecture](architecture.jpg)

Our framework combines three key innovations:

#### 1. Max-NDVI Compression
Instead of using full time-series data (susceptible to phenological shifts), we compute the **Maximum NDVI** across the temporal dimension:

$$\text{Max-NDVI}(x, y) = \max_{t \in T} \left( \frac{NIR_{t} - Red_{t}}{NIR_{t} + Red_{t}} \right)$$

This creates a time-invariant feature that captures peak vegetative vigor regardless of when it occurred.

#### 2. Fourier Domain Adaptation (FDA)
FDA transfers the "style" (spectral characteristics) of Algerian images to European images while preserving their "content" (field boundaries, structures):

```
Source Image (Europe) + Target Style (Algeria) → Adapted Image
```

![FDA Visualization](fda.png)

The process:
- Apply FFT to decompose images into Amplitude (style) and Phase (content)
- Swap low-frequency amplitudes (controlled by β parameter)
- Reconstruct via inverse FFT

#### 3. Mixed-Batch Training Strategy
Train on 50% real Algerian samples + 50% FDA-adapted European samples to stabilize learning.

---

## 📁 Project Structure

```
crop2/
├── 📓 Notebooks (Main Pipeline)
│   ├── data_extraction.ipynb       # Extract Sentinel-2 data from GEE
│   ├── pastis_transformation.ipynb # Process PASTIS → Max-NDVI format
│   ├── fda.ipynb                   # Apply Fourier Domain Adaptation
│   ├── swin.ipynb                  # Train Swin Transformer classifier
│   ├── inference.ipynb             # Run inference on new data
│   └── inference_for_potatoes.ipynb # Potato crop classification
│
├── 📄 Paper
│   ├── main.tex                    # LaTeX source
│   └── research_paper_draft.md     # Draft notes
│
├── 📊 Data Directories
│   ├── PASTIS/                     # European source dataset
│   │   ├── DATA_S2/               # Sentinel-2 patches
│   │   ├── ANNOTATIONS/           # Semantic labels
│   │   └── metadata.geojson       # Acquisition dates
│   │
│   ├── Samples_cereal_In-situ Algerian/
│   │   └── Samples_cereal/        # Ground truth shapefiles
│   │       ├── samples_cereal.shp
│   │       ├── samples_non_cereal.shp
│   │       ├── validation_cereal.shp
│   │       └── validation_non_cereal.shp
│   │
│   └── potato/                    # Potato classification data
│
├── 📦 Output
│   ├── pastis_maxndvi/            # Processed PASTIS (Max-NDVI)
│   ├── algeria_s2_data/           # Extracted Algerian imagery
│   ├── fda_adapted/               # FDA-transformed images
│   ├── swin_cereal_classifier.pth # Trained model weights
│   └── *.json                     # Validation results
│
└── myenv/                         # Python virtual environment
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- PyTorch 2.0+ (CUDA 12.6+ for RTX 40/50 series)
- Google Earth Engine account (for data extraction)

### Installation

```bash
# Clone the repository
git clone https://github.com/Thabetahmed/FDA-Swin-Crop-Classification.git
cd FDA-Swin-Crop-Classification

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# or: myenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
geopandas>=0.13.0
matplotlib>=3.7.0
tqdm>=4.65.0
scikit-learn>=1.2.0
earthengine-api>=0.1.350
geemap>=0.25.0
timm>=0.9.0
```

---

## 📓 Pipeline Walkthrough

### Step 1: Data Extraction (`data_extraction.ipynb`)

Extract Sentinel-2 imagery from Google Earth Engine for Algerian field samples:

```python
# Key parameters
YEAR = 2023
TIME_RANGE = "January - April"  # Growing season
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
PATCH_SIZE = 128  # pixels at 10m resolution
OUTPUT_FORMAT = ".npy"  # NumPy arrays
```

**Output:** `output/algeria_s2_data/DATA_S2/` containing 193 patches (101 train, 92 validation)

### Step 2: PASTIS Processing (`pastis_transformation.ipynb`)

Transform PASTIS time-series data to Max-NDVI format:

```python
# Original PASTIS shape: (43, 10, 128, 128) - 43 timesteps
# Output shape: (1, 10, 128, 128) - Single Max-NDVI composite

def get_max_ndvi_composite(time_series):
    """Select pixels at their maximum NDVI timestamp."""
    # Filter to growing season (Jan-Apr)
    # Compute NDVI for each timestep
    # Select pixel values at max NDVI time
    ...
```

**Output:** `output/pastis_maxndvi/DATA_S2/` containing 2,433 patches

### Step 3: FDA Adaptation (`fda.ipynb`)

Apply Fourier Domain Adaptation to transform European images:

```python
def fda_transfer(source_img, target_img, beta=0.01):
    """
    Transfer low-frequency (style) from target to source.
    
    Args:
        source_img: PASTIS image (C, H, W)
        target_img: Algeria image (C, H, W)
        beta: Size of low-frequency window (0.01-0.05)
    
    Returns:
        Adapted image with Algeria style, Europe structure
    """
    # FFT decomposition
    src_fft = np.fft.fft2(source_img)
    src_amp, src_phase = np.abs(src_fft), np.angle(src_fft)
    
    tgt_fft = np.fft.fft2(target_img)
    tgt_amp = np.abs(tgt_fft)
    
    # Swap low-frequency amplitudes
    mixed_amp = swap_low_freq(src_amp, tgt_amp, beta)
    
    # Reconstruct
    adapted = np.fft.ifft2(mixed_amp * np.exp(1j * src_phase))
    return np.real(adapted)
```

**Key Parameter:** `β = 0.01-0.05` (controls strength of style transfer)

**Output:** `output/fda_adapted/DATA_S2/` containing 2,433 adapted patches

### Step 4: Model Training (`swin.ipynb`)

Train a modified Swin Transformer for binary crop classification:

```python
# Architecture Configuration
CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'epochs': 30,
    'drop_rate': 0.3,
    'img_size': 128,
    'in_channels': 10,      # Sentinel-2 bands
    'num_classes': 1,       # Binary: Cereal vs Non-Cereal
    'fda_ratio': 0.5,       # 50% FDA, 50% Real per batch
}

# Modified Swin Transformer
class SwinCropClassifier(nn.Module):
    def __init__(self, in_channels=10):
        super().__init__()
        # Load pretrained Swin-Tiny
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224')
        
        # Replace first layer for 10-channel input
        self.backbone.patch_embed.proj = nn.Conv2d(
            in_channels, 96, kernel_size=4, stride=4
        )
        
        # Binary classification head
        self.classifier = nn.Linear(768, 1)
```

**Training Strategy:**
- Mixed batches: 50% FDA-adapted PASTIS + 50% real Algerian data
- Weighted sampling to handle class imbalance
- AdamW optimizer with cosine learning rate schedule

### Step 5: Inference (`inference.ipynb`)

Run predictions on new data:

```python
# Load trained model
model = SwinCropClassifier(in_channels=10)
model.load_state_dict(torch.load('output/swin_cereal_classifier.pth'))
model.eval()

# Predict
with torch.no_grad():
    logits = model(image_tensor)
    probability = torch.sigmoid(logits)
    prediction = (probability > 0.5).int()  # 1=Cereal, 0=Non-Cereal
```

---

## 📊 Results & Validation

### Multi-Seed Validation

We validated across 5 random seeds for robust evaluation:

| Seed | Validation F1 | Accuracy |
|------|---------------|----------|
| 42   | 92.8%         | 91.3%    |
| 123  | 94.1%         | 92.6%    |
| 456  | 91.5%         | 90.2%    |
| 789  | 93.6%         | 91.8%    |
| 2024 | 94.0%         | 92.4%    |
| **Mean** | **93.2% ± 1.9%** | **91.7% ± 0.9%** |

### Ablation Study

| Component | F1-Score | Δ |
|-----------|----------|---|
| Full Pipeline (FDA + Mix) | 93.2% | - |
| − FDA (Mix only) | 88.3% | -4.9% |
| − Mix (FDA only) | 85.7% | -7.5% |
| − Both (Direct Transfer) | 58.7% | -34.5% |

---

## 🔧 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1×10⁻⁴ | With cosine decay |
| Weight Decay | 0.05 | L2 regularization |
| Dropout | 0.3 | Prevent overfitting |
| Batch Size | 16 | Mixed: 8 FDA + 8 Real |
| Epochs | 30 | Early stopping at best val F1 |
| FDA β | 0.01-0.05 | Low-frequency window size |
| Optimizer | AdamW | Better than Adam for transformers |

---

## 🛠️ Trained Models

Pre-trained model weights are available in the `output/` directory:

| Model | File | Description |
|-------|------|-------------|
| Final Model | `swin_cereal_classifier.pth` | Best performing model |
| Seed 42 | `swin_model_seed_42.pth` | Reproducibility checkpoint |
| Seed 123 | `swin_model_seed_123.pth` | Reproducibility checkpoint |
| Baseline | `baseline_cereal_classifier.pth` | Without FDA (for comparison) |

---

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@inproceedings{thebat2026crossdomain,
  title={Cross-Domain Cereal Crop Classification in Algeria using Fourier Domain Adaptation and Swin Transformers},
  author={Thebat, Mazouz Ahmed and Sekhsoukh, Hachem Safi Eddine and Guergour, Youcef and Larabi, Mohamed El Amine and Iftene, Meziane},
  booktitle={Proceedings of LNCS Conference},
  year={2026},
  organization={Springer}
}
```

---

## 🤝 Acknowledgments

- **Algerian Space Agency (ASAL)** - Sentinel-2 imagery, ground truth annotations, and computational resources
- **PASTIS Dataset** - [Garnot & Landrieu, 2021](https://github.com/VSainteuf/pastis-benchmark) - European source domain data
- **Swin Transformer** - [Liu et al., 2021](https://github.com/microsoft/Swin-Transformer) - Backbone architecture
- **FDA Method** - [Yang & Soatto, 2020](https://github.com/YanchaoYang/FDA) - Domain adaptation technique

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Ahmed Mazouz Thebat** - ahmed.mazouz@ensia.edu.dz
- **National Higher School of Artificial Intelligence (ENSIA)**, Algiers, Algeria

---

<p align="center">
  <i>Developed at ENSIA in collaboration with the Algerian Space Agency (ASAL)</i>
</p>
