# 🩻 Self-Supervised Learning for Medical Imaging

### Contrastive Representation Learning for Chest X-Ray Classification

This project explores whether **self-supervised representation learning** can reduce dependence on labeled medical images for chest X-ray classification.

Using a **SimCLR-style contrastive-learning pipeline with a ResNet50 encoder**, the project compares representations learned from unlabeled chest X-rays against conventional ImageNet pre-training and random initialization.

The experiments investigate:

* Self-supervised pre-training
* Label efficiency
* Linear probing vs. fine-tuning
* Multi-label chest X-ray classification
* Grad-CAM interpretability
* UMAP embedding analysis
* Cross-dataset transfer from **CheXpert → NIH ChestX-ray14**

> **Research disclaimer:** This project is intended for research and educational purposes only. The models are not clinically validated and must not be used for medical diagnosis or decision-making.

---

## 🎯 Research Questions

The project explores four main questions:

### 1. Can self-supervised learning capture useful chest X-ray representations?

A SimCLR-style model is trained to learn representations without using disease labels during representation learning.

### 2. Does SSL help when labeled data are scarce?

Models initialized from different pre-training strategies are compared while reducing the available labeled training data.

### 3. What representations do the models learn?

Grad-CAM and UMAP are used to investigate model attention and embedding structure.

### 4. Can learned representations transfer across datasets?

The project includes experiments transferring representations learned from **CheXpert** to **NIH ChestX-ray14**.

---

# 🧠 Approach

The overall workflow is:

```text
Unlabeled Chest X-rays
        ↓
Medical-Aware Augmentations
        ↓
Two Augmented Views
        ↓
ResNet50 Encoder
        ↓
Projection Head
        ↓
Contrastive Learning
        ↓
Learned Representation
        ↓
┌──────────────────────────────┐
│ Linear Probe / Fine-Tuning   │
└──────────────────────────────┘
        ↓
Multi-Label Classification
        ↓
AUC Evaluation
        ↓
Grad-CAM / UMAP Analysis
```

---

# 🔬 Self-Supervised Pre-Training

The SSL pipeline follows a SimCLR-style setup.

For each input image, two augmented views are generated and passed through the same encoder.

```text
Chest X-ray
   │
   ├──── Augmentation A ────► Encoder ────► z₁
   │
   └──── Augmentation B ────► Encoder ────► z₂

                    ↓

          Contrastive Objective
```

The objective encourages representations of two transformations of the same image to remain close while separating representations of different images.

---

## Medical-Aware Augmentations

The augmentation pipeline explores transformations such as:

* Small rotations
* Horizontal flipping
* Brightness and contrast changes
* Gaussian blur
* Random cropping
* Resizing

Medical-image augmentations require more care than ordinary natural-image augmentation because aggressive transformations may alter clinically relevant anatomical information.

---

# 🏗️ Model Architecture

The core representation model uses:

```text
ResNet50
    ↓
2048-D Representation
    ↓
Projection Head
    ↓
128-D Contrastive Embedding
```

The experiments compare four downstream strategies:

| Strategy       | Initialization          | Encoder    |
| -------------- | ----------------------- | ---------- |
| Random Init    | Random weights          | Fine-tuned |
| ImageNet       | ImageNet pre-training   | Fine-tuned |
| SSL Fine-tuned | Self-supervised weights | Fine-tuned |
| SSL Linear     | Self-supervised weights | Frozen     |

This comparison helps separate the quality of the learned representation from improvements introduced by downstream fine-tuning.

---

# 📊 Datasets

## CheXpert

CheXpert provides chest radiographs with multi-label observations for thoracic findings.

The main SSL and label-efficiency experiments use CheXpert data.

---

## NIH ChestX-ray14

NIH ChestX-ray14 is used for cross-dataset transfer experiments.

The transfer pipeline evaluates whether representations learned from CheXpert remain useful after moving to a different chest X-ray dataset and label distribution.

Dataset files are **not distributed with this repository**.

---

# 📈 Label-Efficiency Experiment

A key experiment compares Random, ImageNet, and SSL initialization while reducing the amount of labeled training data.

The saved exploratory runs produced:

| Labeled Fraction | Random Init |   ImageNet | SSL Fine-tuned | SSL Linear |
| ---------------: | ----------: | ---------: | -------------: | ---------: |
|               1% |      0.5009 |     0.4667 |     **0.6844** | **0.6937** |
|               5% |      0.6266 |     0.6359 |     **0.6854** | **0.6931** |
|              10% |      0.6344 |     0.6685 |         0.6798 | **0.6954** |
|              30% |      0.6833 | **0.7053** |         0.6844 |     0.6863 |
|              50% |      0.6978 | **0.7261** |         0.6857 |     0.6439 |

### Key Observation

The strongest SSL advantage appears in the **lowest-label regimes**.

With only 1% of the working labeled subset:

```text
ImageNet AUC       = 0.4667
SSL Fine-tuned AUC = 0.6844
SSL Linear AUC     = 0.6937
```

As more labeled data become available, the ImageNet baseline becomes increasingly competitive and eventually outperforms the SSL variants in these exploratory runs.

This is an important result: the experiments do **not** suggest that SSL is universally better. Instead, they suggest that the learned representation may be most useful when labels are especially limited.

---

## ⚠️ Interpretation of Results

These values come from exploratory subset experiments designed to compare training strategies efficiently.

They should **not** be interpreted as final clinical or state-of-the-art benchmark results.

The experiments use:

* A reduced working subset for rapid comparison
* Limited training epochs
* Restricted validation sampling
* A small number of experimental runs

A stronger evaluation would use repeated runs, confidence intervals, complete validation sets, and a fully controlled experimental protocol.

---

# 🗺️ Representation Analysis

## UMAP

The project extracts learned feature vectors and projects them into two dimensions using **UMAP**.

This makes it possible to visually compare representation structure across:

```text
Random Initialization
ImageNet Pre-training
SSL Fine-tuning
SSL Linear Probing
```

The analysis helps investigate whether different pre-training strategies organize chest X-ray representations differently.

---

# 🔍 Grad-CAM Interpretability

Grad-CAM is used to inspect which image regions influence downstream predictions.

The workflow is:

```text
Chest X-ray
     ↓
Trained Classifier
     ↓
Target Prediction
     ↓
Gradient Extraction
     ↓
Activation Weighting
     ↓
Grad-CAM Heatmap
```

Grad-CAM is used here as an interpretability tool rather than proof that a model is attending to clinically correct anatomy.

Attention maps require careful interpretation and are not themselves evidence of clinical validity.

---

# 🔄 Cross-Dataset Transfer

The project also explores:

```text
CheXpert
    ↓
Learned Representation
    ↓
NIH ChestX-ray14
```

The transfer notebook adapts models trained through the CheXpert experiments for NIH's 14-label classification setting.

This portion of the project remains exploratory.

Some low-data NIH experiments produced unstable AUC estimates, so the repository does **not** present those runs as finalized transfer-learning benchmark results.

The experiment is retained because it demonstrates both the potential and the difficulty of cross-dataset generalization in medical imaging.

---

# 🧪 Evaluation

The primary metric used throughout the classification experiments is:

### ROC-AUC

AUC is calculated per available pathology class and aggregated across valid classes.

Additional analyses include:

* Training/validation loss
* Model comparisons
* Per-class behavior
* Grad-CAM inspection
* Embedding visualization

---

# 📓 Notebooks

The repository is organized into three experiment notebooks:

| Notebook                                     | Purpose                                                                       |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| `01_chexpert_ssl_experiments.ipynb`          | SimCLR/ResNet50 setup, supervised baselines, and label-efficiency experiments |
| `02_model_comparison_interpretability.ipynb` | Model comparison, Grad-CAM, and representation visualization                  |
| `03_chexpert_to_nih_transfer.ipynb`          | Cross-dataset CheXpert → NIH transfer experiments                             |

---

# 📁 Repository Structure

```text
self-supervised-medical-imaging/
│
├── notebooks/
│   ├── 01_chexpert_ssl_experiments.ipynb
│   ├── 02_model_comparison_interpretability.ipynb
│   └── 03_chexpert_to_nih_transfer.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

### Deep Learning

`PyTorch` · `PyTorch Lightning` · `torchvision`

### Models

`ResNet50` · `SimCLR` · `Contrastive Learning`

### Computer Vision

`OpenCV` · `Pillow`

### Evaluation

`scikit-learn` · `ROC-AUC`

### Interpretability

`Grad-CAM` · `UMAP`

### Data & Visualization

`NumPy` · `Pandas` · `Matplotlib` · `Seaborn`

### Infrastructure Used During Experiments

`Google Cloud Storage` · `AWS S3` · `Google Colab`

---

# 🚀 Getting Started

## Clone

```bash
git clone https://github.com/Gravity-2010/self-supervised-medical-imaging.git
cd self-supervised-medical-imaging
```

## Create a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Start Jupyter

```bash
jupyter notebook
```

Then open the notebooks under:

```text
notebooks/
```

Dataset access and cloud-storage configuration are required for reproducing the full experiments.

---

# ⚠️ Limitations

Important limitations include:

### Exploratory Training Regime

Several experiments use reduced subsets and shorter training schedules to make model comparisons computationally feasible.

### Limited Repeated Runs

Results are not reported with multi-seed confidence intervals.

### Dataset Shift

CheXpert and NIH differ in acquisition conditions, patient populations, labeling procedures, and label definitions.

Cross-dataset performance should therefore not be expected to match within-dataset performance.

### Interpretability

Grad-CAM highlights model-sensitive regions but does not prove clinical reasoning or causal relevance.

### No Clinical Validation

The models have not undergone prospective clinical validation and should not be used for diagnosis or medical decision-making.

---

# 🔮 Future Work

Potential extensions include:

* Full-dataset controlled experiments
* Multi-seed evaluation with confidence intervals
* More systematic augmentation ablations
* Improved cross-dataset transfer evaluation
* Additional self-supervised approaches
* Domain-specific contrastive objectives
* Per-pathology AUC analysis
* Calibration analysis
* More extensive Grad-CAM evaluation
* Quantitative embedding analysis
* Reproducible training scripts outside notebooks

---

# 🎓 Project Context

This project explores a central challenge in medical AI:

> **Can useful medical-image representations be learned before expensive labels are available?**

The experiments provided hands-on experience with self-supervised learning, medical-image classification, representation analysis, model interpretability, and domain transfer.

---

## 👩‍💻 Author

**Garvita Jain**

M.S. Computer Science — University of Maryland, Baltimore County

[GitHub](https://github.com/Gravity-2010) · [LinkedIn](https://www.linkedin.com/in/garvitajain-605a89160/)
