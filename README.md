# mbddpm


[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.inffus.2025.103880-orange)](https://doi.org/10.1016/j.inffus.2025.103880)

**MB-DDPM: A reproducible Python package for diffusion-based microbiome data simulation**

MB-DDPM is a lightweight and reproducible framework for generating synthetic microbiome data using diffusion models. It supports CSV-based datasets, YAML configuration, GPU acceleration, and a clean CLI interface.

---

## ✨ Features

- Train a UNet-based diffusion model on microbiome datasets  
- Generate synthetic samples via DDPM sampling  
- Unified **CLI interface** (`mbddpm train / sample`)  
- Supports both **CPU** and **GPU**  
- Automatic data reshaping for model input  
- Structured output for reproducibility (`runs/`, `generated/`)  

---

## 🚀 Quick Start (Recommended)

## Demo Dataset

A toy dataset is provided:

data/demo_case_first20.csv

This dataset allows users to test training and sampling workflows quickly.

```bash
# Train
mbddpm train data\demo_case_first20.csv configs\default.yaml --device cuda

# Generate samples
mbddpm sample runs\mbddpm_demo\epoch_10000_code.pt configs\default.yaml --num 10 --device cuda
```
---

## 📦 Installation

Requirements:

- Python >=3.10
- PyTorch >=2.0

Clone repository:

```bash
git clone https://github.com/WVRAINS/mbddpm.git
cd mbddpm

pip install -e .


## 📚 Dependencies
Installed automatically via pip:
```bash
torch>=2.0
numpy>=1.23
pandas>=1.5
tqdm>=4.65
pyyaml>=6.0
timm>=0.9
```
## 📊 Data Format
Input must be a CSV file:

Shape: (n_samples, n_features)
Column names (optional) are used as taxa_list

Example:
```bash
taxa1,taxa2,taxa3
0.1,0.2,0.3
0.4,0.5,0.6
...
```

## ⚙️ Configuration (YAML)
```bash
experiment:
  name: mbddpm_demo
  seed: 42

data:
  batch_size: 16

model:
  num_time_steps: 1000
  add_method: code

training:
  num_epochs: 10000
  lr: 0.00001
  ema_decay: 0.9999
  save_epoch: 5000

sampling:
  generate_num: 1000

device: cuda
```
## 🧪 Training(CLI)
```bash
mbddpm train data.csv config.yaml --device cuda --seed 42
```
## 🎲 Sampling(CLI)
```bash
mbddpm sample runs/<experiment_name>/checkpoints/epoch_XXX_code.pt --num 10 --device cuda
```
--num: number of generated samples
Output is automatically saved

## 📁 Output Structure

```text
runs/
├── <experiment_name>/
│   └── checkpoints/
│       └── epoch_XXX_code.pt

generated/
└── <experiment_name>/
    └── samples_*.csv
```

## 🧠 Python API
Training
```bash
from mbddpm.api import train_model
from mbddpm.data.csv_dataset import csv_dataset

dataset = csv_dataset("data.csv")

train_model(
    data=dataset.data,
    taxa_list=dataset.taxa_list,
    data_name="my_experiment",
    num_epochs=10000,
    batch_size=16,
    device="cuda",
)
```
Sampling

```bash
from mbddpm.api import generate_samples
from mbddpm.utils.save_sample import save_samples

samples = generate_samples(
    checkpoint_path="runs\my_experiment\checkpoints\epoch_100.pt",
    generate_num=10,
    device="cuda",
)

save_samples(
    samples,
    taxa_list=dataset.taxa_list,
    data_name="my_experiment"
)
```

## 📌 Notes

Sampler automatically loads:

- data_shape
- num_features
- taxa_list

Generated tensors are automatically:

- flattened
- cropped to match original feature dimension

Note:
num_time_steps (e.g.,1000) refers to diffusion steps, not sample count.

## 🔁 Reproducibility
Set random seed via CLI (--seed)\
Outputs are organized by experiment name\
Checkpoints include full model state and metadata\

## 📖 Associated publication
This repository accompanies the following publication:
Huo B., Tan Y., Zhu R., Li D., Chen Y., Yan S., Sun H.
MB-DDPM: microbiome simulation based on the denoising diffusion probabilistic model.
Information Fusion, 2025.
DOI: https://doi.org/10.1016/j.inffus.2025.103880

The publication introduces a denoising diffusion probabilistic model for realistic microbiome data simulation, while this repository provides an open-source and reproducible implementation of the framework.

## Citation
If you use MB-DDPM in your work, please cite:
```bibtex
@article{huo2025mbddpm,
title={MB-DDPM: microbiome simulation based on the denoising diffusion probabilistic model},
author={Huo, B. and Tan, Y. and Zhu, R. and Li, D. and Chen, Y. and Yan, S. and Sun, H.},
journal={Information Fusion},
year={2025},
doi={10.1016/j.inffus.2025.103880}
}
```

## Scientific Impact

MB-DDPM provides the official open-source implementation of the framework
presented in:

Huo et al., Information Fusion (2025).

The software was used to generate the experimental results reported in the
publication and enables reproducible microbiome simulation research based on
denoising diffusion probabilistic models.

Potential applications include:

- microbiome benchmarking
- synthetic cohort generation
- privacy-preserving data sharing
- machine learning model evaluation
- microbiome data augmentation

## License
MB-DDPM is released under the MIT License.
See LICENSE for details.