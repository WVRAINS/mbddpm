# mbddpm


[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.inffus.2025.103880-orange)](https://doi.org/10.1016/j.inffus.2025.103880)

**MB-DDPM: A reproducible Python package for diffusion-based microbiome data simulation**

MB-DDPM is a reproducible Python package for generating synthetic microbiome data using diffusion models. It supports CSV-based datasets, YAML configuration, GPU acceleration, command-line execution, and reproducible experiment management.

---

## ✨ Features

- Train a UNet-based diffusion model on microbiome datasets  
- Generate synthetic samples via DDPM sampling  
- Unified **CLI interface** (`mbddpm train / sample`)  
- Supports both **CPU** and **GPU**  
- Automatic data reshaping for model input  
- Structured output for reproducible experiments (`checkpoint/`, `generated/`, experiment records)
- Automatic checkpoint saving and metadata management
- Reproducible training and sampling through YAML configuration
---

## 🚀 Quick Start (Recommended)

## Demo Dataset

A toy dataset is provided:

data/demo_case_first10.csv

This dataset allows users to test training and sampling workflows quickly.

```bash
# Train
mbddpm train configs/config.yaml

# Generate samples
mbddpm sample configs/config.yaml
```
---

## 📦 Installation

Requirements:

- Python >=3.9
- PyTorch >=2.0
- CUDA-enabled GPU is recommended for training

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
einops>=0.7
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
  name: dataset_case
  dataset: "./data/demo_case.csv"
  seed: 42

data:
  batch_size: 16  # 16

model:
  num_time_steps: 1000  # 1000
  add_method: code  # code

training:
  num_epochs: 200000 # 
  lr: 0.00001  # 0.00001
  ema_decay: 0.9999 # 0.9999
  save_epoch: 50000 # num_epochs

sampling:
  checkpoint: "./checkpoint/dataset_case/epoch_200000_dataset_case_code.pt"
  generate_num: 1000  # 1000

device: "cuda:0"
```
## 🧪 Training(CLI)
```bash
mbddpm train configs/config.yaml
```
## 🎲 Sampling(CLI)
```bash
mbddpm sample configs/config.yaml
```
--The checkpoint path and generation number are specified in the YAML configuration file.

## 📁 Output Structure

```text
checkpoint/
├── <experiment_name>/
│   └── epoch_XXX_<experiment_name>_<method>.pt


generated/
├── <experiment_name>/
│   └── <samples>.csv
│── experiment_record.xlsx
```

## 🧠 Python API
Training
```bash
from mbddpm.api import train_model

train_model(
    data="data/demo_case_first10.csv",
    data_name="demo_case",
    num_epochs=20000,
    batch_size=16,
    device="cuda",
)
```
Sampling

```bash
from mbddpm.api import generate_samples

samples = generate_samples(
    checkpoint_path="checkpoint/demo_case/epoch_100_demo_case_code.pt",
    generate_num=1000,
    device="cuda",
)
```

## 🐳 Docker Support

A Dockerfile is provided to create a reproducible execution environment based on PyTorch and CUDA.

Build:

```bash
docker build -t mbddpm .

docker run --gpus all -it mbddpm

mbddpm train configs/config.yaml

mbddpm sample configs/config.yaml
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
Experiments are controlled by YAML configuration files, including:

- random seed
- dataset path
- training parameters
- diffusion settings
- sampling parameters

Checkpoints store:

- model weights
- optimizer state
- EMA parameters
- data shape
- feature information
- experiment metadata

## 📖 Associated publication
This repository accompanies the following publication:
Huo B., Tan Y., Zhu R., Li D., Chen Y., Yan S., Sun H.
MB-DDPM: microbiome simulation based on the denoising diffusion probabilistic model.
Information Fusion, 2025.
DOI: https://doi.org/10.1016/j.inffus.2025.103880

The publication introduces a denoising diffusion probabilistic model for realistic microbiome data simulation, while this repository provides an open-source and reproducible implementation of the framework.

## 📖 Citation
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

## 🌍 Scientific Impact

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

## ⚖️ License
MB-DDPM is released under the MIT License.
See LICENSE for details.