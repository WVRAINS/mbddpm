# mbddpm

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

```bash
# Train
mbddpm train data/demo_case_first10.csv configs/default.yaml --device cuda

# Generate samples
mbddpm sample runs/mbddpm_demo/epoch_10000_code.pt --num 10 --device cuda
```
---

## 📦 Installation

Clone the repository and install as a Python package:

```bash
git clone https://github.com/WVRAINS/mbddpm.git
cd mbddpm
pip install -e . -i https://pypi.org/simple
```


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
runs/
  <experiment_name>/
    checkpoints/
      epoch_XXX_code.pt

generated/
  <experiment_name>/
    samples_*.csv

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
    checkpoint_path="runs/my_experiment/checkpoints/epoch_100.pt",
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
Sampler automatically loads:\
data_shape\
num_features\
taxa_list\
from checkpoint\
Generated tensors are automatically:\
flattened\
cropped to match original feature dimension\
num_time_steps (e.g., 1000) refers to diffusion steps, not number of samples\

## 🔁 Reproducibility
Set random seed via CLI (--seed)\
Outputs are organized by experiment name\
Checkpoints include full model state and metadata\

## 📖 References
MB-DDPM: microbiome simulation based on the denoising diffusionprobabilistic model

