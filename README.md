# RoCLU: Activation Functions Comparison

Official companion repository for the published paper **“A Novel Activation Function for Enhancing Deep Neural Network Stability and Performance”** and the proposed **RoCLU (Robust Cauchy Linear Unit)** activation function.

**Authors:** Özcan Küçükali, M. Hakan Bozkurt, Esma Ulutaş, Işılay Bozkurt  
**Journal:** *Concurrency and Computation: Practice and Experience*  
**Year / volume / article:** 2026; 38:e70944  
**First published:** 03 September 2026  
**DOI:** https://doi.org/10.1002/cpe.70944  
**Repository:** https://github.com/ozcankucukali/activation-functions-comparison-roclu

## RoCLU activation function

The published RoCLU function is

```text
RoCLU(x) = 1.444 * x * ln(1.5 + atan(x) / pi)
```

Equivalently,

```text
g(x) = 1.444 * ln(1.5 + atan(x) / pi)
RoCLU(x) = x * g(x)
```

The paper motivates the arctangent term from the heavy-tailed Cauchy distribution and applies a logarithmic transformation to the gate. The default scaling factor is **k = 1.444**. RoCLU has no trainable activation parameters by default; `k` may be tuned when desired.

PyTorch implementation:

```python
import math
import torch
import torch.nn as nn

class RoCLU(nn.Module):
    def __init__(self, k=1.444):
        super().__init__()
        self.k = k

    def forward(self, x):
        return self.k * x * torch.log(1.5 + torch.atan(x) / math.pi)
```

## Repository structure

```text
activation-functions-comparison-roclu/
├── activations/
│   ├── __init__.py
│   ├── custom_activations.py       # RoCLU implementation
│   └── standard_activations.py     # ReLU, LeakyReLU, ELU, GELU, Mish, Swish
├── datasets/
│   ├── __init__.py
│   ├── cifar10_loader.py
│   └── cifar100_loader.py
├── models/
│   ├── __init__.py
│   ├── resnet50_custom.py
│   ├── senet18_custom.py
│   ├── googlenet_custom.py
│   ├── vgg16_custom.py
│   ├── densenet121_custom.py
│   └── mobilenet_custom.py
├── results/
├── train_and_eval.py               # CIFAR-10/CIFAR-100 CNN comparisons
├── gradient_stability_analysis.py  # PyTorch companion stability implementation
├── requirements.txt
├── CITATION.cff
└── README.md
```

## Included activation functions

- ReLU
- Leaky ReLU
- ELU
- GELU
- Mish
- Swish
- **RoCLU**

Command-line names:

```text
relu, leaky_relu, elu, gelu, mish, swish, roclu
```

## CNN architectures

The published CIFAR-10/CIFAR-100 comparison evaluates:

- ResNet50
- VGG16
- GoogLeNet
- MobileNet
- SENet18
- DenseNet121

The codebase also retains a few compatible architecture variants inherited from the benchmark implementation.

## Datasets

- **CIFAR-10** — 10 classes, 32×32 RGB images
- **CIFAR-100** — 100 classes, 32×32 RGB images

Datasets are downloaded automatically by `torchvision`.

## Installation

```bash
git clone https://github.com/ozcankucukali/activation-functions-comparison-roclu.git
cd activation-functions-comparison-roclu
pip install -r requirements.txt
```

## Usage

### Quick RoCLU test

```bash
python train_and_eval.py \
  --dataset cifar10 \
  --models resnet50 \
  --activations roclu \
  --epochs 5 \
  --num_trials 1
```

### Published baseline comparison set

```bash
python train_and_eval.py \
  --dataset cifar10 \
  --models resnet50 vgg16 googlenet mobilenet_v1 senet18 densenet121 \
  --activations relu leaky_relu elu gelu mish swish roclu \
  --epochs 100 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --num_trials 5
```

### CIFAR-100

```bash
python train_and_eval.py \
  --dataset cifar100 \
  --models resnet50 vgg16 googlenet mobilenet_v1 senet18 densenet121 \
  --activations relu leaky_relu elu gelu mish swish roclu \
  --epochs 100 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --num_trials 5
```

## Published CIFAR experiment settings

The paper reports the following common configuration for the six CNN architectures:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Epochs | 100 |
| Batch size | 128 |
| Independent trials | 5 |
| Scheduler | MultiStepLR |
| LR milestone | 80 |
| Gamma | 0.1 |

## Numerical stability analysis

The published paper reports a separate numerical-stability experiment on CIFAR-10 using **TensorFlow/Keras**. Gradients were monitored every 30 mini-batches, MaxPooling was applied after every five convolutional layers, Batch Normalization was used after convolutional layers, and the first 128 training samples were fixed for gradient evaluation. The paper evaluates depths up to 700 layers.

This repository was adapted from an earlier activation-comparison codebase. Therefore, `gradient_stability_analysis.py` is retained as a **PyTorch companion/reimplementation** of the same general analysis idea; it should not be described as the exact TensorFlow/Keras script used to generate the published stability table.

Example:

```bash
python gradient_stability_analysis.py --layers 50 100 300 500 700
```

## Output

`train_and_eval.py` writes experiment summaries as CSV files into `results/`, for example:

```text
results/results_cifar10_resnet50_relu-roclu.csv
```

## Background code

The CNN comparison structure is based on the open-source benchmark framework associated with:

S. R. Dubey, S. K. Singh, and B. B. Chaudhuri, “Activation functions in deep learning: A comprehensive survey and benchmark,” 2022.  
https://arxiv.org/abs/2109.14545  
Original repository: https://github.com/shivram1987/ActivationFunctions

## Citation

If you use RoCLU or this repository, please cite:

```bibtex
@article{kucukali2026roclu,
  title   = {A Novel Activation Function for Enhancing Deep Neural Network Stability and Performance},
  author  = {Küçükali, Özcan and Bozkurt, M. Hakan and Ulutaş, Esma and Bozkurt, Işılay},
  journal = {Concurrency and Computation: Practice and Experience},
  volume  = {38},
  pages   = {e70944},
  year    = {2026},
  doi     = {10.1002/cpe.70944},
  url     = {https://doi.org/10.1002/cpe.70944}
}
```

## Article scope

The published study additionally evaluates RoCLU in learning-rate sensitivity experiments, timing analyses, and YOLOv8-based object detection, segmentation, pose-estimation, and oriented-bounding-box tasks. Those experiments are described in the paper; this repository package focuses on the activation implementation and the CNN comparison code adapted from the earlier comparison repository.
