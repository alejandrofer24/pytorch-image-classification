# Neural Networks in PyTorch: Tabular and Image Classification

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Two neural-network classifiers built from scratch in **PyTorch**, each accompanied by a systematic hyperparameter study and a written analysis:

| Project | Task | Model | Result |
|---|---|---|---|
| **ASL alphabet recognition** | 26-class image classification of hand signs | 3-block CNN (128-256-512 filters) | **99.72 % accuracy** |
| **Customer segmentation** | 4-class tabular classification | Fully-connected network (16-8 hidden units) | 48.85 % accuracy (random baseline: 25 %) |

The complete write-up is in [`docs/neural_networks_report.pdf`](docs/neural_networks_report.pdf).

## 1. CNN for sign-language letters

**Problem.** Given a 28×28 RGB image of a hand sign, predict one of 26 letters. The network minimises the cross-entropy loss

$$
\mathcal L(\theta)=-\frac1N\sum_{i=1}^{N}\log\frac{e^{z_{i,y_i}}}{\sum_{k=1}^{26}e^{z_{i,k}}} ,
$$

where $z_i$ are the logits for sample $i$, optimised with Adam.

**Baseline architecture** (notebook §3): `Conv(3→32, k=3) → MaxPool(2) → Conv(32→64, k=3) → MaxPool(2) → Flatten(3136) → FC(128) → FC(26)`, images resized to 28×28 and normalised to $[-1,1]$.

**Hyperparameter study.** Learning rate, batch size, number of epochs, number of filters, kernel sizes, activation functions and optimisers were swept one at a time:

| Hyperparameter | Values explored | Best |
|---|---|---|
| Learning rate | 0.01, 0.001, 0.0001 | **0.001** |
| Filters (3 conv layers) | (32,64,128), (64,128,256), (128,256,512) | **(128,256,512)** |
| Kernels | (3,3,3), (5,3,3), (5,5,3), (7,5,3) | **(3,3,3)** |
| Activation | several (see notebook §5) | **ReLU** |
| Batch size | several (see notebook) | **32** |
| Optimiser / loss | Adam, others | **Adam + cross-entropy** |

**Result.** 99.72 % accuracy, fast convergence and no sign of overfitting. Larger or more complex configurations brought no meaningful gain.

## 2. Fully-connected network for customer segmentation

**Problem.** Assign customers of an automobile company to one of four segments (A-D) from demographic and behavioural features (gender, marital status, age, education, profession, work experience, spending score, family size, anonymised category). Pipeline: merge train/test files, handle missing values, one-hot encode categoricals, inspect outliers with box-plots, standardise, train/evaluate.

**Best setup.** Hidden layers of 16 and 8 units, ReLU, Adam (lr = 0.001), batch size 32, 100 epochs, cross-entropy loss.

**Result and analysis.** 48.85 % accuracy (best trial 49.4 %), about twice the random baseline. The confusion matrix shows that **group B is repeatedly confused with A and C**, which, together with the fact that capacity increases did not help, suggests that the limit comes from the information in the features rather than from model size.

## Repository structure

```
pytorch-image-classification/
├── docs/neural_networks_report.pdf
├── notebooks/
│   ├── 01_fully_connected_customer_segmentation.ipynb
│   └── 02_cnn_asl_letter_classification.ipynb
├── data/README.md        # where to place the datasets (not versioned)
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/alejandrofer24/pytorch-image-classification.git
cd pytorch-image-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# add the datasets as described in data/README.md, then:
jupyter notebook notebooks/
```

## Notes

* Training is stochastic, so individual runs vary slightly; the conclusions come from several trials.
* As disclosed in the report, AI assistance was used to write the plotting and `tabulate` summary code for the hyperparameter sweeps.

## Author

**Alejandro Fernández Trillo** · Mathematics graduate · MSc student in Artificial Intelligence
[LinkedIn](https://www.linkedin.com/in/alejandro-fernandez-trillo)
