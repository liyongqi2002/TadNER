Code and data of our paper "Type-Aware Decomposed Framework for Few-Shot Named Entity Recognition" accepted by Findings of EMNLP 2023.

## Overview

![Framework of TadNER](framework.jpg)


## 1 Quick Start
Here we give an easy example for training and test on Domain-Transfer settings.
### 1.1 Environment

Python=3.7

```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### 1.2 train and test Domain Transfer CoNLL2003

```bash
bash run.sh
```
