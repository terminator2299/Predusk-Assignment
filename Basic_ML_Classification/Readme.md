# Basic ML Classification (Iris Dataset) 🌸

A simple neural network built using PyTorch from scratch to classify Iris flower species into one of three classes.

## 🚀 Overview

This project demonstrates:
- Manual data loading & preprocessing
- A basic two-layer neural network (PyTorch)
- Training loop with SGD & Cross-Entropy Loss
- Accuracy tracking and visualization over 50 epochs

---

## 🗂️ Dataset

- **Source**: `sklearn.datasets.load_iris()`
- **Classes**: Setosa, Versicolor, Virginica
- **Features**: Sepal length/width, Petal length/width

---

## 🛠️ Requirements

Install dependencies (after activating your virtual environment):

```bash
pip install torch torchvision matplotlib scikit-learn pandas

basic-ml-classification/
│
├── main.py               # Training and evaluation logic
├── model.py              # Two-layer neural network
├── utils.py              # Data loading and normalization
├── requirements.txt      # Optional: Python packages list
├── accuracy_plot.png     # Saved output plot (generated after running)
└── README.md             # You're reading it!

# Step into the project directory
cd basic-ml-classification

# (Optional) Set up and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # Or install manually

# Run the project
python main.py
