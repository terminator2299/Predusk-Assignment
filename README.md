# 📝 Mini Report – Predusk Assignment

**Author:** Bhavya Khandelwal  
**Date:** July 2025

---

## 🔶 Part 1: ML Classification – Iris Dataset

### ✅ Objective
Build and train a basic neural network classifier using PyTorch to classify the Iris dataset into three flower species.

### 🧠 Model Architecture
- Input Layer: 4 features
- Hidden Layer: 10 neurons with ReLU activation
- Output Layer: 3 classes (softmax via CrossEntropyLoss)

### ⚙️ Hyperparameters
- Optimizer: SGD (`lr=0.01`)
- Loss: CrossEntropyLoss
- Epochs: 50
- Batch Size: Full batch (no minibatching)
- Normalization: StandardScaler on input features

### 📊 Final Performance
- **Train Accuracy:** ~96.67%
- **Test Accuracy:** ~93.33%

### 🔍 Interpretation
The model learned quickly and converged within the first 20–30 epochs. Accuracy remained stable with no overfitting due to the simplicity of the dataset and model. The small model was effective because the Iris dataset is linearly separable with clean features.

---

## 🔷 Part 2: GenAI Text Generation – GPT-2

### ✅ Objective
Generate text using a pre-trained GPT-2 model with top-k sampling. Compare output quality using two different temperatures.

### ⚙️ Settings
- Model: `gpt2` (small)
- Max tokens: 50
- Top-k: 50
- Temperatures: **0.7** and **1.0**

### 📄 Generated Samples (Excerpt)

**Temperature = 0.7**

Once upon a time there was a little village nestled in the hills. The people were kind and peaceful...


**Temperature = 1.0**

Once upon a time a unicorn hacker surfed the deep web to decode messages from interstellar beings...


### 🧠 Observations
- **Temperature 0.7**: Output was more logical and coherent, with smoother transitions and consistent style.
- **Temperature 1.0**: More imaginative and unpredictable output, but sometimes less grammatically sound or coherent.
- **Conclusion**: Lower temperatures improve coherence; higher temperatures increase creativity at the cost of structure.

---

## 📚 Key Learnings

### 🔬 ML Classification
- Learned how to build a full training pipeline from scratch.
- Understood the importance of feature normalization and tuning basic hyperparameters.

### 💡 GenAI Experiment
- Got hands-on experience with `transformers` and generation parameters like `temperature` and `top-k`.
- Observed first-hand how temperature drastically alters the nature of generated text.

### 🔧 Challenges
- Handling PyTorch warnings related to missing attention masks and pad token configuration.
- Debugging large file issues when accidentally pushing the virtual environment to GitHub.

---

## ✅ Conclusion

This assignment was a great blend of fundamentals (ML classification) and creativity (text generation). It helped reinforce practical PyTorch skills and gave valuable exposure to large language models through Hugging Face.
