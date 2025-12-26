# Convergence Behavior of Feed-Forward Neural Networks on Imbalanced Data

## 📘 Overview
This repository contains the implementation and analysis for **Mini Project 3** of  
**MATH 5383 – Predictive Analytics**.

The project investigates how the **learning rate** affects the **convergence behavior** of a fixed Feed-Forward Neural Network (FNN) when trained on a **highly imbalanced binary dataset** with an inner–outer clustered structure. Special attention is given to class-wise performance differences between the minority and majority classes.

---

## 🎯 Objectives
- Study the effect of learning rate on training convergence
- Measure convergence speed using a fixed misclassification threshold
- Track **class-wise Mean Squared Error (MSE)** during training
- Analyze bias introduced by class imbalance
- Identify an optimal learning-rate range balancing speed and stability

---

## 📊 Dataset Description
- **Feature Space:** \([0,1] \times [0,1]\)
- **Class 0 (Minority – Inner Cloud):**
  - Samples: 120  
  - Distribution: Bivariate Gaussian  
  - Center: (0.5, 0.5), σ = 0.05
- **Class 1 (Majority – Outer Ring):**
  - Samples: 600  
  - Distribution: Uniform sampling with rejection
- **Imbalance Ratio:** 5:1  
- **Structure:** Non-overlapping concentric geometry

This dataset was designed to clearly expose the effects of class imbalance on neural network training.

---

## 🧠 Model Architecture
A fixed **2–8–4–1 Feed-Forward Neural Network** was used throughout all experiments:

- Input Layer: 2 neurons  
- Hidden Layer 1: 8 neurons (ReLU activation)  
- Hidden Layer 2: 4 neurons (ReLU activation)  
- Output Layer: 1 neuron (Sigmoid activation)  
- **Total trainable parameters:** 52  

**Optimizer:** Adam  
**Loss Function:** Mean Squared Error (MSE)

The architecture was intentionally kept constant so that the learning rate remained the only variable influencing convergence behavior.

---

## ⚙️ Experimental Setup
- **Learning Rates Tested:**  
  \(\{0.001, 0.003, 0.005, 0.007, 0.009, 0.010\}\)
- **Runs per Learning Rate:** 5 (averaged results)
- **Epochs:** Up to 2000
- **Convergence Criterion:**  
  Overall misclassification error ≤ **ϵ = 0.1**
- **Metrics Tracked:**
  - Epochs to convergence
  - Total MSE
  - Class-wise MSE (Minority vs Majority)

---

## 📈 Key Findings
- Increasing the learning rate significantly reduces convergence time
- Fastest convergence observed at **η = 0.010**
- Majority class consistently achieves much lower MSE
- Minority class remains under-optimized across all learning rates
- Demonstrates inherent bias of MSE-based training on imbalanced datasets

These results highlight that faster convergence does not guarantee balanced learning across classes.

---

## 🛠️ Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib

---

## 📂 Repository Structure
