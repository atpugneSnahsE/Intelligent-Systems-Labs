# Perceptron-Based Apple–Pear Classification System

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue.svg)
![Status](https://img.shields.io/badge/Project-Working-brightgreen.svg)
![Model](https://img.shields.io/badge/Model-Perceptron-orange.svg)

A simple MATLAB-based binary classifier that distinguishes apples from pears using a **single-layer perceptron** trained on **color** and **roundness** features extracted from fruit images.


## 🧠 Overview

This project implements a perceptron classifier to label fruit images as **apple (+1)** or **pear (−1)**.  
Each image is reduced to a 2D feature vector:

- **Color** (`spalva_color`)
- **Roundness** (`apvalumas_roundness`)

A perceptron is trained on five images and evaluated on a separate test set to measure accuracy and generalization.

---

## 🎨 Feature Extraction

Each image is mapped to:

\[
(x_1, x_2) = (\text{color}, \text{roundness})
\]

### 1. `spalva_color`  
Computes a representative color feature (e.g., mean hue in HSV space).

### 2. `apvalumas_roundness`  
Computes geometric roundness, commonly defined as:

\[
\text{Roundness} = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}
\]

These two features form a discriminative 2D space for the perceptron.

---

## 📘 Training Dataset

Training samples:

| Image | Class | Label |
|-------|--------|--------|
| A1 | Apple | +1 |
| A2 | Apple | +1 |
| A3 | Apple | +1 |
| P1 | Pear  | -1 |
| P2 | Pear  | -1 |

The feature matrix:

\[
P = \begin{bmatrix}
x_1(1) & \dots & x_1(5) \\
x_2(1) & \dots & x_2(5)
\end{bmatrix}
\]

Target vector:

\[
T = [1, 1, 1, -1, -1]^T
\]

---

## 🔢 Perceptron Model

The decision function:

\[
v = w_1x_1 + w_2x_2 + b
\]

Prediction:

\[
y = \text{sign}(v)
\]

Decision boundary:

\[
w_1 x_1 + w_2 x_2 + b = 0
\]

This line separates apples and pears in the 2D feature space.

---

## 🔄 Learning Procedure

Weights and bias are initialized randomly.  
For each sample, the error is computed as:

\[
e(i) = T(i) - y(i)
\]

If misclassified, parameters are updated using:

\[
\begin{aligned}
w_j &\leftarrow w_j + \eta \, e(i) x_j(i) \\
b   &\leftarrow b + \eta \, e(i)
\end{aligned}
\]

Training continues until all samples are correctly classified (guaranteed if linearly separable).

---

## 🧪 Testing Procedure

Test samples:

- Apples: A4–A9  
- Pears: P3, P4  

Accuracy is computed as:

\[
\text{Accuracy} = 1 - \frac{\text{Misclassified Samples}}{N_{\text{test}}}
\]

Outputs include:

- Number of misclassified images  
- Overall test accuracy  
- Final learned weights \( (w_1, w_2, b) \)

---

