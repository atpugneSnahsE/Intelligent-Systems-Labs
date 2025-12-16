# RBF Neural Network Implementation

## Overview

This repository contains two MATLAB implementations of Radial Basis Function (RBF) Neural Networks for function approximation. The first implementation employs fixed RBF parameters with training restricted to the output layer, whereas the second implementation realizes a fully adaptive RBF network in which all parameters are learned via gradient descent.


## Problem Definition

The objective is to approximate the nonlinear function
```
[
y = \frac{1}{2}\left(1 + 0.6\sin\left(\frac{2\pi x}{0.7}\right) + 0.3\sin(2\pi x)\right)
]
```
Training data consist of 22 uniformly sampled points in the interval (x \in [0.1, 1.0]).

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue.svg)
![Status](https://img.shields.io/badge/Project-Working-brightgreen.svg)
![RBF Model](https://img.shields.io/badge/RBF_Model-v1.0-pink)

## Implementation 1: Fixed RBF Network

### Key Features

* Fixed RBF parameters: centers and radii are manually predefined.
* Trainable parameters: output layer weights and bias only.
* Architecture: two Gaussian RBF neurons followed by a linear combiner.
* Optimization: gradient descent on a linear-in-parameters model.

### Architecture

```
Input → [2 Fixed RBF Neurons] → [Linear Combiner] → Output
```

### Mathematical Model

For each RBF neuron:
[
\phi_i(x) = \exp\left(-\frac{(x - c_i)^2}{2 r_i^2}\right)
]

Network output:
[
y_{out} = w_1\phi_1(x) + w_2\phi_2(x) + b
]

### Training

* Method: online gradient descent (output layer only)
* Loss function: mean squared error (MSE)
* Learning rate: (\eta = 0.01)
* Stopping criteria: MSE < (10^{-5}) or 1000 epochs

### Fixed Parameters

```matlab
c1 = 0.19;  r1 = 0.18;
c2 = 0.90;  r2 = 0.17;
```

## Implementation 2: Adaptive RBF Network

### Key Features

* Fully adaptive parameters: centers, radii, weights, and bias.
* Automatic feature learning through gradient-based optimization.
* Separate learning rates for different parameter groups.
* Radius lower bound to ensure numerical stability.

### Architecture

```
Input → [2 Adaptive RBF Neurons] → [Linear Combiner] → Output
```

### Training Algorithm

All parameters are updated using online gradient descent.

| Parameter    | Update Rule                                        | Learning Rate    |
| ------------ | -------------------------------------------------- | ---------------- |
| Weights (wᵢ) | (\Delta w_i = \eta_w e \phi_i)                     | (\eta_w = 0.01)  |
| Bias (b)     | (\Delta b = \eta_w e)                              | (\eta_w = 0.01)  |
| Centers (cᵢ) | (\Delta c_i = \eta_c e w_i \phi_i (x-c_i)/r_i^2)   | (\eta_c = 0.005) |
| Radii (rᵢ)   | (\Delta r_i = \eta_r e w_i \phi_i (x-c_i)^2/r_i^3) | (\eta_r = 0.005) |

### Stability Constraints

```matlab
r_min = 0.05;
max_epochs = 10000;
threshold = 1e-6;
```

## Performance Comparison

| Aspect                        | Fixed RBF | Adaptive RBF |
| ----------------------------- | --------- | ------------ |
| Parameters learned            | 3         | 7            |
| Flexibility                   | Low       | High         |
| Training speed                | Fast      | Slow         |
| Final accuracy                | Good      | Higher       |
| Sensitivity to initialization | High      | Lower        |
| Overfitting risk              | Low       | Moderate     |
| Hyperparameters               | 1         | Multiple     |

## Usage Instructions

### Fixed RBF

```matlab
rbf_fixed
```

### Adaptive RBF

```matlab
rbf_adaptive
```

### Parameter Modification

Fixed RBF:

```matlab
c1 = 0.2;  r1 = 0.2;
c2 = 0.8;  r2 = 0.2;
```

Adaptive RBF:

```matlab
eta_w = 0.02;
eta_c = 0.001;
eta_r = 0.001;
```

## Output Visualization

Both implementations generate:

* Training data and target function plot
* MSE versus epoch curve
* Final function approximation

The adaptive RBF additionally visualizes basis function evolution.

## Technical Details

### RBF Activation Function

Gaussian basis:
[
\phi(x) = \exp\left(-\frac{(x-c)^2}{2r^2}\right)
]

### Error Metric

Mean Squared Error:
[
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - y_{out,i})^2
]

### Convergence Criterion

Training terminates when MSE falls below the defined threshold or when the maximum number of epochs is reached.
