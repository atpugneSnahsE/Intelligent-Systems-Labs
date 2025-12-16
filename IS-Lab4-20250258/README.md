# Handwritten Digit Recognition using RBF Neural Network

## 📋 Project Overview

This project implements a handwritten digit recognition system using **MATLAB** and **Radial Basis Function (RBF) neural networks**. The system recognizes digits (0–9) from scanned images of handwritten text.

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue.svg)
![Status](https://img.shields.io/badge/Project-Working-brightgreen.svg)
![RBF Model](https://img.shields.io/badge/RBF_Model-v1.0-pink)
![MLP Model](https://img.shields.io/badge/MLP_Model-v1.0-blue)

## ✨ Features

* **Image Preprocessing**: Noise removal, binarization, and segmentation
* **Feature Extraction**: Block density, projection profiles, and structural features
* **Neural Network**: RBF network with tunable parameters
* **Visualization**: Display of original images and recognition results
* **Robustness**: Handles handwriting and image-quality variations

## 🛠️ Technical Implementation

### Core Functions

#### 1. `helper_func.m`

Feature extraction utility providing:

* Adaptive binarization
* Morphological cleaning
* Row-wise sorting for training samples
* Extraction of a **49-dimensional feature vector**

#### 2. Main Script (`lab4.m`)

* RBF network training and testing
* Visualization of recognition output
* Performance evaluation

### Feature Set (49 Features)

1. **Block Density (35 features)**: 7 × 5 grid pixel density
2. **Horizontal Projection (7 features)**: Row-wise intensity sums
3. **Vertical Projection (5 features)**: Column-wise intensity sums
4. **Structural Features (2 features)**:

   * Euler number
   * Aspect ratio (normalized)

## 📁 Project Structure

```text
handwritten-digit-recognition/
├── lab4.m              # Main execution script using RBF
├── helper_func.m       # Feature extraction function
├── lab4e.m             # MLP-based implementation
├── train_data.png      # Training dataset (digits 0–9)
├── test_1.png          # Test image
├── test_2.png          # Additional test image
└── README.md           # Project documentation
```

## 🚀 Quick Start

### Prerequisites

* MATLAB R2016b or later
* Image Processing Toolbox

### Running the System

#### 1. Prepare Training Data

* Create `train_data.png` with handwritten digits arranged row-wise
* Each row must contain digits **0–9 in order**
* Multiple rows can be used for improved generalization

#### 2. Prepare Test Data

* Create test images containing handwritten digits
* Name files as `test_1.png`, `test_2.png`, etc.

#### 3. Execute

```matlab
% Navigate to the project directory
lab4
```

## ⚙️ Configuration Parameters

### RBF Network Parameters

```matlab
spread = 30;        % Recommended range: 15–40
goal_error = 0.001; % Training error threshold
max_neurons = 150;  % Maximum hidden neurons
```

### Image Processing Parameters

* Adaptive binarization sensitivity: **0.5**
* Noise removal threshold: **150 pixels**
* Row grouping threshold: **35 pixels (Y-distance)**

## 📊 Performance Tuning

### If Accuracy Is Low

1. **Adjust Spread Parameter**

```matlab
spread = 15;
```

2. **Validate Training Data**

   * Clear, non-overlapping digits
   * Correct digit ordering
3. **Refine Preprocessing**

   * Modify `imbinarize` sensitivity
   * Tune morphological operations

## 📝 Usage Examples

### Training

```matlab
features_train = helper_func('train_data.png', 10, 'train');
P = cell2mat(features_train);
T = create_target_matrix(size(P,2));
net = newrb(P, T, 0.001, 30, 150);
```

### Testing

```matlab
features_test = helper_func('test_image.png', 1, 'test');
P_test = cell2mat(features_test);
Y = sim(net, P_test);
[~, detected] = max(Y);
```

## 🔍 Methodology

### 1. Image Preprocessing

* Grayscale conversion
* Adaptive binarization
* Morphological cleaning
* Connected component analysis

### 2. Feature Extraction

Each digit is resized to **70 × 50 pixels** and represented using 49 features derived from spatial density, projections, and geometry.

### 3. Neural Network Processing

* RBF network learns feature-to-digit mapping
* Outputs class confidence scores
* Maximum response determines predicted digit

### 4. Visualization

* Side-by-side display of test image and recognized digits
* Clean textual output formatting

## 🎯 Possible Applications

* Handwritten form digitization
* Bank cheque processing
* Historical document analysis
* Educational handwriting assessment

## ⚠️ Limitations

* Requires structured training data
* Sensitive to handwriting clarity
* Limited robustness to cursive or stylized digits
* Single-digit recognition only

## 🔮 Future Enhancements

1. Connected digit segmentation
2. Online and incremental learning
3. Confidence estimation
4. Graphical user interface (GUI)
5. Multilingual symbol support

## 👥 Contributors

* Developed as part of a **Intelligent System** course project
* Based on instructor-provided letter recognition code
* Enhanced preprocessing and feature extraction

