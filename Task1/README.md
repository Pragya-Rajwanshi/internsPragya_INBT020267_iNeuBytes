# Computer Vision using CNN - CIFAR-10 Image Classification

## Internship

**iNeuBytes Internship**

## Project Overview

This project implements image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNN). Two models were developed:

- Baseline CNN
- Custom CNN

The custom CNN introduces Batch Normalization, Dropout regularization, and a deeper architecture to improve classification performance.

---

## Dataset

Dataset: CIFAR-10

- 50,000 Training Images
- 10,000 Testing Images
- 10 Classes

Classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

---

## Project Structure

```
Task1_CNN/
│
├── architecture/
├── code/
├── confusion_matrix/
├── dataset/
├── graphs/
├── report/
├── saved_model/
├── requirements.txt
└── README.md
```

---

## Baseline CNN Results

- Test Accuracy: **72.02%**
- Test Loss: **1.0048**

---

## Custom CNN Results

- Test Accuracy: **74.43%**
- Test Loss: **0.7492**

---

## Improvements in Custom CNN

- Added Batch Normalization
- Added Dropout layers
- Increased CNN depth
- Tuned Adam optimizer
- Used validation data during training

---

## Generated Outputs

### Baseline

- Accuracy Graph
- Loss Graph
- Confusion Matrix
- Classification Report
- Saved Model

### Custom

- Accuracy Graph
- Loss Graph
- Confusion Matrix
- Classification Report
- Saved Model

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline model:

```bash
python Task1_CNN/code/cnn_baseline.py
```

Run the custom model:

```bash
python Task1_CNN/code/cnn_custom.py
```

---

## Author

**Pragya Rajwanshi**

iNeuBytes Internship Project