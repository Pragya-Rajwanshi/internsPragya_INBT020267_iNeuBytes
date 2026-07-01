# Task 1 Report

## Project Title

Computer Vision using Convolutional Neural Networks (CNN) for CIFAR-10 Image Classification



## Objective

The objective of this project is to develop a Convolutional Neural Network (CNN) capable of classifying images from the CIFAR-10 dataset into ten different object categories. The project demonstrates the implementation of deep learning techniques for image recognition using TensorFlow and Keras.



## Dataset

Dataset Used: CIFAR-10

The dataset contains:

- 50,000 Training Images
- 10,000 Testing Images
- 10 Image Classes

The classes include:

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



## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn



## Methodology

1. Loaded the CIFAR-10 dataset.
2. Normalized image pixel values.
3. Split the dataset into training and validation sets.
4. Designed a CNN architecture using Conv2D, MaxPooling2D, Flatten, and Dense layers.
5. Compiled the model using the Adam optimizer and Sparse Categorical Crossentropy loss.
6. Trained the CNN model for 10 epochs.
7. Evaluated the trained model using the testing dataset.
8. Generated predictions for all test images.
9. Created accuracy and loss graphs.
10. Generated a confusion matrix and classification report.
11. Saved the trained model for future use.



### Baseline CNN
- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Conv2D
- Flatten
- Dense (ReLU)
- Dense (Softmax)

### Custom CNN
- Conv2D (64 Filters)
- Batch Normalization
- MaxPooling2D
- Conv2D (128 Filters)
- Batch Normalization
- MaxPooling2D
- Conv2D (256 Filters)
- Batch Normalization
- MaxPooling2D
- Flatten
- Dropout (0.5)
- Dense (256, ReLU)
- Dropout (0.5)
- Dense (10, Softmax)



## Results

### Baseline CNN
- Test Accuracy: 72.02%
- Test Loss: 1.0048

### Custom CNN
- Test Accuracy: 74.43%
- Test Loss: 0.7492

Additional outputs generated:
- Accuracy Graph
- Loss Graph
- Confusion Matrix
- Classification Report
- Trained Model Saved Successfully



## Conclusion

The baseline CNN achieved a test accuracy of 72.02% on the CIFAR-10 dataset. A custom CNN architecture was then developed by introducing Batch Normalization, Dropout regularization, a deeper convolutional network, and a tuned Adam optimizer. These improvements increased the test accuracy to 74.43% while reducing the test loss. The project successfully demonstrates the complete workflow of image classification using Convolutional Neural Networks, including data preprocessing, model development, training, evaluation, visualization, and model saving.



## Output Files

### Baseline Outputs
- baseline_cnn_model.keras
- baseline_accuracy_graph.png
- baseline_loss_graph.png
- baseline_confusion_matrix.png
- baseline_classification_report.txt

### Custom Outputs
- custom_cnn_model.keras
- custom_accuracy_graph.png
- custom_loss_graph.png
- custom_confusion_matrix.png
- custom_classification_report.txt


## Custom CNN Improvements

The following enhancements were made to the baseline CNN model:

- Increased the number of convolutional filters.
- Added Batch Normalization after convolutional layers.
- Added Dropout layers with a dropout rate of 0.5.
- Tuned the Adam optimizer using a learning rate of 0.0005.
- Used a validation dataset during training.
- Improved feature extraction using a deeper CNN architecture.

These improvements enhanced the model's generalization ability and increased classification performance on the CIFAR-10 dataset.


## Results Comparison

| Metric | Baseline CNN | Custom CNN |
|---------|--------------|------------|
| Test Accuracy | 72.02% | **74.43%** |
| Test Loss | 1.0048 | **0.7492** |
| Macro Precision | 0.73 | **0.75** |
| Macro Recall | 0.72 | **0.74** |
| Macro F1-Score | 0.72 | **0.74** |
| Weighted Precision | 0.73 | **0.75** |
| Weighted Recall | 0.72 | **0.74** |
| Weighted F1-Score | 0.72 | **0.74** |

### Performance Summary

The custom CNN model outperformed the baseline CNN by improving the test accuracy from **72.02%** to **74.43%**, representing an improvement of **2.41%**. The custom model also achieved a lower test loss and better precision, recall, and F1-score. These improvements were achieved by introducing Batch Normalization, Dropout regularization, a deeper convolutional architecture, and a tuned Adam optimizer.

## Author

Pragya Rajwanshi

iNeuBytes Internship Project