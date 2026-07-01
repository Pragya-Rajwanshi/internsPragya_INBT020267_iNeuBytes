import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("=" * 50)
print("Computer Vision using CNN - Task 1")
print("=" * 50)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("\nDataset Loaded Successfully!")

print(f"Training Images : {x_train.shape}")
print(f"Training Labels : {y_train.shape}")

print(f"Testing Images  : {x_test.shape}")
print(f"Testing Labels  : {y_test.shape}")

print("\nImage Shape:", x_train[0].shape)

print("\nClass Labels Shape:", y_train[0])


# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print("\nAfter Preprocessing")
print("-" * 40)
print(f"Training Images   : {x_train.shape}")
print(f"Validation Images : {x_val.shape}")
print(f"Testing Images    : {x_test.shape}")

print("\nPixel Value Range")
print(f"Minimum : {x_train.min()}")
print(f"Maximum : {x_train.max()}")


# Build Traditional CNN Model


model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu'),

    Flatten(),

    Dense(256, activation='relu'),

    Dense(10, activation='softmax')
])

model.summary()


# Compile Model


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled Successfully!")

# Train Model


history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)

print("\nTraining Completed Successfully!")


# Evaluate Model


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print("\n====================================")
print("Test Accuracy :", round(test_accuracy * 100, 2), "%")
print("Test Loss :", round(test_loss, 4))
print("====================================")


# Save Trained Model


model.save("Task1_CNN/saved_model/cnn_model.keras")

print("\nModel Saved Successfully!")


# Plot Accuracy and Loss Graphs


import matplotlib.pyplot as plt

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("Task1_CNN/graphs/accuracy_graph.png")
plt.close()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("Task1_CNN/graphs/loss_graph.png")
plt.close()

print("\nGraphs Saved Successfully!")


# Generate Predictions


import numpy as np

y_pred = model.predict(x_test)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true = y_test.flatten()

print("\nPredictions Generated Successfully!")

# Confusion Matrix


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("Task1_CNN/confusion_matrix/confusion_matrix.png")
plt.close()

print("\nConfusion Matrix Saved Successfully!")


# Classification Report


from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred_classes)

print("\nClassification Report\n")
print(report)

with open("Task1_CNN/report/classification_report.txt", "w") as f:
    f.write(report)

print("\nClassification Report Saved Successfully!")