import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_imdb.csv")

# Features and labels
X = df["clean_review"]
y = df["sentiment"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, pos_label="positive")

recall = recall_score(y_test, y_pred, pos_label="positive")

f1 = f1_score(y_test, y_pred, pos_label="positive")

print("\nAccuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

# Classification Report
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# Save Classification Report
with open("report/logistic_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative","Positive"],
    yticklabels=["Negative","Positive"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Logistic Regression Confusion Matrix")

plt.savefig("confusion_matrix/logistic_confusion_matrix.png")

plt.show()

print("\nClassification Report Saved!")

print("Confusion Matrix Saved!")

# Save Logistic Regression Model
joblib.dump(model, "saved_model/logistic_model.pkl")

print("\nLogistic Regression Model Saved Successfully!")