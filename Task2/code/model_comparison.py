import pandas as pd

# Model Performance Comparison
results = {
    "Model": [
        "Logistic Regression",
        "Support Vector Machine",
        "LSTM Neural Network"
    ],

    "Accuracy": [
        0.8869,
        0.8891,
        0.8831
    ],

    "Precision": [
        0.8771,
        0.8835,
        0.8770
    ],

    "Recall": [
        0.9020,
        0.8984,
        0.8932
    ],

    "F1 Score": [
        0.8893,
        0.8909,
        0.8851
    ]
}

df = pd.DataFrame(results)

print("\nModel Comparison\n")
print(df)

df.to_csv("report/model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")