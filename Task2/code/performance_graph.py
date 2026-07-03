import pandas as pd
import matplotlib.pyplot as plt

# Read comparison file
df = pd.read_csv("report/model_comparison.csv")

# Plot
plt.figure(figsize=(10,6))

plt.plot(df["Model"], df["Accuracy"], marker='o', linewidth=2, label="Accuracy")
plt.plot(df["Model"], df["Precision"], marker='o', linewidth=2, label="Precision")
plt.plot(df["Model"], df["Recall"], marker='o', linewidth=2, label="Recall")
plt.plot(df["Model"], df["F1 Score"], marker='o', linewidth=2, label="F1 Score")

plt.title("Performance Comparison of Sentiment Analysis Models")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0.80,1.00)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("graphs/model_comparison.png")

plt.show()

print("\nPerformance comparison graph saved successfully!")