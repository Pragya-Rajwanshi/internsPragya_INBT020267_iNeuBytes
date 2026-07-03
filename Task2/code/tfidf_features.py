import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_imdb.csv")

# Features and Labels
X = df["clean_review"]
y = df["sentiment"]

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("TF-IDF Vectorization Completed!")

print("\nTraining Samples:", X_train.shape)
print("Testing Samples :", X_test.shape)