import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords

# Download stopwords (runs only once)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/IMDB Dataset.csv")

print("Dataset Loaded Successfully!")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = text.strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
df["clean_review"] = df["review"].apply(clean_text)

print("\nOriginal Review:\n")
print(df["review"][0])

print("\nCleaned Review:\n")
print(df["clean_review"][0])

# Save cleaned dataset
df.to_csv("dataset/cleaned_imdb.csv", index=False)

print("\nCleaned dataset saved successfully!")