import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
fake = pd.read_csv("Fake.csv", encoding='latin1')
real = pd.read_csv("True.csv", encoding='latin1')

# Add labels
fake['label'] = 1
real['label'] = 0

# Combine datasets
data = pd.concat([fake, real])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Keep only needed columns
data = data[['text', 'label']]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['text'] = data['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_news(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    if prediction == 1:
        return f"Fake News ❌ (Confidence: {confidence:.2f})"
    else:
        return f"Real News ✅ (Confidence: {confidence:.2f})"

# ===== INPUT PART =====# 
print("\n===== MODEL READY =====\n")

while True:
    user_input = input("Enter news text (or type 'exit'): ")

    if user_input.strip().lower() == "exit":
        print("Exiting...")
        break

    if user_input.strip() == "":
        print("Please enter some text.")
        continue

    result = predict_news(user_input)
    print(result)

# 👇 FORCE PROGRAM TO STAY OPEN
input("\nPress Enter to close...")