import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction
def predict_news(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    if prediction == 1:
        return f"Fake News ❌ (Confidence: {confidence:.2f})"
    else:
        return f"Real News ✅ (Confidence: {confidence:.2f})"

# Input loop
print("\n===== FAKE NEWS DETECTOR =====\n")

while True:
    user_input = input("Enter news text (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    print(predict_news(user_input))