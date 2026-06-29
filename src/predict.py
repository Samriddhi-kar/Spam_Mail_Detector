import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

stemmer = PorterStemmer()

stop_words = set(stopwords.words("english"))

important_words = {
    "free",
    "win",
    "won",
    "winner",
    "claim",
    "prize",
    "cash",
    "gift",
    "urgent",
    "call",
    "now"
}

stop_words = stop_words - important_words

def preprocess_text(text):

    text = text.lower()

    words = word_tokenize(text)

    cleaned_words = []

    for word in words:

        if word not in string.punctuation and word not in stop_words:

            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)

current_dir = os.path.dirname(__file__)

model_path = os.path.join(current_dir, "..", "models", "spam_model.pkl")

vectorizer_path = os.path.join(current_dir, "..", "models", "vectorizer.pkl")

model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)

print("\n========== SPAM DETECTOR ==========")
print("Type 'exit' to quit.\n")

while True:

    user_message = input("Enter a message: ")

    if user_message.lower() == "exit":
        print("\nThank you for using Spam Mail Detector!")
        break

    processed_message = preprocess_text(user_message)

    message_vector = vectorizer.transform([processed_message])

    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        print("Prediction: SPAM\n")
    else:
        print("Prediction: HAM\n")