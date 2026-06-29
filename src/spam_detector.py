import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

file_path = os.path.join(current_dir, "..", "dataset", "spam.csv")

df = pd.read_csv(file_path, encoding="latin-1")

print("========== ORIGINAL DATA ==========")
print(df.head())

df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\n========== LABELS CONVERTED ==========")
print(df.head())

print("\n========== LABEL COUNT ==========")
print(df["label"].value_counts())

print("\n========== CLEANED DATA ==========")
print(df.head())

print("\n========== NEW SHAPE ==========")
print(df.shape)

print("\n========== NEW COLUMN NAMES ==========")
print(df.columns)

df["message"] = df["message"].apply(preprocess_text)

print("\n========== PREPROCESSED DATA ==========")
print(df.head())

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000
)

X = vectorizer.fit_transform(df["message"])

model_folder = os.path.join(current_dir, "..", "models")
os.makedirs(model_folder, exist_ok=True)

joblib.dump(
    vectorizer,
    os.path.join(model_folder, "vectorizer.pkl")
)

print("Vectorizer saved to:")
print(os.path.join(model_folder, "vectorizer.pkl"))

y = df["label"]

print("\n========== TF-IDF SHAPE ==========")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Training Data :", X_train.shape)

print("Testing Data  :", X_test.shape)

model = LinearSVC()
model.fit(X_train, y_train)

joblib.dump(
    model,
    os.path.join(model_folder, "spam_model.pkl")
)

print("Model saved to:")
print(os.path.join(model_folder, "spam_model.pkl"))

print("\n========== MODEL TRAINED SUCCESSFULLY ==========")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

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