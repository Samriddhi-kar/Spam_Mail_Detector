# Spam Mail Detector

## Objective

The objective of this project is to build a Machine Learning model that classifies SMS messages as *Spam* or *Ham (Not Spam)* using Natural Language Processing (NLP) techniques.

## Dataset

* SMS Spam Collection Dataset
* Total Messages: *5572*
* Classes:

  * Spam
  * Ham

## Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn
* Joblib

## Project Workflow

1. Load the dataset.
2. Clean the dataset by removing unnecessary columns.
3. Convert labels into numerical values.
4. Preprocess the text:

   * Lowercase conversion
   * Tokenization
   * Stopword removal
   * Stemming
5. Convert text into TF-IDF vectors.
6. Split the dataset into training and testing sets.
7. Train the model.
8. Evaluate the model using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
9. Save the trained model and TF-IDF vectorizer.
10. Predict whether a new message is Spam or Ham.

## Model Performance

* Accuracy: *98.21%*

## Folder Structure

```
Spam_Mail_Detector/
│
├── dataset/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── screenshots/
│
├── src/
│   ├── predict.py
│   └── spam_detector.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

Train the model:

```
python src/spam_detector.py
```

Run the prediction program:

```
python src/predict.py
```

## Sample Prediction

Input:

```
Congratulations! You won free 5000rs!
```

Output:

```
Prediction: SPAM
```

Input:

```
Can we start the meeting?
```

Output:

```
Prediction: HAM
```
