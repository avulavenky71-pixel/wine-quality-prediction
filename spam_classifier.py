import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Dataset
# We are using a publicly hosted SMS/Email spam dataset
print("Loading Spam dataset...")
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
# This file is tab-separated, so we use sep='\t'
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 2. Split the Data
# X contains the raw text messages, y contains the labels ('ham' or 'spam')
X = data['message']
y = data['label']

# Split into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Text Preprocessing (Vectorization)
# Machine learning models can't read text directly; we must convert words into numbers
print("Vectorizing text data...")
vectorizer = CountVectorizer()
X_train_numeric = vectorizer.fit_transform(X_train)
X_test_numeric = vectorizer.transform(X_test)

# 4. Build and Train the Model
print("Training Naive Bayes Spam Classifier...")
model = MultinomialNB()
model.fit(X_train_numeric, y_train)

# 5. Make Predictions and Evaluate
print("Evaluating model...")
predictions = model.predict(X_test_numeric)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\n--- Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, predictions))