import numpy as np
import pandas as pd
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download("stopwords")
nltk.download("wordnet")

# data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
data = pd.read_csv('new_spam.csv', encoding='ISO-8859-1')

data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]) # used to remove the commas ,
data = data.rename(columns={'v1': 'category', 'v2': 'text'})

data['category'] = data['category'].map({'ham': 0, 'spam': 1})

data.drop_duplicates(inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

data['cleaned_text'] = data['text'].apply(process)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['category']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

classifier = MultinomialNB(alpha=0.1)  # Default alpha=1, lower alpha means less smoothing
classifier.fit(x_train, y_train)

joblib.dump(classifier, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

train_acc = accuracy_score(y_train, classifier.predict(x_train))
test_acc = accuracy_score(y_test, classifier.predict(x_test))

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("Model and vectorizer saved successfully!")

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = classifier.predict(x_test)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)  # Added this missing line

print(report)
print(f'Accuracy: {accuracy:.4f}')

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
