import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def process(text):
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

vectorizer = joblib.load('vectorizer.pkl')
classifier = joblib.load('spam_classifier.pkl')

def predict_message(msg):
    msg_transformed = vectorizer.transform([msg])
    prediction = classifier.predict(msg_transformed)
    return "Spam" if prediction[0] == '1' else "Ham"

example_message = "Your Amazon order has been shipped and will arrive soon."
print(f"Message: {example_message}")
print(f"Prediction: {predict_message(example_message)}")