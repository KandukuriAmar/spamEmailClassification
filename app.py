from flask import Flask, request, render_template
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
classifier = joblib.load("spam_classifier.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = process(message)
        msg_transformed = vectorizer.transform([cleaned_message])
        prediction = classifier.predict(msg_transformed)[0]
        result = "Spam" if prediction == 1 else "Ham"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)