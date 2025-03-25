**Spam Email Classification using Multinomial Naive Bayes and NLP**

**Overview**

This project implements a spam email classifier using the Multinomial Naive Bayes algorithm and Natural Language Processing (NLP) techniques. The model is trained to differentiate between spam and legitimate (ham) emails based on textual features.

**Features**

* Preprocessing of email text data (tokenization, stopword removal, lemmatization, etc.)

* Feature extraction using Term Frequency-Inverse Document Frequency (TF-IDF)

* Training a Multinomial Naive Bayes classifier

* Evaluation using accuracy, precision, recall, and F1-score

* Flask-based web application for classifying new emails

* Visualization of model performance using a confusion matrix

**Installation**

Clone the repository:

* [git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier](https://github.com/KandukuriAmar/spamEmailClassification/tree/main)

* cd Spam_Detection_NLP.py

* Install the required dependencies manually:

pip install numpy pandas nltk scikit-learn joblib flask matplotlib seaborn

* Download NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

**Usage**

* Run the training script to train the model:

python Spam_Detection_NLP.py

* Start the Flask web application:

python app.py

**Evaluation**

* The model is evaluated using the test dataset with the following metrics:

Accuracy

Precision

Recall

F1-score
             
              precision    recall  f1-score   support

         ham       0.99      0.98      0.98       901
        spam       0.89      0.94      0.91       159

    accuracy                           0.97      1060
   macro avg       0.94      0.96      0.95      1060
weighted avg       0.97      0.97      0.97      1060

* Accuracy: 0.9736

*  **Confusion Matrix Visualization**
![image](https://github.com/user-attachments/assets/04efdccb-2999-4191-85e1-4e76e1525e6a)


![image](https://github.com/user-attachments/assets/6364854c-4806-417f-a084-5b1b0ae5fb30)
![image](https://github.com/user-attachments/assets/7d2f27ad-7f02-4103-8bbe-a03f3e4db206)
![image](https://github.com/user-attachments/assets/ccc143ec-cab1-495c-8cf0-c28417b8da8a)

