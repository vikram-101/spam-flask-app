
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'spam.csv'), encoding="latin-1")
df.rename(columns={'v1': 'class', 'v2': 'message'}, inplace=True)
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save trained model and vectorizer
out_dir = os.path.dirname(__file__)
model_file = os.path.join(out_dir, 'NB_spam_model.pkl')
vec_file = os.path.join(out_dir, 'vectorizer.pkl')
joblib.dump(clf, model_file)
joblib.dump(cv, vec_file)
print(f"Saved model to {model_file}")
print(f"Saved vectorizer to {vec_file}")