from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
df = pd.read_csv("archive\emotions.txt", sep = ";", header = None, names = ["text", "emotion"])

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["emotion"], test_size = 0.2, random_state = 42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Presnost:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
