import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.feature_extraction.text import CountVectorizer

csv_path = "data\emotions.csv"
df = pd.read_csv(csv_path)

trained_filename = './models/trained_model.sav'
vectorizer_filename = './models/vectorizer_model.sav'
le_filename = './models/le_model.sav'

le = LabelEncoder()
vectorizer = CountVectorizer()

cry_words = ["cry", "tears", "sob"]
sadness_words = ["sad", "bad", "waiting", "thinking", "sorry", "feel", "miss"]
happy_words = ["happy", "happiness", "sob" ]
love_words = ["love"]
hateful_words = ["hate"]

columns_words = {
    "cry" : cry_words,
    "happy": happy_words,
    "sadness": sadness_words,
    "love": love_words,
    "hate": hateful_words
}

def include_word(full_phrase, vector_keywords):
  for keyword in vector_keywords:
    if keyword in full_phrase:
      return 1
  return 0

def fill_columns(df, columns_words):
    for key, keywords in columns_words.items():
        df[key] = df["content"].apply(lambda full_phrase : include_word(full_phrase.lower(), keywords))
    return df

def train(df):
    df["sentiment"] = le.fit_transform(df["sentiment"])

    df = fill_columns(df, columns_words)

    X = vectorizer.fit_transform(df["content"]).toarray()
    Y = df["sentiment"]
    clf = CategoricalNB()
    clf.fit(X, Y)

    joblib.dump(clf, trained_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    joblib.dump(le, le_filename)

def predict(msg):
    loaded_trained_model = joblib.load(trained_filename)
    loaded_vectorized_model = joblib.load(vectorizer_filename)
    loaded_le_model = joblib.load(le_filename)

    my_df = pd.DataFrame(data=[ msg ], columns=["content"]) 
    my_df = fill_columns(my_df, columns_words)
    print(my_df)

    sample = loaded_vectorized_model.transform(my_df["content"]).toarray()
    result = loaded_trained_model.predict(sample)
    print(loaded_le_model.inverse_transform(result)[0])

# train(df)
predict("i love and hate school")