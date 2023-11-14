import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data\emotions.csv")

cry_words = ["cry", "tears", "sob"]
sadness_words = ["sad", "bad", "waiting"]
happy_words = ["happy", "happiness", "sob" ]

columns_words = {
    "cry" : cry_words,
    "happy": happy_words,
    "sadness": sadness_words
}

def encode_sentiment():
    le = LabelEncoder()
    target = df["sentiment"]
    Y = pd.DataFrame(target)
    labels = le.fit_transform(Y)
    return labels

def include_word(full_phrase, keywords):
    return 1 if full_phrase.lower() in keywords else 0

def fill_columns(columns_words):
    for key, keywords in columns_words.items():
        df[key] = df["content"].apply(lambda full_phrase : include_word(full_phrase, keywords))


fill_columns(columns_words)