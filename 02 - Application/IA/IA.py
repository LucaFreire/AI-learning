import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


csv_path = "data\emotions.csv"
df = pd.read_csv(csv_path)

trained_naive_bayes = './models/trained_naive_bayes.sav'
vectorizer_naive_bayes = './models/vectorizer_naive_bayes.sav'
le_naive_bayes = './models/le_naive_bayes.sav'

trained_decision_tree = './models/trained_decision_tree.sav'
vectorizer_decision_tree = './models/vectorizer_decision_tree.sav'
le_decision_tree = './models/le_decision_tree.sav'


cry_words = ["cry", "tears", "sob"]
sadness_words = ["sad", "bad", "waiting", "thinking", "sorry", "feel", "miss"]
happy_words = ["happy", "happiness", "sob" ]
love_words = ["love"]
hateful_words = ["hate"]
tired_words = ["tired", "exhausted", "tired out", "fatigued"]
sleep_words = ["sleepy", "drowsy", "nodding", "dozy", "somnolent"]

columns_words = {
    "cry" : cry_words,
    "happy": happy_words,
    "sadness": sadness_words,
    "love": love_words,
    "hate": hateful_words,
    "tired": tired_words,
    "sleep": sleep_words
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

def train_naivy_bayes(df):
    le = LabelEncoder()
    vectorizer = CountVectorizer()
    clf = CategoricalNB()

    df["sentiment"] = le.fit_transform(df["sentiment"])

    df = fill_columns(df, columns_words)

    X = vectorizer.fit_transform(df["content"]).toarray()
    Y = df["sentiment"]
    clf.fit(X, Y)

    joblib.dump(clf, trained_naive_bayes)
    joblib.dump(vectorizer, vectorizer_naive_bayes)
    joblib.dump(le, le_naive_bayes)

def train_decision_tree(df):
    le = LabelEncoder()
    df = fill_columns(df, columns_words)
    vectorizer = CountVectorizer()
    model = DecisionTreeClassifier(max_depth=12, min_samples_split=10)

    data = vectorizer.fit_transform(df["content"]).toarray()
    df["sentiment"] = le.fit_transform(df["sentiment"])

    X = pd.DataFrame(data)
    Y = df["sentiment"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=111)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)

    joblib.dump(model, trained_decision_tree)
    joblib.dump(le, le_decision_tree)
    joblib.dump(vectorizer, vectorizer_decision_tree)
    
    print(accuracy)


def predict_naive_bayes(msg):
    loaded_trained_model = joblib.load(trained_naive_bayes)
    loaded_vectorized_model = joblib.load(vectorizer_naive_bayes)
    loaded_le_model = joblib.load(le_naive_bayes)

    my_df = pd.DataFrame(data=[ msg ], columns=["content"]) 
    my_df = fill_columns(my_df, columns_words)
    print(my_df)

    sample = loaded_vectorized_model.transform(my_df["content"]).toarray()
    result = loaded_trained_model.predict(sample)
    print(loaded_le_model.inverse_transform(result)[0])
    return loaded_le_model.inverse_transform(result)[0]

def predict_decision_tree(msg):
    loaded_trained_model = joblib.load(trained_decision_tree)
    loaded_le_model = joblib.load(le_decision_tree)
    loaded_vectorizer_model = joblib.load(vectorizer_decision_tree)

    my_df = pd.DataFrame(data=[ msg ], columns=["content"])
    my_df = fill_columns(my_df, columns_words)
    print(my_df)
    
    sample = loaded_vectorizer_model.transform(my_df["content"]).toarray()
    prediction = loaded_trained_model.predict(sample)

    print(loaded_le_model.inverse_transform(prediction)[0])
    return loaded_le_model.inverse_transform(prediction)[0]


# train_naivy_bayes(df)
# train_decision_tree(df)
predict_decision_tree("i love someone")
predict_naive_bayes("i love someone")