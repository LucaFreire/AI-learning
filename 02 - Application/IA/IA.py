
if __name__ == "IA":
    import numpy as np
    import pandas as pd
    import os
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from joblib import load, dump
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("IA\data\emotions.csv")

def show(num):
    df.head(num)

def data_cleanup():
    df.fillna(0)
    le = LabelEncoder()
    target = df["sentiment"]
    Y = pd.DataFrame(target)
    labels = le.fit_transform(Y)
    return labels

def training():
    data = df[["content"]]

    X = pd.DataFrame(data)
    Y = data_cleanup()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=800)
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(accuracy)



print("Came")
training()



# confusion_matrix(Y_test, Y_pred)
# plot_tree(model)