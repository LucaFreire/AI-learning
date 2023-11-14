




# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=800)

# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)

# Y_pred = model.predict(X_test)
# accuracy = accuracy_score(Y_test, Y_pred)
# print(accuracy)

# confusion_matrix(Y_test, Y_pred)
# plot_tree(model)





# print("yeah")
# df = pd.read_csv("data\emotions.csv")

# def data_cleanup(num):
#     df.head(num)
#     df.fillna(0)

# def encode_sentiment():
#     le = LabelEncoder()
#     target = df["sentiment"]
#     Y = pd.DataFrame(target)
#     labels = le.fit_transform(Y)
#     return labels

# def training():
#     data_cleanup()
#     data = df[["content"]]
#     X = pd.DataFrame(data)

#     Y = encode_sentiment()

#     vectorizer = TfidfVectorizer()
#     X_train_tfidf_vectorize = vectorizer.fit_transform(X)

#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(70), random_state=1, verbose=True)
#     clf.fit(X_train_tfidf_vectorize, Y)

#     phrase_test = ["Im very sad"]
#     X_new_tfidf_vectorize = vectorizer.transform(phrase_test)

#     pred = clf.predict(X_new_tfidf_vectorize)

#     for doc, category in zip(phrase_test, pred):
#         print('%r => %s' % (doc, category))
# print("Came")
# training()




