import IA
from flask import Flask

app = Flask(__name__)

@app.route('/API/naive-bayes/<msg>', methods = ["GET"])
def predict_naive_bayes(msg):
    return IA.predict_naive_bayes(msg)

@app.route('/API/decision-tree/<msg>', methods = ["GET"])
def predict_decision_tree(msg):
    return IA.predict_decision_tree(msg)

app.run(host="localhost", port=3030)