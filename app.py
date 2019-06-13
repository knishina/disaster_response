import re
import os
import json
import nltk
import plotly
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

from flask import Flask, render_template, request, jsonify, redirect
from flask_sqlalchemy import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, inspect


app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    # replace the webpage with a urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # lemmatize, turn into lowercase, strip spaces
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


# load the data.
engine = create_engine("sqlite:///PY/DisasterResponse.db")
df = pd.read_sql_table('disaster_response', engine)

# load the model.
model = joblib.load("PY/classifier.pkl")

@app.route("/genres")
def raw_data1():
    dict = df["genre"].value_counts()
    genres_dict = {}
    genres_dict["direct"] = int(dict["direct"])
    genres_dict["news"] = int(dict["news"])
    genres_dict["social"] = int(dict["social"])

    return jsonify(genres_dict)


@app.route("/categories")
def raw_data2():
    category_names = df.iloc[:, 4:].columns
    category_values = (df.iloc[:, 4:] != 0).sum().values
    cat_df = pd.DataFrame([category_values], columns=category_names)

    cat_dict = {}
    for column in cat_df:
        cat_dict[column] = int(cat_df[column])
    return jsonify(cat_dict)


@app.route('/')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'index.html',
        query=query,
        classification_result=classification_results
    )


if __name__ == '__main__':
     app.debug = True
     port = int(os.environ.get("PORT", 5000))
     app.run(host='0.0.0.0', port=port)