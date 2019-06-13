"""
The purpose of this python file is to use the cleaned data and to run the ML model.
"""

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin


def load_data():
    """
    Input: None
    Output: X_train, X_test, y_train, y_test, and column_names
    Tasks Performed:
        - read in db.
        - split into X and y.
        - split X into X_train and X_test; split y into y_train and y_test.
    """
    # load data from database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('disaster_response', con=engine)
    
    # split data into X and y.
    X = df["message"]
    y = df.iloc[:, 4:]
    
    # split data into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # obtain column names.
    column_names = list(y.columns)
    
    return X_train, X_test, y_train, y_test, column_names


def tokenize(text):
    """
    Input: text (a list of text messages (english))
    Output: clean, tokenized text ready for ML modeling.
    Tasks Performed:
        - replace any urls in the text with a placeholder.
        - tokenize text
        - lemmatize, turn into lowercase, strip spaces.
    """
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


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Purpose: a class that extracts the starting verb of a sentence.  This is a new feature for the ML classifier.
    """
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


def new_model_pipeline():
    """
    Input: None.
    Output: ML pipeline with StartingVerbExtractor, AdaBoostClassifier, and paramenters with grid_search.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000),
        'features__text_pipeline__tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def save_model(model, model_filepath):
    """
    Input: model.
    Output: saved model called classified.sav
    """
    filename = model_filepath
    pickle.dump(model, open(filename, "wb"))
    pass


def main():
    """
    Input: none
    Output: saved model.
    Tasks Performed:
        - split data into train and test.
        - bring in the model and fit.
        - predict on model
        - print out accuracy/recall/f1 score.
        - save the model.
    """
    # split the data into train and test.
    X_train, X_test, y_train, y_test, column_names = load_data()
   
    # bring in the pipeline and fit it with data.
    model = new_model_pipeline()
    model.fit(X_train, y_train)
   
    # test the new model.
    results = model.predict(X_test)

    # overall accuracy of the model.
    overall_accuracy = (results == y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))

    # scores for each category using classification_report.
    y_pred_pd = pd.DataFrame(results, columns = column_names)
    for column in column_names:
        print(f"FEATURE: {column}")
        print(classification_report(y_test[column],y_pred_pd[column]))
        print("------------------------------------------------------\n")

    # return the saved model.
    with open('classifier.pkl', 'wb') as file:
        pickle.dump(model, file)
    print ("doned")


if __name__ == '__main__':
    main()