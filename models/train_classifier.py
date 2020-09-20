#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

import pandas as pd
import pickle

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def merge_low_count_ctg(df, threshold):
    col_sizes = {}
    cols_to_merge = []
    for col in df.columns:
        col_sizes[col]=df[col].sum()
        if (df[col].sum()<1500): cols_to_merge.append(col)
    df['other'] = df[cols_to_merge].sum(axis=1)
    df.drop(cols_to_merge, axis=1, inplace=True)
    df['other']=df['other'].apply(lambda x: 1 if x> 0 else x)
    return df

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("select * from Messages", engine)
    print ('Loaded: ', df.shape)
    # drop data with no categories assigned
    df = df[df['cat_num']>0]
    X = df['message']
    Y = df.iloc[:, 4:-1]
    # optional - consolidate low counts categories to improve accuracy
    Y = merge_low_count_ctg(Y, 1500)
    return X, Y, Y.columns

def tokenize(text):
'''The function will remove punctuation, normalize the text case, lemmatize and remove the stop words'''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = text.lower()
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''The custom transformer will return the number of characters in each message'''
    def fit(self, X, y=None):
        return self

    def transform(self, X):  
        X_len = pd.Series(X).apply(lambda x: len(x))
        #print(pd.DataFrame(X_len))
        return pd.DataFrame(X_len)
       
class POSCounter(BaseEstimator, TransformerMixin):
    '''The custom transformer will return the number of nouns, verbs and adjectives for each message'''
    def pos_counts(self, text):
        sentence_list = nltk.sent_tokenize(text)
        noun_count = 0
        verb_count = 0
        adj_count = 0
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            for w, tag in pos_tags:
               # print (w,tag)
                if (tag=='NN'): noun_count+=1
                elif (tag=='VBZ'): verb_count+=1
                elif (tag=='JJ'): adj_count+=1
        return noun_count, verb_count, adj_count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.pos_counts)
        columns = ['noun_count', 'verb_count', 'adj_count']
        # source: https://stackoverflow.com/questions/53402584/how-to-convert-a-series-of-tuples-into-a-pandas-dataframe
        df = pd.DataFrame([[a,b,c] for a,b,c in X_tagged.values], columns=columns)
        df.head()
        return df
    



def build_model():
    pipeline_mlp = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('pos_counts', POSCounter()),
            ('text_len', TextLengthExtractor())
        ])),

        ('clf', MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(10,))))
    ])
    return pipeline_mlp


def evaluate_model(model, X_test, Y_test, category_names):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=category_names))
    return

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

