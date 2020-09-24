import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier

nltk.download('averaged_perceptron_tagger')


app = Flask(__name__)

def tokenize(text):
    '''The function will normalize and lemmatize the text. Returns tokenized text.'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
  
    categories = df.columns[4:-1]
    cat_counts = [df[col].sum() for col in categories]
    cat_names = categories
    
    # add sorting by counts to show better graph
    d = dict(zip(cat_counts, cat_names))
    d_sorted = sorted(d.items(), reverse=True)
    cat_counts = [i[0] for i in d_sorted]
    cat_names = [i[1] for i in d_sorted]
    
    # Identify to categories that co-occur or correlated with the category 'food'
    df_food = df[df['food']==1]
    df_food=df_food.drop('food', axis=1)
    food_categories = df_food.columns[4:-1]
    food_counts = [df_food[col].sum() for col in food_categories]
    food_names = food_categories
    
    # create visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle':-45,
                    'title_standoff':200
                }
            }
        },
        {
            'data': [
                {                    
                    'values':food_counts,
                    'labels':food_names,
                    'type':'pie'
                }
            ],

            'layout': {
                'title': 'Categories that Correlated with Food Categories',
                'height':600
                
            }
        },
        {
            'data': [
                {     
                    'x':df['cat_num'],
                    'type':'histogram'
                }
            ],

            'layout': {
                'title': 'Distribution of messages by number of categories assigned'
                
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:-1], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()