import json
import plotly
import pandas as pd
import os
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

#Create flask app
app = Flask(__name__)

#Define tokenize function
def tokenize(text):
    """
    This tokenize function processes input text to generate useful word tokens
    Input: text content
    Output: processed word tokens
    """
    # Define stop_words and create stemming and lemmatizing objects
    stop_words = stopwords.words("english")
    stemming = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # First use regular expression to get rid of punctuations
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # Then tokenize the text and eliminate stop words
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]

    # In the last step, process the word token list with lemmitizer and then stemmer
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return stemmed

#Define new custom transformer called Special_Punc_Counter to get average amount of special punctuation per sentence
class Special_Puncs_Counter():
    #count_special_puncs method is the main method to count special pnctuations
    def count_special_puncs(self, text):
        """
        This function analyzes the amount of special punctuations and divide it by number of sentences
        Imput: Text to analyze
        Output: Calculated special punctuation per sentence
        """
        #Use sent_tokenize to process input text
        sentence_list = nltk.sent_tokenize(text)
        num_sentence = len(sentence_list)
        count = 0
        
        #Iterate through sentences to count special punctuations
        for sentence in sentence_list:
            puncs = re.findall(r'[!?~<>({:;]',sentence)
            count += len(puncs)
     
        #Calculate special punctuation per sentence
        try:
            return count/num_sentence
        except:
            return 0
        
    #Define fit method
    def fit(self, X, y=None):
        return self
    
    #Define transform method to transform series of interest
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.count_special_puncs)
        return pd.DataFrame(X_tagged)

#template

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_Response_Table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Type of messages in training set
    Quantity_Type = df.sum(axis=0)
    X_type_count = Quantity_Type.iloc[3:].index
    Y_type_count = Quantity_Type.iloc[3:].values
    
    #Metrics of models
    Metrics = pd.read_csv("../models/Metrics.csv")
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=list(genre_names),
                    y=list(genre_counts)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=list(X_type_count),
                    y=list(Y_type_count)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
           {
            'data': [
                {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['RandomForest_accuracy']),
                 'name':'RandomForest',
                 'mode':'lines+markers'},
                 {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['RandomForest with New Feature_accuracy']),
                 'name':'RandomForest with Custom Feature',
                 'mode':'lines+markers'},
                {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['AdaBoostClassifier_accuracy']),
                 'name':'AdaBoost',
                 'mode':'lines+markers'},
                {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['AdaBoostClassifier with New Feature_accuracy']),
                 'name':'AdaBoost with Custom Feature',
                 'mode':'lines+markers'}
               
            ],

            'layout': {
                'title': 'Accuracy Comparison of Algorithms',
                'yaxis': {
                    'title': "Accuracy"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
                {
            'data': [
                {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['RandomForest_f1_score']),
                 'name':'RandomForest',
                 'mode':'lines+markers'},
                 {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['RandomForest with New Feature_f1_score']),
                 'name':'RandomForest with Custom Feature',
                 'mode':'lines+markers'},
                {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['AdaBoostClassifier_f1_score']),
                 'name':'AdaBoost',
                 'mode':'lines+markers'},
                {'x':list(Metrics.iloc[:,0]),
                 'y':list(Metrics['AdaBoostClassifier with New Feature_f1_score']),
                 'name':'AdaBoost with Custom Feature',
                 'mode':'lines+markers'}
               
            ],

            'layout': {
                'title': 'f1 score Comparison of Algorithms',
                'yaxis': {
                    'title': "f1 score"
                },
                'xaxis': {
                    'title': "Categories"
                }
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
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    port = int(os.environ.get('PORT', 3001))
    app.run(host='127.0.0.1', port=port, debug=True)


if __name__ == '__main__':
    main()