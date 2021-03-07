import sys
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster_Response_Table', engine)
 
    #Create X and Y 
    X = df["message"].values
    Y = df.iloc[:,4:].values

    #Retrieve number of labels and label names
    num_of_labels = Y.shape[1]
    label_names = df.columns[4:]
    return X, Y, label_names

def tokenize(text):
    """
    This tokenize function processes input text to generate useful word tokens
    Input: text content
    Output: processed word tokens    
    """
    #Define stop_words and create stemming and lemmatizing objects
    stop_words = stopwords.words("english")
    stemming = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    #First use regular expression to get rid of punctuations
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    
    #Then tokenize the text and eliminate stop words
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    #In the last step, process the word token list with lemmitizer and then stemmer
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
        sentence_list = nltk.sent_tokenize(text)
        num_sentence = len(sentence_list)
        count = 0
        for sentence in sentence_list:
            puncs = re.findall(r'[!?~<>({:;]',sentence)
            count += len(puncs)
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
    
def build_model():
    #Create pipeline object
    Original_pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_processing', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('count_special_puncs', Special_Puncs_Counter())
        ])),

        ('Estimator', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return Original_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    #Report test set results
    y_pred = model.predict(X_test)

    #Create dataframe to store metrics
    Metrics = pd.DataFrame(columns = ["accuracy", "precision", "recall", "f1-score"])
    num_of_labels = Y_test.shape[1]
    
    for i in range(num_of_labels):
        #Retrieve test and predicted value for that label
        test = Y_test[:,i]
        predict = y_pred[:,i]
    
        #Find number of True Positives, False Negatives, False positives and True Negatives
        TP = np.sum(np.logical_and(test == 1, predict == 1))
        FN = np.sum(np.logical_and(test == 1, predict == 0))
        FP = np.sum(np.logical_and(test == 0, predict == 1))
        TN = np.sum(np.logical_and(test == 0, predict == 0))
    
        #Calculate accuracy, preicsion, recall and f1_score
        accuracy = (TP + TN)/(TP + TN + FP +FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2*(precision*recall)/(precision + recall)
    
        #Append metrics to dataframe
        Metrics = Metrics.append({"accuracy": accuracy, "precision": precision, "recall": recall, "f1-score": f1_score}, ignore_index = True)

        #Export results to CSV
        Metrics.to_csv("Metrics.csv")
        


def save_model(model, model_filepath):
    import pickle
    pickle_file = open(model_filepath,'wb')
    pickle.dump(model, pickle_file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X[:1000], Y[:1000], test_size=0.2)
        print(X_train.shape)
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()