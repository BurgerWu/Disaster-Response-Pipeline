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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('averaged_perceptron_tagger')
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
    #RandomForest Algorithm without new feature
    RF_pipeline = Pipeline([
                    ("Vectorizer",CountVectorizer(tokenizer = tokenize)),\
     			    ("TFIDF",TfidfTransformer()),\
                    ("Estimator",MultiOutputClassifier(RandomForestClassifier(n_estimators = 100, min_samples_split = 3)))])

    #RandomForest Algorithm with new feature
    RF_pipeline_feat = Pipeline([
        ('features', FeatureUnion([

            ('text_processing', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('count_special_puncs', Special_Puncs_Counter())
        ])),

        ('Estimator', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100, min_samples_split = 3)))
    ])

    #AdaBoost Algorithm without new feature
    AB_pipeline = Pipeline([("Vectorizer",CountVectorizer(tokenizer = tokenize)),\
                            ("TFIDF",TfidfTransformer()),\
                            ("Estimator",MultiOutputClassifier(AdaBoostClassifier()))])

    #AdaBoost Algorithm without new feature
    AB_pipeline_feat = Pipeline([
        ('features', FeatureUnion([

            ('text_processing', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('count_special_puncs', Special_Puncs_Counter())
        ])),

        ('Estimator', MultiOutputClassifier(AdaBoostClassifier()))
    ])
	
    model_dict = {"RandomForest":RF_pipeline,\
              "RandomForest with New Feature":RF_pipeline_feat,\
              "AdaBoostClassifier":AB_pipeline,\
              "AdaBoostClassifier with New Feature":AB_pipeline_feat}

    return model_dict


def evaluate_model(model_dict, X_test, Y_test, category_names, num_of_labels):

    Metrics = pd.DataFrame(index = category_names)
    for classifier in model_dict.keys():
        print("Evaluating {}".format(classifier))
        model = model_dict[classifier]
        y_pred = model.predict(X_test)
     
        accuracy = []
        precision = []
        recall = []
        f1_score = []

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
            accuracy.append((TP + TN)/(TP + TN + FP +FN))
            precision.append(TP / (TP + FP))
            recall.append(TP / (TP + FN))
            f1_score.append(2*(TP / (TP + FP)*TP / (TP + FN))/(TP / (TP + FP) + TP / (TP + FN)))
    
        #Append metric column to Metrics table
        Metrics[classifier + "_accuracy"] = accuracy
        Metrics[classifier + "_precsion"] = precision
        Metrics[classifier + "_recall"] = recall
        Metrics[classifier + "_f1_score"] = f1_score
    
    print("Exporting Metrics table")
    #Export results to CSV
    Metrics.to_csv("Metrics.csv")
    
    f1_columns = [x for x in Metrics.columns if "f1" in x]
    
    Average_f1_Table = pd.DataFrame(Metrics[f1_columns].mean(), columns = ["Average f1"])
    Average_f1_Table.sort_values(by='Average f1', ascending = False, inplace = True)
    Chosen_Model_Name = Average_f1_Table.index[0].split('_')[0]
    Chosen_Model_File = model_dict[Chosen_Model_Name]
    print(Chosen_Model_Name)
    return Chosen_Model_File

	

#def gridsearch_model(Chosen_Model_Dict, X, Y):

#    Chosen_Model_Name = Chosen_Model_Dict.keys()
#    Chosen_Model = Chosen_Model_Dict

#    if "RandomForest" in Chosen_Model_Name:
#	parameters = {
#        'Estimator__estimator__n_estimators': [50, 100, 200]}
#    else:
#	parameters = {
#        'Estimator__estimator__learning_rate': [0.5, 1, 2],
#	'Estimator__estimator__n_estimators':[50,100]}
 
#    print("Grid Searching for {}".format(Chosen_Model_Name))
#    cv = GridSearchCV(Chosen_Model, param_grid = parameters)
#    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X, Y,train_size = 0.1, test_size = 0.025)
#    cv.fit(X_train_cv, y_train_cv)        

#    return cv



def save_model(model, model_filepath):
    import pickle
    pickle_file = open(model_filepath,'wb')
    pickle.dump(model, pickle_file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        num_of_labels = Y.shape[1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
        print('Building model...')
        model_dict = build_model()
 
        print('Training model...')
        for clf in model_dict.keys():
            print("Training model {}".format(clf))
            model_dict[clf].fit(X_train, Y_train)
        
        print('Evaluating model...')
        final_model = evaluate_model(model_dict, X_test, Y_test, category_names, num_of_labels)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(final_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()