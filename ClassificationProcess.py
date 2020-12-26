#%%
import DataPreparation as prep
import ClassificationAlgorithms as cl

from PreprocessingPhase import custom_tokenizer

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import model_selection, preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection  import cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn.metrics import classification_report, confusion_matrix, SCORERS

import numpy as np
import time
import pandas
################## PATH VARIABLES #################################

### ORIGINAL DATASETS ####
clickbait_trial_path = './dataset/TrialSet/clickbait'
non_clickbait_trial_path = './dataset/TrialSet/nonclickbait'
clickbait_II_path = './dataset/DatasetII/clickbait_data'
non_clickbait_II_path = './dataset/DatasetII/non_clickbait_data'
clickbait_III_path = './dataset/DatasetIII/clickbait'
non_clickbait_III_path = './dataset/DatasetIII/not-clickbait'

#### PREPROCESSING DATA SAVE PATHS ####
tokens_path = './dataset/DatasetIII/1/tokens'
vectors_III_path = './dataset/DatasetIII/1/vectors.txt'
vectors_custom_III_path = './dataset/DatasetIII/1/vectors_custom.txt'
trainX_III_path = './dataset/DatasetIII/1/trainX'
validX_III_path = './dataset/DatasetIII/1/validX'
textOutput_III_path = './dataset/DatasetIII/1/textOutput'

##### PATH VARIABLES FOR DIFFERENT ALGORITHMS, DATA AND SCENARIOS ######
main_path = './dataset/Dataset'
dataI = 'I'
dataII = 'II'
dataIII = 'III'
case1 = '1'
case2 = '2'
case3 = '3'
nb_model = '/nb_model_'
rf_model = '/rf_model_'
svm_model = '/svm_model_'
cnn_model = '/cnn_model_'
pred = '_y_pred'


def construct_path(data, model, case, pred=False):
    """a method to construct paths for saving results from training
    Parameters
    ----------
    data : String 
        values: I, II, III - to point to each dataset's own path
    model : String
        different for each algorithm, e.g., /nb_model - for NaiveBayes
    case : String
        values: 1, 2, 3 - to identify different scenarios             
    Return
    ------
    path : String
    """
    if pred:
        return main_path+data+model+case+pred
    return main_path+data+model+case


label = 'label'
title = 'title'
#np.random.seed(500)

################# DATA CREATION #################################
def remove_empty_lines():
    """ invoke method on dataset II to remove empty lines
    """
    prep.removeEmptyLines(clickbait_II_path)
    prep.removeEmptyLines(non_clickbait_II_path)

######### Create Small trial set for experiments ##########################
train_trialDF = prep.constructDataset(clickbait_trial_path+"_edited", non_clickbait_trial_path+"_edited")
x_trial, y_trial = train_trialDF[title], train_trialDF[label]
sentence = ["8 Fall Shows To Be Excited About, 10 To Give A Chance, And 6 To Avoid",
             "16 Signs You Are Too Stubborn To Live",
             "“That one was definitely alive”: An undercover video at one of the nation’s biggest pork processors",
             "17 Easy Slow Cooker Soups That Will Warm You Right Up", 
             "Political Shifts on Gay Rights Lag Behind Culture", 
             "Taliban militant kills at least thirteen in northwest Pakistan",
             "17 Pictures Hot People Will Never Understand",
             "25 Reasons A Trip To Costa Rica Could Actually Change Your Life"]
sent_labels = [1, 1, 1, 1, 0, 0, 1, 1]             
text = tfidf_III_1.transform(sentence).toarray()

######### Create Dataset II ##########################
#remove_empty_lines() #no need to invoke this method each time, as dataset is already modified 
trainDF_II = prep.constructDataset(clickbait_II_path+"_edited", non_clickbait_II_path+"_edited")
x_II, y_II = trainDF_II[title], trainDF_II[label]

######### Create Dataset III ##########################
trainDF_III = prep.constructDataset(clickbait_III_path, non_clickbait_III_path)
x_III, y_III= trainDF_III[title], trainDF_III[label]

################### PREPROCESSING ################################

###### 1st Scenario: only tokenizer #######################
tfidf_trial = TfidfVectorizer(tokenizer=custom_tokenizer)
trial_vector = tfidf_trial.fit_transform(x_trial).toarray()
print("Trial vector shape is {} ".format(trial_vector.shape))

tfidf_II_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_II_1 = tfidf_II_1.fit_transform(x_II).toarray()
print("Trial vector shape is {} ".format(vectors_II_1.shape))

tfidf_III_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_III = tfidf_III_1.fit_transform(x_III).toarray()
print("Trial vector shape is {} ".format(vectors_III.shape))


#print(tfidf_2.vocabulary_)

#print(tfidf.get_feature_names())
########### DataSetS Separation ##########################

train_x_III, test_x_III, train_y_III, test_y_III = model_selection.train_test_split(vectors_III, y_III, test_size=0.33, random_state=0)

########## Classification Stage ########################
random_forest_classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
naive_bayes_classifier = GaussianNB()
cnn_classifier = 4
svm_classifier_linear = svm.SVC(kernel='linear')

train_mode = False

def train_dataI(train_x, test_x, train_y, test_y, case, mode=False):
    """ a method to invoke training methods on the dataset I without any cross-validations

    trains or loads models for all 4 algorithms, saves all the models into pickle files, 
    while results & reports are saved to csv and txt files 

    possible to use for all the 3 scenarios, since it can take differerent train and test sets

    Parameters
    ----------
    case : String 
        required to construct correct saving paths
    mode : boolean 
        identifies either train mode or loading
    """
    nb_path = construct_path(dataI, nb_model, case)
    rf_path = construct_path(dataI, rf_model, case)
    svm_path = construct_path(dataI, svm_model, case)
    print("Current model is RANDOM FOREST")
    cl.train_or_load(random_forest_classifier, train_x, test_x, train_y, test_y, rf_path+pred, rf_path, mode)
    print("Current model is NAIVE BAYES")
    cl.train_or_load(naive_bayes_classifier, train_x, test_x, train_y, test_y, nb_path+pred, nb_path, mode)
    print("Current model is SVM")
    cl.train_or_load(svm_classifier_linear, train_x, test_x, train_y, test_y, svm_path+pred, svm_path, mode)

#train_dataI(train_x_III, test_x_III, train_y_III, test_y_III, case1, False)

def train_cross_validate(x, y, k, data, case):
    nb_path = construct_path(data, nb_model, case)
    rf_path = construct_path(data, rf_model, case)
    svm_path = construct_path(data, svm_model, case)
    print("Cross-validating with %s folds" %(k))
    print("Training Naive Bayes")
    all_scores = []
    NB_scores = cl.kfoldvalidation(naive_bayes_classifier, k, x, y, nb_path)
    print("Training SVM")
    SVM_scores = cl.kfoldvalidation(svm_classifier_linear, k, x, y, svm_path)
    print("Training RF")
    RF_scores = cl.kfoldvalidation(random_forest_classifier, k, x, y, rf_path)
    all_scores.append(NB_scores)
    all_scores.append(SVM_scores)
    all_scores.append(RF_scores)
    return all_scores
  
#print(scores)

#test_cv = train_cross_validate(vectors_III, y_III, 5, dataIII, case1)
#print(test_cv)

def kfoldsplit(times, x, y):
    k_fold = KFold(n_splits=times)
    for n, (train_indices, test_indices) in enumerate(k_fold.split(x)):
        print('Fold #%s' % (n))
        print('Train: %s | test: %s' % (train_indices, test_indices)) 
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        print('X_train: {} \n y_train: {}'.format(x_train, y_train))
        print('X_test: {} \n y_test: {}'.format(x_test, y_test))


sent_labels = np.array(sent_labels)
train_y_III = np.array(train_y_III)
#kfoldsplit(3, text, sent_labels)
#scores = kfoldvalidation(3, train_x_III, train_y_III)
############################################################

def token_test():
    test_sentence = " "
    print(custom_tokenizer(test_sentence))   
    print(test_sentence)

token_test()


########### STOPWORDS REMOVAL #############
def old_method():
    cv = CountVectorizer() 
    english_stopwords = set(stopwords.words('english')) 
    #print(english_stopwords, len(english_stopwords))
    cv_stopwords = CountVectorizer(stop_words=english_stopwords)
    cv_stopwords.fit(trainDF_I[title])
    vector_without_stopwords = cv_stopwords.transform(trainDF_I[title])
    print(vector_without_stopwords.shape)

    def list_stopwords():
        for ind, sw in enumerate(english_stopwords, start=1): 
            print(ind, sw)
     
#list_stopwords()

########## STEMMING #######################

