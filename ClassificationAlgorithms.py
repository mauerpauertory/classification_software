import time
from DataPreparation import saveMatrixToFile, saveTrainingModel, loadTrainModel, saveMetrics, saveReport
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, SCORERS
from sklearn.model_selection import cross_val_predict, cross_validate, KFold

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn import model_selection, preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

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
all_models = '/all_models_'
pred = '_y_pred'
result_path = '/results'



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
        return main_path+data+result_path+model+case+pred
    return main_path+data+result_path+model+case


############# ALGORITHMS #############################################
random_forest_classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
naive_bayes_classifier = MultinomialNB()
cnn_classifier = 4
svm_classifier_linear = svm.SVC(kernel='linear')

############# TRAIN METHODS ####################################################

def train_or_load(classifier, train_x, test_x, train_y, test_y, path, file_name, train_mode=True):
    """ method to envoke either train or load methods depending on train_mode value

    Parameters
    ----------
    train_mode : boolean
        by default is set to True
    """
    if train_mode:
        print("training the model: {}".format(file_name))
        predictions = train_model_predictions(classifier, train_x, train_y, test_x, path, file_name)
    else:
        print("Using existing model: {}".format(file_name))
        predictions = predictOnExistingModel(file_name, test_x) 
    print("The results for the following model: {}".format(file_name))    
    report = evaluate_model(predictions, test_y, file_name)
    return report


def train_model_predictions(classifier, train_x, train_y, valid_x, path, file_name): 
    start = time.time()
    classifier.fit(train_x, train_y)
    fit_time = time.time() - start
    y_predictions = classifier.predict(valid_x)
    saveMatrixToFile(y_predictions, path)
    execution_time = time.time() - start
    time_report = "training time is\n{}\noverall execution time is\n{}".format(fit_time, execution_time)
    saveReport(time_report, file_name)
    print(time_report)
    saveTrainingModel(classifier, file_name)
    return y_predictions

def evaluate_model(y_predictions, actual_y, file_name):
    print("Predicted values {} \n".format(y_predictions))
    print("Actual values {} \n".format(actual_y))
    matrix = confusion_matrix(actual_y, y_predictions)
    report = classification_report(actual_y,y_predictions)
    report_dict = classification_report(actual_y,y_predictions, output_dict=True)
    report_unrounded = saveMetrics(report_dict, file_name)
    print(report)
    print(accuracy_score(actual_y, y_predictions))
    saveReport(report, file_name)
    saveReport(matrix, file_name)
    saveReport(report_unrounded, file_name)
    return report_dict


def predictOnExistingModel(file_name, test_set):
    model = loadTrainModel(file_name)
    return model.predict(test_set)

def kfoldvalidation(classifier, kfold, x, y, file_name):
    """ performs cross validation on the full dataset,
    uses cross_validate for that with the usual set of metrics -
                    accuracy, precision, recall, f1

    Parameters
    ----------
    classifier - classifier Pipeline
    kfold : int 
        number determining the number of {k}-fold validations 
    x : array 
        feature vectors 
    y : array 
        labels 
    file_name : String 
        points to the file_name or path to store achieved values to 

    Return
    ------
    scores : Dict 
        calculated values from cross_validate function
        includes time, accuracy, precision, recall and weighted and mean values of them               
    """
    start = time.time()
    scoring_dict = {'accuracy': 'accuracy',
                    'precision' : 'precision',
                    'recall' : 'recall',
                    'f1' : 'f1',
                    'average_precision' : 'average_precision',
                    'precision_weighted' : 'precision_weighted',
                    'recall_weighted' : 'recall_weighted',
                    'f1_weighted' : 'f1_weighted'}
    scores = cross_validate(classifier, x, y, cv=kfold, scoring=scoring_dict)
    finish_time = "execution time is {} ".format(time.time() - start)
    print(finish_time)
    saveMetrics(scores, file_name, False)
    saveReport(scores, file_name, open_mode="a")
    saveReport(finish_time, file_name)
    print("Scores are\n{}".format(scores))
    return scores

def train_dataI(train_x, test_x, train_y, test_y, case, mode=False, text=''):
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
    all_path = construct_path(dataI, all_models, case)
    saveReport(text, all_path)
    start = time.time()
    all_scores = []
    print("Current model is NAIVE BAYES")
    NB_score = train_or_load(naive_bayes_classifier, train_x, test_x, train_y, test_y, nb_path+pred, nb_path, mode)
    print("Current model is SVM")
    SVM_score = train_or_load(svm_classifier_linear, train_x, test_x, train_y, test_y, svm_path+pred, svm_path, mode)
    print("Current model is RANDOM FOREST")
    RF_score = train_or_load(random_forest_classifier, train_x, test_x, train_y, test_y, rf_path+pred, rf_path, mode)
    all_scores.append(NB_score)
    all_scores.append(SVM_score)
    all_scores.append(RF_score)
    saveReport(all_scores, all_path)
    saveMetrics(all_scores, all_path, False)
    full_time = time.time()-start
    saveReport(full_time, all_path)
    print("the whole process took {}".format(full_time))
    return all_scores

def train_cross_validate(x, y, k, data, case, text, custom_tokenizer):
    """
    Parameters
    ----------
    x - dataset

    """
    start = time.time()
    k_fold = KFold(n_splits=k, shuffle=True, random_state=42)
    nb_path = construct_path(data, nb_model, case)
    rf_path = construct_path(data, rf_model, case)
    svm_path = construct_path(data, svm_model, case)
    all_path = construct_path(data, all_models, case)
    saveReport(text, all_path)
    print("Cross-validating with %s folds" %(k))
    print("Training Naive Bayes")
    clf_nb = Pipeline([('vect', TfidfVectorizer(tokenizer=custom_tokenizer)), ('nb', naive_bayes_classifier)])
    clf_rf = Pipeline([('vect', TfidfVectorizer(tokenizer=custom_tokenizer)), ('rf', random_forest_classifier)])
    clf_svm = Pipeline([('vect', TfidfVectorizer(tokenizer=custom_tokenizer)), ('svm', svm_classifier_linear)])
    all_scores = []
    NB_scores = kfoldvalidation(clf_nb, k_fold, x, y, nb_path)
    print("Training SVM")
    SVM_scores = kfoldvalidation(clf_svm, k_fold, x, y, svm_path)
    print("Training RF")
    RF_scores = kfoldvalidation(clf_rf, k_fold, x, y, rf_path)
    all_scores.append(NB_scores)
    all_scores.append(SVM_scores)
    all_scores.append(RF_scores)
    print("Saving all_scores to {}".format(all_path))
    saveReport(all_scores, all_path)
    saveMetrics(all_scores, all_path, False)
    finish_time = "execution time is {} ".format(time.time() - start)
    return all_scores   
  

def kfoldsplit(times, x, y, custom_tokenizer):
    k_fold = KFold(n_splits=times, shuffle=True, random_state=42)
    for n, (train_indices, test_indices) in enumerate(k_fold.split(x)):
        print('Fold #%s' % (n))
        #print('Train: %s | test: %s' % (train_indices, test_indices)) 
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        print(Counter(y_train))
        print(Counter(y_test))
        tfidf_n = TfidfVectorizer(tokenizer=custom_tokenizer)
        train_vector = tfidf_n.fit_transform(x_train)
        test_vector = tfidf_n.transform(x_test)
        print(train_vector.shape)
        print(test_vector.shape)
        start = time.time()
        naive_bayes_classifier.fit(train_vector, y_train)
        svm_classifier_linear.fit(train_vector, y_train)
        nb_y_pred = naive_bayes_classifier.predict(test_vector)
        svm_y_pred = svm_classifier_linear.predict(test_vector)
        report_dict_nb = classification_report(y_test,nb_y_pred, output_dict=True)
        report_dict_svm = classification_report(y_test,svm_y_pred, output_dict=True)
        execution_time = time.time() - start
        print("Execution time is %s" %(execution_time))
        print("Naive Bayes #%s results:\n" %n)
        print(report_dict_nb)
        print("SVM #%s results:\n" %n)
        print(report_dict_svm)
