import pandas
import numpy as np
import pickle 
import json
import jsonlines 
import gc

from collections import defaultdict
from sklearn import model_selection, preprocessing

label = 'label'
title = 'title'
#np.random.seed(500)

############# Load the TXT data ###################
def constructDataset(clickbait_path, non_clickbait_path):
    """ constructs DataFrame object with titles and labels as columns 
    from two text files: clickbait and non-clickbait

    Parameters
    ----------
    clickbait_path : String
        path to the txt file for clickbait data
    non_clickbait_path : String
        path to the txt file for non-clickbait data

    Returns
    -------
    trainDF : DataFrame object
        contains both clickbait and non-clickbait data: 
            [title] - for titles
            [label] - for lables: clickbait=1, non-clickbait=0    
    """
    clickbait_dataset = open(clickbait_path, encoding="utf8").read()
    non_clickbait_dataset = open(non_clickbait_path, encoding="utf8").read()
    labels, titles = [], []

    for i, line in enumerate(clickbait_dataset.split("\n")):
        labels.append(1)
        titles.append(line)
        #print(i, line)
    for i, line in enumerate(non_clickbait_dataset.split("\n")):
        labels.append(0)
        titles.append(line)
        #print(i, line)

    trainDF = pandas.DataFrame()
    trainDF[title] = titles
    trainDF[label] = labels
    return trainDF
    #print(titles)
    #print(labels)

def loadTrainModel(name):
    with open(name, 'rb') as training_model:
        return pickle.load(training_model)

############# Original Dataset File Modifiers ####################

####### Dataset I ###########

def jsonl_to_dict_list(file_name, key1, key2):
    json_dict_list = []
    with jsonlines.open(file_name+'.jsonl') as json_file, open(file_name+'_edited', 'w', encoding="utf8") as outfile:
        for line in json_file:
            key_value_1 = line[key1]
            key_value_2 = line[key2]
            line.clear()
            line[key1] = key_value_1
            line[key2] = key_value_2
            print(line, file=outfile)
            json_dict_list.append(line)
    return json_dict_list       

def merge_lists(file_name, titles_list, labels_list, key_id):
    d = defaultdict(dict)
    for item in titles_list + labels_list: 
        d[item[key_id]].update(item)
    merged_list = list(d.values())  
    save_list_to_file(file_name, merged_list) 
    return merged_list

def save_list_to_file(file_name, list_name):
    with open(file_name, 'w', encoding="utf8") as outfile:
        for item in list_name:
            print(item, file=outfile) 

def create_final_version(list_name, file_name, class_label, id):
    for item in list_name: 
        if item[class_label] == 'no-clickbait':
            item[class_label] = 0
        else:
            item[class_label] = 1 
        item.pop(id, None)
    save_list_to_file(file_name, list_name)    
    return list_name        

def prepare_dataset_I():
    data_trial_path = './dataset/DatasetI/instances'
    evaluation_trial_path = './dataset/DatasetI/truth'
    merged_trial_path = './dataset/DatasetI/merged_list'
    final_version_trial_path = './dataset/DatasetI/final_list'

    
    key_id = 'id'
    key_title = 'targetTitle'
    key_class = 'truthClass'

    titles_list = jsonl_to_dict_list(data_trial_path, key_id, key_title)
    labels_list = jsonl_to_dict_list(evaluation_trial_path, key_id, key_class)

    full_list = merge_lists(merged_trial_path, titles_list, labels_list, key_id)

    final_list = create_final_version(full_list, final_version_trial_path, key_class, key_id)

    json_DF =  pandas.DataFrame(final_list).rename(columns={key_title : title, key_class : label})

    return json_DF


####### Dataset II ##################
def removeEmptyLines(file_name):
    """ method removes empty lines from the identified path, writes the edited version 
    to a new file with "_edited" added to the original file_name 
    
    Parameters
    ----------
    file_name : String 
    """
    with open(file_name, encoding="utf8") as infile, open(file_name+"_edited", 'w', encoding="utf8") as outfile:
        empty_lines = 0
        total = 0 
        for line in infile: 
            if not line.strip(): 
                empty_lines += 1
                continue
            outfile.write(line)
            total+=1
        print("Empty lines are %s\n total lines are %s" % (empty_lines, total))



############# Train and Test data splitting methods ####################        
def splitDataFrame(trainDF):
    """ splits Dataframe object into train and test sets with 2:1 ratio

    Parameters
    ----------
    trainDF - DataFrame object

    Returns 
    -------
    train_x, test_x, train_y, test_y 
    """
    return model_selection.train_test_split(trainDF[title], trainDF[label], test_size=0.33, random_state=0)

def splitData(x, y): 
    """ splits X and Y data into train and test sets with 2:1 ratio

    Parameters
    ----------
    x - headlines
    y - labels

    Returns 
    -------
    train_x, test_x, train_y, test_y 
    """
    return model_selection.train_test_split(x, y, test_size=0.33, random_state=0)    

############ Saving to Files methods ####################

def writeTrainAndTestTitlesToFile(train_x, valid_x, train_path, valid_path):
    """ writing the titles of train and test sets to a different files

    Parameters
    ----------
    train_x : array 
        either strings or decimals, train set of titles the classifiers will learn on
    valid_x : array    
        strings or decimals, test set of titles classifier will predict on    
    Returns 
    ---------
    none

    """
    trainX_output = open(train_path, "w", encoding="utf8")
    validX_output = open(valid_path, "w", encoding="utf8")
    for line in train_x:
        trainX_output.write(line)
        trainX_output.write("\n")
    trainX_output.close()  
    for line in valid_x:
        validX_output.write(line)
        validX_output.write("\n")
    validX_output.close()  

def writeVocabulary(vec, path):
    """ write either CountVectorizer or Tfidf vocabulary to a file 

    Parameters
    ----------
    vec - CountVectorizer or TfidfVectorizer
    path - path to file
    """
    with open(path, "w", encoding="utf8") as f: 
        print(vec.vocabulary_, file=f)
        for line in vec.vocabulary_:
            print(line, file=f)

def saveMatrixToFile(vector, path):
    """ save sparse matrix to a txt file in a pretty format 

    Parameters 
    ----------
    vector - sparse matrix 
    path - path to file
    """
    np.savetxt(path, vector, fmt="%1.4f", encoding="utf8")

def saveTrainingModel(classifier, name):
    with open(name, 'wb') as picklefile:
        pickle.dump(classifier,picklefile)   

def saveMetrics(metrics, file_name, is_class_report=True, open_mode='a'):
    """
    Parameters
    ----------
    is_classification : boolean 

    """
    if is_class_report:
        df = pandas.DataFrame(metrics).transpose()
    else: 
        df = pandas.DataFrame(metrics)   
    with open(file_name+".csv", open_mode) as f:
        df.to_csv(f)
    return df 

def saveReport(report, file_name, open_mode="a"): 
    with open(file_name+".txt", open_mode, encoding="utf8") as f: 
        print(report, file=f)
        print("\n")

############ Print methods ##############################
def print_stats(train_x, train_y, valid_x, valid_y):
    print("train_x of length {} is\n {} \n valid_x of length {} is\n {} and \n".format(len(train_x),train_x,len(valid_x),valid_x))
    clickbait_num = 0
    valid_clickbait_num = 0
    non_clickbait_num = 0
    valid_non_clickbait_num = 0
    for y in train_y: 
        if y==1:
            clickbait_num+=1   
        if y==0:
            non_clickbait_num+=1
    for v in valid_y: 
        if v==1:
            valid_clickbait_num+=1    
        if v==0:
            valid_non_clickbait_num+=1   
    print("Clickbait and non-clickbait articles in train set are {} and {}\n".format(clickbait_num, non_clickbait_num))
    print("Clickbait and non-clickbait articles in validation set are {} and {}\n".format(valid_clickbait_num, valid_non_clickbait_num))
