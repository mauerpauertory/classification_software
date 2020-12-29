#%%
import DataPreparation as prep
import ClassificationAlgorithms as cl

from PreprocessingPhase import full_preprocessing, simple_tokenizer, tokenize_stemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn import model_selection

import pandas 
import psutil

label = 'label'
title = 'title'
dataI = 'I'
case1 = '1'
case2 = '2'
case3 = '3'

def print_memory_status():
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    print('memory % used:', psutil.virtual_memory()[2])

print_memory_status()

######## Crrate Dataset I ##########################
trainDF_I = prep.prepare_dataset_I()
x_I, y_I = trainDF_I[title], trainDF_I[label]

train_x_I, test_x_I, train_y_I, test_y_I = model_selection.train_test_split(x_I, y_I, test_size=0.33, random_state=0)
print(Counter(y_I))

######## TF-IDF Vectorizing ##############
print_memory_status()

##===== 1st Scenario =============########
tfidf_1 = TfidfVectorizer(tokenizer=simple_tokenizer)
train_vector_1 = tfidf_1.fit_transform(train_x_I)
test_vector_1 = tfidf_1.transform(test_x_I)

#--- oversampling clickbait ----------#
oversample = RandomOverSampler(sampling_strategy='minority')
train_x_over_1, train_y_over_1 = oversample.fit_resample(train_vector_1, train_y_I)

##===== 2nd Scenario =============########
tfidf_2 = TfidfVectorizer(tokenizer=tokenize_stemmer)
train_vector_2 = tfidf_2.fit_transform(train_x_I)
test_vector_2 = tfidf_2.transform(test_x_I)
#--- oversampling clickbait ----------#
train_x_over_2, train_y_over_2 = oversample.fit_resample(train_vector_2, train_y_I)

##===== 3rd Scenario =============########
tfidf_3 = TfidfVectorizer(tokenizer=full_preprocessing)
train_vector_3 = tfidf_3.fit_transform(train_x_I)
test_vector_3 = tfidf_3.transform(test_x_I)

#--- oversampling clickbait ----------#
train_x_over_3, train_y_over_3 = oversample.fit_resample(train_vector_3, train_y_I)



######### CLASSIFICATION ###################################

state = "#1 Testing first scenario with oversampling"
scores_1 = cl.train_dataI(train_x_over_1, test_vector_1, train_y_over_1, test_y_I, case1, True, state)

state = "#2 Testing second scenario with oversampling"
scores_2 = cl.train_dataI(train_x_over_2, test_vector_2, train_y_over_2, test_y_I, case2, True, state)

state = "#3 Testing third scenario with oversampling"
scores_3 = cl.train_dataI(train_x_over_3, test_vector_3, train_y_over_3, test_y_I, case3, True, state)



##### Print functions to monitor tf-idf and oversampling ############ 

def print_vectors():
    print("#1 only tokenized train_vector: {}".format(train_vector_1.shape))
    print(Counter(train_y_I))
    print("#1 only tokenized test_vector: {}".format(test_vector_1.shape))
    print("#1 oversampled train_vector: {}".format(train_x_over_1.shape))
    print(Counter(train_y_over_1))

    print("#2 train_vector: {}".format(train_vector_2.shape))
    print(Counter(train_y_I))
    print("#2 test_vector: {}".format(test_vector_2.shape))
    print("#2 oversampled train_vector: {}".format(train_x_over_2.shape))
    print(Counter(train_y_over_2))

    print("#3 train_vector: {}".format(train_vector_3.shape))
    print(Counter(train_y_I))
    print("#3 test_vector: {}".format(test_vector_3.shape))
    print("#3 oversampled train_vector: {}".format(train_x_over_3.shape))
    print(Counter(train_y_over_3))
