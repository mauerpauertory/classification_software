#%%
import gc
import DataPreparation as prep
import ClassificationAlgorithms as cl

from PreprocessingPhase import full_preprocessing, custom_tokenizer, tokenize_stemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection

import numpy as np
import time
import pandas 
import psutil



label = 'label'
title = 'title'
dataI = 'I'
case1 = '1'
case2 = '2'
case3 = '3'

######## Crrate Dataset I ##########################
trainDF_I = prep.prepare_dataset_I()
x_I, y_I = trainDF_I[title], trainDF_I[label]

########## PREPROCESSING & FEATURE ENGINEERING ##############

print(psutil.cpu_percent())
print(psutil.virtual_memory())  # physical memory usage
print('memory % used:', psutil.virtual_memory()[2])

tfidf_I_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_I_1 = tfidf_I_1.fit_transform(x_I)
print("Vectors_I_1 shape is {} ".format(vectors_I_1.shape))
train_x_I, test_x_I, train_y_I, test_y_I = model_selection.train_test_split(vectors_I_1, y_I, test_size=0.33, random_state=0)

print(psutil.cpu_percent())
print(psutil.virtual_memory())  # physical memory usage
print('memory % used:', psutil.virtual_memory()[2])


tfidf_I_2 = TfidfVectorizer(tokenizer=tokenize_stemmer)
vectors_I_2 = tfidf_I_2.fit_transform(x_I)
print("Vectors_I_1 shape is {} ".format(vectors_I_2.shape))
train_x_I_2, test_x_I_2, train_y_I_2, test_y_I_2 = model_selection.train_test_split(vectors_I_2, y_I, test_size=0.33, random_state=0)

print(psutil.cpu_percent())
print(psutil.virtual_memory())  # physical memory usage
print('memory % used:', psutil.virtual_memory()[2])


tfidf_I_3 = TfidfVectorizer(tokenizer=full_preprocessing)
vectors_I_3 = tfidf_I_3.fit_transform(x_I)
print("Vectors_I_1 shape is {} ".format(vectors_I_3.shape))
train_x_I_3, test_x_I_3, train_y_I_3, test_y_I_3 = model_selection.train_test_split(vectors_I_3, y_I, test_size=0.33, random_state=0)

print(psutil.cpu_percent())
print(psutil.virtual_memory())  # physical memory usage
print('memory % used:', psutil.virtual_memory()[2])

######### CLASSIFICATION ###################################

state = "#1 Testing first scenario"
scores_1 = cl.train_dataI(train_x_I, test_x_I, train_y_I, test_y_I, case1, True, state)

state = "#2 Testing second scenario"
scores_2 = cl.train_dataI(train_x_I_2, test_x_I_2, train_y_I_2, test_y_I_2, case2, True, state)

state = "#3 Testing third scenario"
scores_3 = cl.train_dataI(train_x_I_3, test_x_I_3, train_y_I_3, test_y_I_3, case3, True, state)
