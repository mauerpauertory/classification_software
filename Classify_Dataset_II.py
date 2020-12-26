#%%
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
dataII = 'II'
case1 = '1'
case2 = '2'
case3 = '3'

clickbait_II_path = './dataset/DatasetII/clickbait_data'
non_clickbait_II_path = './dataset/DatasetII/non_clickbait_data'

######### Create Dataset III ##########################
trainDF_II = prep.constructDataset(clickbait_II_path+"_edited", non_clickbait_II_path+"_edited")
x_II, y_II= trainDF_II[title], trainDF_II[label]

########## PREPROCESSING & FEATURE ENGINEERING ##############
def print_memory_status():
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    print('memory % used:', psutil.virtual_memory()[2])

print_memory_status()

#============= 1st Scenario: tokenizer only -================#
tfidf_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_1 = tfidf_1.fit_transform(x_II)
print("Vectors_III_1 shape is {} ".format(vectors_1.shape))

#============= 2nd Scenario: tokenizer + stemmer -================#
tfidf_2 = TfidfVectorizer(tokenizer=tokenize_stemmer)
vectors_2 = tfidf_2.fit_transform(x_II)
print("Vectors_III_2 shape is {} ".format(vectors_2.shape))

#======== 3rd Scenario: tokenizer + stemmer + stop_words removal ====#
tfidf_3 = TfidfVectorizer(tokenizer=full_preprocessing)
vectors_3 = tfidf_3.fit_transform(x_II)
print("Vectors_III_3 shape is {} ".format(vectors_3.shape))


print_memory_status()

######################## CLASSIFICATION ###################################

state = "#1 Testing first scenario"
print(state)
scores_1 = cl.train_cross_validate(vectors_1, y_II, 10, dataII, case1, state)
print(scores_1)

state = "#2 Testing second scenario"
print(state)
scores_2 = cl.train_cross_validate(vectors_2, y_II, 10, dataII, case3, state)
print(scores_2)

state = "#3 Testing third scenario"
print(state)
scores_3 = cl.train_cross_validate(vectors_3, y_II, 10, dataII, case3, state)
print(scores_3)