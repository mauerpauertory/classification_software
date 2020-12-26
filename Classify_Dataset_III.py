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
dataIII = 'III'
case1 = '1'
case2 = '2'
case3 = '3'

clickbait_III_path = './dataset/DatasetIII/clickbait'
non_clickbait_III_path = './dataset/DatasetIII/not-clickbait'

######### Create Dataset III ##########################
trainDF_III = prep.constructDataset(clickbait_III_path, non_clickbait_III_path)
x_III, y_III= trainDF_III[title], trainDF_III[label]

########## PREPROCESSING & FEATURE ENGINEERING ##############
def print_memory_status():
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    print('memory % used:', psutil.virtual_memory()[2])

print_memory_status()

tfidf_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_1 = tfidf_1.fit_transform(x_III)
print("Vectors_III_1 shape is {} ".format(vectors_1.shape))

tfidf_2 = TfidfVectorizer(tokenizer=tokenize_stemmer)
vectors_2 = tfidf_2.fit_transform(x_III)
print("Vectors_III_2 shape is {} ".format(vectors_2.shape))


tfidf_3 = TfidfVectorizer(tokenizer=full_preprocessing)
vectors_3 = tfidf_3.fit_transform(x_III)
print("Vectors_III_3 shape is {} ".format(vectors_3.shape))


print_memory_status()

######### CLASSIFICATION ###################################

state = "#1 Testing first scenario"
print(state)
scores_1 = cl.train_cross_validate(vectors_1, y_III, 5, dataIII, case1, state)
print(scores_1)

state = "#2 Testing second scenario"
print(state)
scores_2 = cl.train_cross_validate(vectors_2, y_III, 5, dataIII, case3, state)
print(scores_2)

state = "#3 Testing third scenario"
print(state)
scores_3 = cl.train_cross_validate(vectors_3, y_III, 5, dataIII, case3, state)
print(scores_3)