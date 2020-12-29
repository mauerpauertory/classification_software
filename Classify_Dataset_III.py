#%%
import DataPreparation as prep
import ClassificationAlgorithms as cl

from PreprocessingPhase import full_preprocessing, simple_tokenizer, tokenize_stemmer

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

######### CLASSIFICATION ###################################

state = "#1 Testing first scenario"
print(state)
scores_1 = cl.train_cross_validate(x_III, y_III, 5, dataIII, case1, state, simple_tokenizer)
print(scores_1)
state = "#2 Testing second scenario"
print(state)
scores_2 = cl.train_cross_validate(x_III, y_III, 5, dataIII, case2, state, tokenize_stemmer)
print(scores_2)
state = "#3 Testing third scenario"
print(state)
scores_3 = cl.train_cross_validate(x_III, y_III, 5, dataIII, case3, state, full_preprocessing)
print(scores_3)