#%%
import DataPreparation as prep
import ClassificationAlgorithms as cl

from PreprocessingPhase import full_preprocessing, simple_tokenizer, tokenize_stemmer

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

######################## CLASSIFICATION ###################################

state = "#1 Testing first scenario"
print(state)
scores_1 = cl.train_cross_validate(x_II, y_II, 10, dataII, case1, state, simple_tokenizer)
print(scores_1)

state = "#2 Testing second scenario"
print(state)
scores_2 = cl.train_cross_validate(x_II, y_II, 10, dataII, case2, state, tokenize_stemmer)
print(scores_2)

state = "#3 Testing third scenario"
print(state)
scores_3 = cl.train_cross_validate(x_II, y_II, 10, dataII, case3, state, full_preprocessing)
print(scores_3)