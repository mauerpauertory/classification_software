#%%
import DataPreparation as prep
import ClassificationAlgorithms as cl

from PreprocessingPhase import custom_tokenizer

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection

import numpy as np
import time
import pandas 

label = 'label'
title = 'title'
dataI = 'I'
dataII = 'II'
dataIII = 'III'
dataTrial = 'Trial'
case1 = '1'
case2 = '2'
case3 = '3'

################## PATH VARIABLES #################################

### ORIGINAL DATASETS ####
clickbait_trial_path = './dataset/DatasetTrial/clickbait'
non_clickbait_trial_path = './dataset/DatasetTrial/nonclickbait'
clickbait_II_path = './dataset/DatasetII/clickbait_data'
non_clickbait_II_path = './dataset/DatasetII/non_clickbait_data'
clickbait_III_path = './dataset/DatasetIII/clickbait'
non_clickbait_III_path = './dataset/DatasetIII/not-clickbait'

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

######## Crrate Dataset I ##########################
trainDF_I = prep.prepare_dataset_I()
x_I, y_I = trainDF_I[title], trainDF_I[label]

######### Create Dataset II ##########################
#remove_empty_lines() #no need to invoke this method each time, as dataset is already modified 
trainDF_II = prep.constructDataset(clickbait_II_path+"_edited", non_clickbait_II_path+"_edited")
x_II, y_II = trainDF_II[title], trainDF_II[label]

######### Create Dataset III ##########################
trainDF_III = prep.constructDataset(clickbait_III_path, non_clickbait_III_path)
x_III, y_III= trainDF_III[title], trainDF_III[label]

################### PREPROCESSING & FEATURE ENGINEERING ##############

###### 1st Scenario: only tokenizer #######################
tfidf_trial = TfidfVectorizer(tokenizer=custom_tokenizer)
trial_vector = tfidf_trial.fit_transform(x_trial)
print("Trial vector shape is {} ".format(trial_vector.shape))
text = tfidf_trial.transform(sentence).toarray()

tfidf_I_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_I_1 = tfidf_I_1.fit_transform(x_I)
print("Vectors_I_1 shape is {} ".format(vectors_I_1.shape))
train_x_I, test_x_I, train_y_I, test_y_I = model_selection.train_test_split(vectors_I_1, y_I, test_size=0.33, random_state=0)

tfidf_II_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_II_1 = tfidf_II_1.fit_transform(x_II)
print("Vectors_II_1 shape is {} ".format(vectors_II_1.shape))

tfidf_III_1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectors_III_1 = tfidf_III_1.fit_transform(x_III)
print("Vectors_III_1 shape is {} ".format(vectors_III_1.shape))

state = "#1\tNo modification, simple tokenizer \n"
scores_III_1 = cl.train_cross_validate(vectors_III_1, y_III, 5, dataIII, case1, state)

#------------------------------------------------------------#


tfidf_trial = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000)
trial_vector = tfidf_trial.fit_transform(x_trial).toarray()
print("Trial vector shape is {} ".format(trial_vector.shape))
text = tfidf_trial.transform(sentence).toarray()

tfidf_I_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000)
vectors_I_1 = tfidf_I_1.fit_transform(x_I).toarray()
print("Vectors_I_1 shape is {} ".format(vectors_I_1.shape))
train_x_I, test_x_I, train_y_I, test_y_I = model_selection.train_test_split(vectors_I_1, y_I, test_size=0.33, random_state=0)

tfidf_II_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000)
vectors_II_1 = tfidf_II_1.fit_transform(x_II).toarray()
print("Vectors_II_1 shape is {} ".format(vectors_II_1.shape))

tfidf_III_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000)
vectors_III_1 = tfidf_III_1.fit_transform(x_III).toarray()
print("Vectors_III_1 shape is {} ".format(vectors_III_1.shape))

state = "#2\tMax_features=5000 only \n"
scores_I_1 = cl.train_dataI(train_x_I, test_x_I, train_y_I, test_y_I, case1, True, state)
scores_II_1 = cl.train_cross_validate(vectors_II_1, y_II, 5, dataII, case1, state)
scores_III_1 = cl.train_cross_validate(vectors_III_1, y_III, 5, dataIII, case1, state)

#------------------------------------------------------------#

tfidf_trial = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, max_df=0.7)
trial_vector = tfidf_trial.fit_transform(x_trial).toarray()
print("Trial vector shape is {} ".format(trial_vector.shape))
text = tfidf_trial.transform(sentence).toarray()

tfidf_I_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, max_df=0.7)
vectors_I_1 = tfidf_I_1.fit_transform(x_I).toarray()
print("Vectors_I_1 shape is {} ".format(vectors_I_1.shape))
train_x_I, test_x_I, train_y_I, test_y_I = model_selection.train_test_split(vectors_I_1, y_I, test_size=0.33, random_state=0)

tfidf_II_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, max_df=0.7)
vectors_II_1 = tfidf_II_1.fit_transform(x_II).toarray()
print("Vectors_II_1 shape is {} ".format(vectors_II_1.shape))

tfidf_III_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, max_df=0.7)
vectors_III_1 = tfidf_III_1.fit_transform(x_III).toarray()
print("Vectors_III_1 shape is {} ".format(vectors_III_1.shape))

state = "#3\tMax_features=5000, max_df=0.7\n"
scores_I_1 = cl.train_dataI(train_x_I, test_x_I, train_y_I, test_y_I, case1, True, state)
scores_II_1 = cl.train_cross_validate(vectors_II_1, y_II, 5, dataII, case1, state)
scores_III_1 = cl.train_cross_validate(vectors_III_1, y_III, 5, dataIII, case1, state)

#------------------------------------------------------------#

tfidf_trial = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, min_df=5, max_df=0.7)
trial_vector = tfidf_trial.fit_transform(x_trial).toarray()
print("Trial vector shape is {} ".format(trial_vector.shape))
text = tfidf_trial.transform(sentence).toarray()

tfidf_I_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, min_df=5, max_df=0.7)
vectors_I_1 = tfidf_I_1.fit_transform(x_I).toarray()
print("Vectors_I_1 shape is {} ".format(vectors_I_1.shape))
train_x_I, test_x_I, train_y_I, test_y_I = model_selection.train_test_split(vectors_I_1, y_I, test_size=0.33, random_state=0)

tfidf_II_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, min_df=5, max_df=0.7)
vectors_II_1 = tfidf_II_1.fit_transform(x_II).toarray()
print("Vectors_II_1 shape is {} ".format(vectors_II_1.shape))

tfidf_III_1 = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=5000, min_df=5, max_df=0.7)
vectors_III_1 = tfidf_III_1.fit_transform(x_III).toarray()
print("Vectors_III_1 shape is {} ".format(vectors_III_1.shape))

state = "#4\tMax_features=5000, min_df=5 max_df=0.7\n"
scores_I_1 = cl.train_dataI(train_x_I, test_x_I, train_y_I, test_y_I, case1, True, state)
scores_II_1 = cl.train_cross_validate(vectors_II_1, y_II, 5, dataII, case1, state)
scores_III_1 = cl.train_cross_validate(vectors_III_1, y_III, 5, dataIII, case1, state)
#print(tfidf_2.vocabulary_)

#print(tfidf.get_feature_names())

############### CLASSIFICATION STAGE ###############################
"""scores_I_1 = cl.train_dataI(train_x_I, test_x_I, train_y_I, test_y_I, case1, True)
scores_II_1 = cl.train_cross_validate(vectors_II_1, y_II, 5, dataII, case1)
scores_III_1 = cl.train_cross_validate(vectors_III_1, y_III, 5, dataIII, case1)"""
