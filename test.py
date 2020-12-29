#%%
#%%
import DataPreparation as prep
import ClassificationAlgorithms as cl

from PreprocessingPhase import simple_tokenizer, tokenize_stemmer, full_preprocessing

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer

import numpy as np
import time
import pandas 
import re 

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
sentence = ["8 Fall Shows To Be Excited About, 10 Shouldn't To Give A Chance, And 6 To Avoid",
             "16 Signs You Are Too Stubborn To Live",
             "“That one was definitely alive”: An undercover video at one of the nation’s biggest pork processors",
             "17 Easy Slow Cooker Soups That Will Warm You Right Up", 
             "Political Shifts on Gay Rights Lag Behind Culture", 
             "Taliban militant kills at least thirteen in northwest Pakistan",
             "17 Pictures Hot People Will Never Understand",
             "25 Reasons A Trip To Costa Rica Could Actually Change Your Life"]
sent_labels = [1, 1, 1, 1, 0, 0, 1, 1]             

stemming_example = ["They Released 14 Wolves Into A Park. What Happens Next Is A Miracle And Proves That We Must Take Care Of Our Amazing Planet.",
                    "A Stanford professor says eliminating 2 phrases from your vocabulary can make you more successful"]

ps = PorterStemmer()

stemming_example_transformed = []
for line in stemming_example:
    stemming_example_transformed.append(" ".join(tokenize_stemmer(line)))

stemming_example_transformed[0] = stemming_example_transformed[0].replace(" .", ".")
stemming_example_transformed[0] = stemming_example_transformed[0].replace("into park", "into a park")
stemming_example_transformed[0] = stemming_example_transformed[0].replace("is miracl", "is a miracl")
print()
for k,line in enumerate(stemming_example):
    print()
    print("\033[1mOriginal line #%d:\033[1m\x1b[0m \n%s\x1b[0m" % (k+1, line))
    print("\033[1mStemmed line #%d:\033[1m\x1b[0m \n%s\x1b[0m" % (k+1, stemming_example_transformed[k]))
print()
def for_thesis_tokenization():
    thesis_examples = ["Apple might finally get rid of the most disliked iPhone model", 
                    "You'll Never Guess Who Chicago Is Hiring To Help Deal With It's Rat Problem. It's Perfect!",
                    "Lil Wayne's Miami Mansion Raided... Police Find $30 Million Stash Of..."]


    print()
    print("")
    print("#1 Simple line:     \033[1m{}\033[1m\x1b[0m\x1b[0m'".format(thesis_examples[0]))
    print("Tokenized to:\t \x1B[3m{}\x1B[3m".format(custom_tokenizer(thesis_examples[0])))
    print('\x1b[0m\x1b[0m')
    print("#2 Line with special characters:")
    print("\t\033[1m{}\033[1m".format(thesis_examples[2]))
    #print("\t\x1b[0m2) \x1b[0m\033[1m{}\033[1m".format(thesis_examples[2]))
    print("\x1b[0m\x1b[0mTokenized removing \x1b[0m\033[1m'\033[1m\x1b[0m and single alphabetic characters:")
    #print("\t1) \x1B[3m{}\x1B[3m".format(custom_tokenizer(thesis_examples[1])))
    print("\t\x1B[3m{}\x1B[3m".format(custom_tokenizer(thesis_examples[2])))
    print('\x1b[0m\x1b[0m')


# %%
