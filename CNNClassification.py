#%%
import DataPreparation as prep
import ClassificationAlgorithms as cl

from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim.models import Word2Vec

import pandas as pd
import psutil

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
# Use cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)

porter_stemmer = PorterStemmer()

label = 'label'
title = 'title'
dataIII = 'III'
case1 = '1'
case2 = '2'
case3 = '3'

clickbait_III_path = './dataset/DatasetIII/clickbait'
non_clickbait_III_path = './dataset/DatasetIII/not-clickbait'
OUTPUT_FOLDER = './dataset/DatasetIII/'
######### Create Dataset III ##########################
trainDF_III = prep.constructDataset(clickbait_III_path, non_clickbait_III_path)
x_III, y_III= trainDF_III[title], trainDF_III[label]


stoplist = set(stopwords.words('english')) 
def removeStopWords(tokens): 
    return [word for word in tokens if word not in stoplist]
######## Tokenization #########################

#---- Simply tokenized ----- #
print(x_III)
trainDF_III['tokenized_title'] = [simple_preprocess(line, deacc=True) for line in trainDF_III[title]]
x_III = trainDF_III['tokenized_title'] 
print(x_III)
#----- Tokenized + stemmed ------#
trainDF_III['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in trainDF_III['tokenized_title']]
x_III = trainDF_III['stemmed_tokens']
print(x_III)
#----- Tokenized + stemmed + stopwords removal ------#
trainDF_III['title_without_stopwords'] = [removeStopWords(sen) for sen in trainDF_III['tokenized_title']]
trainDF_III['stemmed_stopwords_title'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in trainDF_III['title_without_stopwords']]
x_III = trainDF_III['stemmed_stopwords_title']
print(x_III)

size = 300
window = 3
min_count = 1
workers = 3
sg = 1

# Function to train word2vec model
def make_word2vec_model(data_x, padding=True, sg=1, min_count=1, size=300, workers=3, window=3, data_column='stemmed_tokens'):
    if  padding:
        print(len(data_x))
        temp_df = pd.Series(data_x[data_column]).values
        temp_df = list(temp_df)
        temp_df.append(['pad'])
        word2vec_file = OUTPUT_FOLDER + 'word2vec_' + str(size) + '_PAD.model'
    else:
        temp_df = data_x[data_column]
        word2vec_file = OUTPUT_FOLDER + 'word2vec_' + str(size) + '.model'
    w2v_model = Word2Vec(temp_df, min_count = min_count, size = size, workers = workers, window = window, sg = sg)

    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file


w2vmodel, word2vec_file = make_word2vec_model(trainDF_III, padding=True, sg=sg, min_count=min_count, size=size, workers=workers, window=window,  data_column='stemmed_stopwords_title')

max_sen_len = trainDF_III.stemmed_tokens.map(len).max()
print(max_sen_len)
padding_idx = w2vmodel.wv.vocab['pad'].index
print(padding_idx)
def make_word2vec_vector_cnn(sentence):
    padded_X = [padding_idx for i in range(max_sen_len)]
    i = 0
    for word in sentence:
        if word not in w2vmodel.wv.vocab:
            padded_X[i] = 0
            print(word)
        else:
            padded_X[i] = w2vmodel.wv.vocab[word].index
        i += 1
    return torch.tensor(padded_X, dtype=torch.long, device=device).view(1, -1)

    