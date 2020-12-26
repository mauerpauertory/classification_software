from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import re 

def custom_tokenizer(text):
    """ custom tokenizer that uses nltk.word_tokenize for the main logic

    additionaly, removes from the line: 
        single alphabetics characters
        ’ characters
        separates single quotes ' '  with space 
    and lowercases the text

    Parameters
    ----------
    text - String to process

    Returns
    -------
    tokenized line 
    
    """
    line = text.replace("'", " ")
    line = line.replace("’", " ")
    # remove all single alphabetic characters 
    line = re.sub(r'\s+[a-zA-Z]\s+', ' ', line)
    # remove single alphabetic characters in the beginning
    line = re.sub(r'^[a-zA-Z]\s+', ' ', line) 
    return word_tokenize(line.lower())


def tokenize_stemmer(text):
    """ tokenizer + stemmer method; uses custom_tokenizer first
    then applies proter_stemmer.stem method per each token 

    Parameters
    ----------
    text - String to process

    Returns
    -------
    stemmed : String
        tokenized and stemmed line 
    
    """
    ps = PorterStemmer()
    stemmed = []
    tokens = custom_tokenizer(text)
    for token in tokens:
        stemmed.append(ps.stem(token))
    return stemmed 

def full_preprocessing(text):
    """ tokenizer + stemmer + stopword removal method; 
    uses custom_tokenizer first, then removes the words from nltk.stopword corpus,
    finally applies proter_stemmer.stem method per each token 

    Parameters
    ----------
    text - String to process
    
    Returns
    -------
    preprocessed_line : String
        tokenized and stemmed line 
    
    """
    english_stopwords = set(stopwords.words('english')) 
    tokens = custom_tokenizer(text)
    ps = PorterStemmer()
    preprocessed_line = []
    for token in tokens:
        if token not in english_stopwords:
            preprocessed_line.append(ps.stem(token))    
    return preprocessed_line 
