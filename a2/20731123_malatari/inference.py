import csv
import sys
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import nltk.tokenize as tok
import ssl
import re
import pickle

inference_filepath = './infer_test.txt'
cv_uni_path = "./cv_uni.pkl"
mnb_uni_model_path = "./mnb_uni.pkl"
terminal_input_pickle_type = "mnb_uni"

def retriveTextFromFile(filePath):
    with open(filePath) as f:
        posFileString = f.readlines()
        print("retrieved data from file ", filePath)
        return posFileString

def generateArray(list):
    # for word in list:
    listOfWordsTraining = []
    i = 0
    while i < len(list):
        word = list[i]
        wordWithOutNewLineDelimiter = word.rstrip('\n')
        wordWithOutSpecialChars = removeSpecailCharFromString(wordWithOutNewLineDelimiter)
        listOfWordsTraining.append( wordWithOutSpecialChars)
        i += 1;
    return listOfWordsTraining

def removeSpecailCharFromString(orgWord):
  return re.sub(r"[^a-zA-Z0-9]+", ' ', orgWord).strip()

##take in text from file input
inference_string_list = retriveTextFromFile(inference_filepath)
inference_list = generateArray(inference_string_list)

##depending on terminal argument load the needed pkcl files




def makePrediction(cv,model):
    for sentence in inference_list:
        inference_mapping = cv.transform([sentence])
        predictions = model.predict(inference_mapping)
        #result_statement = sentence.join("=>").join(str(predictions[0]))
        result_statement = sentence + str(" => ") + (str(predictions[0]))
        print(result_statement)
def NLBUnigram():
    cv_unigram = pickle.load(open(cv_uni_path,'rb'))
    model_mnb_uni = pickle.load(open(mnb_uni_model_path, 'rb'))
    makePrediction(cv_unigram,model_mnb_uni)
    #inference_unigram_mapping = cv_unigram.transform(inference_list)
    #predictions_unigrams = model_mnb_uni.predict(inference_unigram_mapping)

def load_necessary_pickles(terminal_input_pickle_type):
    if terminal_input_pickle_type =="mnb_uni":
        NLBUnigram()
        print("terminal_input_pickle_type ", "mnb_uni")
    elif terminal_input_pickle_type =="mnb_bi":
        print("terminal_input_pickle_type ", "mnb_bi")
    elif terminal_input_pickle_type =="mnb_uni_bi":
        print("terminal_input_pickle_type ", "mnb_uni_bi")
    elif terminal_input_pickle_type =="mnb_uni_ns":
        print("terminal_input_pickle_type ", "mnb_uni_ns")
    elif terminal_input_pickle_type =="mnb_bi_ns":
        print("terminal_input_pickle_type ", "mnb_bi_ns")
    elif terminal_input_pickle_type =="mnb_uni_bi_ns":
        print("terminal_input_pickle_type ", "mnb_uni_bi_ns")

load_necessary_pickles(terminal_input_pickle_type)
