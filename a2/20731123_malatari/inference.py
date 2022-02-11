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

dataFolderPath = "./data/"

#cv_uni_path = dataFolderPath + "/cv_uni.pkl"
cv_uni_path = os.path.join(dataFolderPath,"cv_uni.pkl")
mnb_uni_model_path = os.path.join(dataFolderPath,  "mnb_uni.pkl")
#mnb_uni_model_path = dataFolderPath + "/mnb_uni.pkl"


cv_bi_path =os.path.join(dataFolderPath,"cv_bi.pkl")
    #dataFolderPath + "/cv_bi.pkl"
mnb_bi_model_path = os.path.join(dataFolderPath,"mnb_bi.pkl")
    #dataFolderPath + "/mnb_bi.pkl"

cv_uni_bi_path =os.path.join(dataFolderPath,"cv_uni_bi.pkl")
    #dataFolderPath + "/cv_uni_bi.pkl"
mnb_uni_bi_model_path = os.path.join(dataFolderPath,"mnb_uni_bi.pkl")
    #dataFolderPath + "/mnb_uni_bi.pkl"

cv_uni_ns_path =os.path.join(dataFolderPath,"cv_uni_ns.pkl")
    #dataFolderPath + "/cv_uni_ns.pkl"
mnb_uni_ns_model_path =os.path.join(dataFolderPath,"mnb_uni_ns.pkl")
    #dataFolderPath + "/mnb_uni_ns.pkl"

cv_bi_ns_path = os.path.join(dataFolderPath,"cv_bi_ns.pkl")
    #dataFolderPath + "/cv_bi_ns.pkl"
mnb_bi_ns_model_path = os.path.join(dataFolderPath,"mnb_bi_ns.pkl")
    #dataFolderPath + "/mnb_bi_ns.pkl"

cv_uni_bi_ns_path =os.path.join(dataFolderPath,"cv_uni_bi_ns.pkl")
    #dataFolderPath + "/cv_uni_bi_ns.pkl"
mnb_uni_bi_ns_model_path = os.path.join(dataFolderPath,"mnb_uni_bi_ns.pkl")
    #dataFolderPath + "/mnb_uni_bi_ns.pkl"

#terminal_input_pickle_type = "mnb_uni"
#inference_filepath = './infer_test.txt'
inference_filepath = str(sys.argv[1])
terminal_input_pickle_type = str(sys.argv[2])



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
        result_statement = sentence + str(" => ") + (str(predictions[0]))
        print(result_statement)

def makePredictionV2SameTime(cv,model):
    inference_mapping = cv.transform(inference_list)
    predictions = model.predict(inference_mapping)
    printPredictions(predictions)
    #score_val_bigrams = metrics.accuracy_score(test_df_tagged_tag, predictions_bigrams)


def printPredictions(predictions):
    for i in range(len(inference_list)):
        result_statement = inference_list[i] + str(" => ") + (str(predictions[i]))
        print(result_statement)

def NLBUnigram():
    cv_unigram = pickle.load(open(cv_uni_path,'rb'))
    model_mnb_uni = pickle.load(open(mnb_uni_model_path, 'rb'))
    makePrediction(cv_unigram,model_mnb_uni)
    #inference_unigram_mapping = cv_unigram.transform(inference_list)
    #predictions_unigrams = model_mnb_uni.predict(inference_unigram_mapping)
def NLBUnigramNS():
    cv = pickle.load(open(cv_uni_ns_path,'rb'))
    model = pickle.load(open(mnb_uni_ns_model_path, 'rb'))
    makePrediction(cv,model)


def NLBBigram():
    cv = pickle.load(open(cv_bi_path,'rb'))
    model = pickle.load(open(mnb_bi_model_path, 'rb'))
    makePrediction(cv,model)
def NLBBigramNS():
    cv = pickle.load(open(cv_bi_ns_path,'rb'))
    model = pickle.load(open(mnb_bi_ns_model_path, 'rb'))
    makePrediction(cv,model)

def NLBUnigramsBigram():
    cv = pickle.load(open(cv_uni_bi_path,'rb'))
    model = pickle.load(open(mnb_uni_bi_model_path, 'rb'))
    makePrediction(cv,model)

def NLBUnigramsBigramNS():
    cv = pickle.load(open(cv_uni_bi_ns_path,'rb'))
    model = pickle.load(open(mnb_uni_bi_ns_model_path, 'rb'))
    makePrediction(cv,model)




def load_necessary_pickles(terminal_input_pickle_type):
    if terminal_input_pickle_type =="mnb_uni":
        NLBUnigram()
        print("terminal_input_pickle_type ", "mnb_uni")
    elif terminal_input_pickle_type =="mnb_bi":
        NLBBigram()
        print("terminal_input_pickle_type ", "mnb_bi")
    elif terminal_input_pickle_type =="mnb_uni_bi":
        NLBUnigramsBigram()
        print("terminal_input_pickle_type ", "mnb_uni_bi")
    elif terminal_input_pickle_type =="mnb_uni_ns":
        NLBUnigramNS()
        print("terminal_input_pickle_type ", "mnb_uni_ns")
    elif terminal_input_pickle_type =="mnb_bi_ns":
        NLBBigramNS()
        print("terminal_input_pickle_type ", "mnb_bi_ns")
    elif terminal_input_pickle_type =="mnb_uni_bi_ns":
        NLBUnigramsBigramNS()
        print("terminal_input_pickle_type ", "mnb_uni_bi_ns")


load_necessary_pickles(terminal_input_pickle_type)
