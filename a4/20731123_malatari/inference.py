import sys
import gensim.models
import os
import pickle

import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
from keras.layers.embeddings import Embedding
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

import re
def retriveTextFromFile(filePath):
  with open(filePath) as f:
    fileString = f.readlines()
    print("File data has been retrived")
    return fileString

def removeSpecailCharFromString(orgWord):
  return re.sub(r"[^a-zA-Z0-9]+", ' ', orgWord).strip()

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
    print("list generated")
    return listOfWordsTraining

#tokenize wordlist
def tokenizeWordList(wordList):
  tokenizedWordList = [];
  for word in wordList:
    wordArray = word.split()
    tokenizedWordList.append(wordArray)
  print("input has been tokenized")
  return tokenizedWordList

def getFolderName(type):
    if type== "sigmoid":
        return "nn_sigmoid.model"
    elif type =="relu":
        return "nn_relu.model"
    return "nn_tanh.model"
def loadModel(fcnType):
    #pathToModel = os.path.join('.', "data","nn.sigmoid.model" ,'saved_model.pb')
    stringPathVal = getFolderName(fcnType)
    pathToModel = os.path.join(".", "data", stringPathVal)
    #pathToModel = os.path.join('.',"model.h5")
    model = keras.models.load_model(pathToModel)
    printStatement = fcnType +  "model has been loaded"
    print(printStatement)
    return model;


def build_printable_predictions(predictions):
    for pred in predictions:
        res= np.average(pred)
        if res >0.5 :
            res = 1
            printStatement = "positive " + str(res)
            print(printStatement)
        else:
            res=0
            printStatement = "negative " + str(res)
            print(printStatement)

def makePrediction(vals,model):
    build_printable_predictions(model.predict(vals))

def loadTokenizer():
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("tokenizer loaded")
    return tokenizer
pathToInputFile  = str(sys.argv[1])
userDefinedFcn = str(sys.argv[2])

##retreive text from input file
inputStringValues = retriveTextFromFile(pathToInputFile)

##tokenize & remove special chars
#inferenceList = tokenizeWordList(generateArray(inputStringValues))
MAX_SENT_LEN=30
word_seq = [text_to_word_sequence(sent) for sent in inputStringValues]
tokenizer = loadTokenizer()
X_predict = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
X_predict = pad_sequences(X_predict, maxlen=MAX_SENT_LEN, padding='post', truncating='post')
print("X_predict is ready")
##load model given user input
model = loadModel(userDefinedFcn)

##make predictions
makePrediction(X_predict,model)