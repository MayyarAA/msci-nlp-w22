import sys
import os
import re
import nltk.tokenize as tok
from gensim.models import Word2Vec
import pickle

def retriveTextFromFile(filePath):
  with open(filePath) as f:
    fileString = f.readlines()
    return fileString

def predictSimilarWords(inputStringValues):
    for word in inputStringValues:
        sims = model.wv.most_similar('word', topn=20)
        print(sims)
##render model
pathToPickledModel = os.path.join(".","data","w2v.model.pkl")
model = pickle.load(open(pathToPickledModel, 'rb'))

##take in user input
#pathToInputFile  = str(sys.argv[1])
pathToInputFile = './testtext.txt'
inputStringValues = retriveTextFromFile(pathToInputFile)

##pass input to prediction model
predictSimilarWords(inputStringValues)


