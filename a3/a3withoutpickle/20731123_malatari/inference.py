import sys
import os
import re
import nltk.tokenize as tok
from gensim.models import Word2Vec
import pickle

def retriveTextFromFile(filePath):
  with open(filePath) as f:
    fileString = f.readlines()
    print("File data has been retrived")
    return fileString

def remove_white_space_chars(inputStringValues):
    list = []
    for word in inputStringValues:
        orgWord = word.rstrip('\n')
        list.append(orgWord)
    print("words have been parsed and white space removed")
    return list
def predictSimilarWords(inputStringValues):
    print("pickled version results ")
    for word in inputStringValues:
        sims = model.wv.most_similar(word, topn=20)
        print(f'for the word: {word} it is similar to: {sims}')

def predictSimilarWordsNoPickle(inputStringValues):
    print("non pickled version results ")
    for word in inputStringValues:
        sims = modelNoPickle.wv.most_similar(word, topn=20)
        print(f'for the word: {word} it is similar to: {sims}')

##render model
pathToPickledModel = os.path.join(".","data","w2v.model.pkl")
model = pickle.load(open(pathToPickledModel, 'rb'))
pathToNoPickledModel = os.path.join(".","data","w2v.model")
modelNoPickle = Word2Vec.load(pathToNoPickledModel)
##take in user input
pathToInputFile  = str(sys.argv[1])
#pathToInputFile = './testtext.txt'
inputStringValues = retriveTextFromFile(pathToInputFile)

#remove whitespace chars
inputStringValues= remove_white_space_chars(inputStringValues)
##pass input to prediction model
predictSimilarWords(inputStringValues)

predictSimilarWordsNoPickle(inputStringValues)


