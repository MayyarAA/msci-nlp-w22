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
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download('punkt')
print("test")

#get text from file
def retriveTextFromFile(filePath):
  with open(filePath) as f:
    posFileString = f.readlines()
    return posFileString


def removeSpecailCharFromList(type,shouldIncludeType,orgWordList):
  wordList = [];
  for curr in orgWordList:
    if shouldIncludeType:
      word = type + ":" + removeSpecailCharFromString(curr)
    else:
      word = removeSpecailCharFromString(curr)
    wordList.append(word)
  return wordList;

def removeSpecailCharFromString(orgWord):
  return re.sub(r"[^a-zA-Z0-9]+", ' ', orgWord).strip()
listOfWordsTraining = []
def generateTwoDArray(list):
  column_name = ["Tag","Sentence"]
  twoD_list = pd.DataFrame(columns = column_name)
  #for word in list:
  i=0
  while i<len(list):
    word = list[i]
    tag = word[:3]
    tag_binary = 1 if tag=="pos" else 0
    wordWithOutSpecialChars = removeSpecailCharFromString(word[4:])
    #tokenizedSentence = tok.word_tokenize(wordWithOutSpecialChars)
    #listOfWordsTraining.append(tokenizedSentence)
    twoD_list.loc[i] = [tag_binary,wordWithOutSpecialChars]
    i+=1;
  return twoD_list

# Join various path components for the input files

tempPath = "../../a1/20731123_malatari/data/"
#path_To_out_csv =os.path.join(str(sys.argv[1]),"out.csv")
#path_To_train_csv =os.path.join(str(sys.argv[1]),"labels-train.csv")
path_To_train_csv =os.path.join(tempPath,"labels-train-temp.csv")
path_To_test_csv =os.path.join(tempPath,"labels-test-temp.csv")
#path_To_val_csv =os.path.join(str(sys.argv[1]),"val.txt")
#path_To_test_csv =os.path.join(str(sys.argv[1]),"test.txt")
#pathToStopWordstxt = os.path.join("./nltkstopwordslist.txt")


#read file inputs into project
train_string_list = retriveTextFromFile(path_To_train_csv)
test_string_list = retriveTextFromFile(path_To_test_csv)

#withstop words
#create 2d dataframe[tag,senetence]
train_df_tagged = generateTwoDArray(train_string_list)
test_df_tagged = generateTwoDArray(test_string_list)


def NLBUnigram(type, train_df_tagged, test_df_tagged):
  # vectorize the training data set
  cv_unigram = CountVectorizer(ngram_range=(1, 1))
  train_unigram_mapping = cv_unigram.fit_transform(train_df_tagged["Sentence"].values.tolist())
  # vectorize the testing data set
  test_unigram_mapping = cv_unigram.transform(test_df_tagged["Sentence"].values.tolist())
  # train & predict MultinomialNB on unigram
  NLB = MultinomialNB()
  model_unigrams = NLB.fit(train_unigram_mapping, train_df_tagged["Tag"].values.tolist())
  predictions_unigrams = model_unigrams.predict(test_unigram_mapping)
  print(predictions_unigrams)
  score_val_unigrams = metrics.accuracy_score(train_df_tagged["Tag"].values.tolist(), predictions_unigrams)
  print("score_val_unigrams ",type," " , score_val_unigrams)

def NLBBigram(type, train_df_tagged, test_df_tagged):
  # bigrams:vectorize the training data set
  cv_digrams = CountVectorizer(ngram_range=(2, 2))
  train_digrams_mapping = cv_digrams.fit_transform(train_df_tagged["Sentence"].values.tolist())
  # bigrams:vectorize the testing data set
  test_bigrams_mapping = cv_digrams.transform(test_df_tagged["Sentence"].values.tolist())
  # bigrams:train & predict MultinomialNB on bigram
  NLB_bigrams = MultinomialNB()
  model_digrams = NLB_bigrams.fit(train_digrams_mapping, train_df_tagged["Tag"].values.tolist())
  predictions_bigrams = model_digrams.predict(test_bigrams_mapping)
  print(predictions_bigrams)
  score_val_bigrams = metrics.accuracy_score(train_df_tagged["Tag"].values.tolist(), predictions_bigrams)
  print("score_val_bigrams ",type," ", score_val_bigrams)

def NLBUnigramsBigram(type, train_df_tagged, test_df_tagged):
  # unigrams+bigrams with stopwords vectorization
  cv_unigrams_digrams = CountVectorizer(ngram_range=(1, 2))
  train_unigrams_digrams_mapping = cv_unigrams_digrams.fit_transform(train_df_tagged["Sentence"].values.tolist())
  # unigrams+bigrams:vectorize the training data set
  cv_unigrams_digrams = CountVectorizer(ngram_range=(1, 2))
  train_unigrams_digrams_mapping = cv_unigrams_digrams.fit_transform(train_df_tagged["Sentence"].values.tolist())
  # unigrams+bigrams:vectorize the testing data set
  test_unigrams_bigrams_mapping = cv_unigrams_digrams.transform(test_df_tagged["Sentence"].values.tolist())
  # unigrams+bigrams:train & predict MultinomialNB on unigram
  NLB_unigrams_bigrams = MultinomialNB()
  model_unigrams_digrams = NLB_unigrams_bigrams.fit(train_unigrams_digrams_mapping,
                                                    train_df_tagged["Tag"].values.tolist())
  predictions_unigrams_bigrams = model_unigrams_digrams.predict(test_unigrams_bigrams_mapping)
  print(predictions_unigrams_bigrams)
  score_val_unigrams_bigrams = metrics.accuracy_score(train_df_tagged["Tag"].values.tolist(),
                                                      predictions_unigrams_bigrams)
  print("score_val_unigrams_bigrams ",type, " ", score_val_unigrams_bigrams)

#unigram with stopwords
NLBUnigram("withstopwords",train_df_tagged,test_df_tagged)
#bigrams with stopwords
NLBBigram("withstopwords",train_df_tagged,test_df_tagged)
#unigrams+bigrams with stopwords
NLBUnigramsBigram("withstopwords",train_df_tagged,test_df_tagged)

# #without stopwords
# Join various path components for the input files
path_To_train_withoutstopwords_csv =os.path.join(tempPath,"labels-train_ns-temp.csv")
path_To_test_withoutstopwords_csv =os.path.join(tempPath,"labels-test_ns-temp.csv")
#read file inputs into project
train_string_list_withoutstopwords = retriveTextFromFile(path_To_train_withoutstopwords_csv)
test_string_list_withoutstopwords = retriveTextFromFile(path_To_test_withoutstopwords_csv)
#create 2d dataframe[tag,senetence]
train_df_tagged_withoutstopwords = generateTwoDArray(train_string_list_withoutstopwords)
test_df_tagged_withoutstopwords = generateTwoDArray(test_string_list_withoutstopwords)


#unigram without stopwords
NLBUnigram("withoutstopwords",train_df_tagged_withoutstopwords,test_df_tagged_withoutstopwords)
#bigrams without stopwords
NLBBigram("withoutstopwords",train_df_tagged_withoutstopwords,test_df_tagged_withoutstopwords)
#unigrams+bigrams without stopwords
NLBUnigramsBigram("withoutstopwords",train_df_tagged_withoutstopwords,test_df_tagged_withoutstopwords)
