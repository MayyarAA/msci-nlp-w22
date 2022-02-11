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
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# nltk.download('punkt')
# print("test")

# get text from file
def retriveTextFromFile(filePath):
    with open(filePath) as f:
        posFileString = f.readlines()
        print("retrieved data from file ", filePath)
        return posFileString


def removeSpecailCharFromString(orgWord):
    return re.sub(r"[^a-zA-Z0-9]+", ' ', orgWord).strip()




def generateTwoDArray(list):
    # for word in list:
    listOfWordsTraining = []
    i = 0
    while i < len(list):
        word = list[i]
        tag = word[:3]
        tag_binary = 1 if tag == "pos" else 0
        # wordWithOutSpecialChars = removeSpecailCharFromString(word[4:])
        wordWithOutSpecialChars = word[4:].replace(',', ' ')
        wordWithOutSpecialChars = wordWithOutSpecialChars.rstrip('\n')
        listOfWordsTraining.append([tag_binary, wordWithOutSpecialChars])
        # listOfWordsTraining.append([tag_binary, word[4:]])
        i += 1;
    return listOfWordsTraining

def generateTwoDArrayV2(list):
    # for word in list:
    listOfWordsTraining = []
    i = 0
    while i < len(list):
        word = list[i]
        tag = word[:1]
        tag_binary = 1 if tag == "p" else 0
        # wordWithOutSpecialChars = removeSpecailCharFromString(word[4:])
        wordWithOutSpecialChars = word[2:].replace(',', ' ')
        wordWithOutSpecialChars = wordWithOutSpecialChars.rstrip('\n')
        listOfWordsTraining.append([tag_binary, wordWithOutSpecialChars])
        # listOfWordsTraining.append([tag_binary, word[4:]])
        i += 1;
    return listOfWordsTraining

def column(matrix, i):
    return [row[i] for row in matrix]


# Join various path components for the input files

#tempPath = "../../a1/20731123_malatari/data/"
#tempPath = "../../a1/20731123_malatari_usingtlk/data/"
tempPath = "./a1-input-data/"
# path_To_out_csv =os.path.join(str(sys.argv[1]),"out.csv")
# path_To_train_csv =os.path.join(str(sys.argv[1]),"labels-train.csv")
# path_To_train_csv =os.path.join(tempPath,"labels-train-temp.csv")
# path_To_test_csv =os.path.join(tempPath,"labels-test-temp.csv")
path_To_train_csv = os.path.join(tempPath, "labels-train.csv")
path_To_test_csv = os.path.join(tempPath, "labels-test.csv")
path_To_val_csv = os.path.join(tempPath, "labels-val.csv")
# path_To_train_csv = os.path.join(tempPath, "trainv2.csv")
# path_To_test_csv = os.path.join(tempPath, "testv2.csv")
# path_To_val_csv =os.path.join(str(sys.argv[1]),"val.txt")
# path_To_test_csv =os.path.join(str(sys.argv[1]),"test.txt")
# pathToStopWordstxt = os.path.join("./nltkstopwordslist.txt")


# read file inputs into project
train_string_list = retriveTextFromFile(path_To_train_csv)
test_string_list = retriveTextFromFile(path_To_test_csv)
validation_string_list = retriveTextFromFile(path_To_val_csv)
# withstop words
# create 2d list[tag,senetence]
train_df_tagged = generateTwoDArray(train_string_list)
print("finished generating training 2dlist")
test_df_tagged = generateTwoDArray(test_string_list)
print("finished generating testing 2dlist")
validation_df_tagged = generateTwoDArray(validation_string_list)
print("finished generating validation 2dlist")




def NLBUnigramV2(type, train_df_tagged_sentence, train_df_tagged_tag, test_df_tagged_sentence, test_df_tagged_tag,validation_df_tagged_sentence,validation_df_tagged_tag):
    cv_unigram = CountVectorizer(ngram_range=(1, 1));
    # vectorize the training data set
    stringVal = "score_val_unigrams "
    train_unigram_mapping = cv_unigram.fit_transform(train_df_tagged_sentence)
    # vectorize the testing data set
    inference_unigram_mapping = cv_unigram.transform(inference_list)
    # train & predict MultinomialNB on unigram
    NLB = MultinomialNB()
    #tuneModel(NLB)
    model_unigrams = NLB.fit(train_unigram_mapping, train_df_tagged_tag)
    #tweak the model on validation set
    predictions_unigrams = model_unigrams.predict(inference_unigram_mapping)
    score_val_unigrams = metrics.accuracy_score(test_df_tagged_tag, predictions_unigrams)
    print(stringVal, type, " ", score_val_unigrams)

def NLBUnigramV3WithPickle(type, train_df_tagged_sentence, train_df_tagged_tag, test_df_tagged_sentence, test_df_tagged_tag,validation_df_tagged_sentence,validation_df_tagged_tag):
    cv_unigram = CountVectorizer(ngram_range=(1, 1));
    # vectorize the training data set
    stringVal = "score_val_unigrams "
    train_unigram_mapping = cv_unigram.fit_transform(train_df_tagged_sentence)
    # vectorize the testing data set
    test_unigram_mapping = cv_unigram.transform(test_df_tagged_sentence)
    # train & predict MultinomialNB on unigram
    NLB = MultinomialNB()
    #tuneModel(NLB)
    model_unigrams = NLB.fit(train_unigram_mapping, train_df_tagged_tag)
    pickleModel(type,model_unigrams,cv_unigram)
    #tweak the model on validation set
    predictions_unigrams = model_unigrams.predict(test_unigram_mapping)
    score_val_unigrams = metrics.accuracy_score(test_df_tagged_tag, predictions_unigrams)
    print(stringVal, type, " ", score_val_unigrams)

def pickleModel(type, model,cv_unigram):
    #filename = "pickled_model_" + type + ".sav"
    model_filename = "mnb_" + type + ".pkl"
    pickle.dump(model,open(model_filename,'wb'))
    cv_filename = "cv_" + type + ".pkl"
    pickle.dump(cv_unigram, open(cv_filename, 'wb'))
def NLBUnigramV3WithTunning(type, train_df_tagged_sentence, train_df_tagged_tag, test_df_tagged_sentence, test_df_tagged_tag,validation_df_tagged_sentence,validation_df_tagged_tag):
    cv_unigram = CountVectorizer(ngram_range=(1, 1));
    stringVal = "score_val_unigrams "
    # vectorize the training data set
    train_unigram_mapping = cv_unigram.fit_transform(train_df_tagged_sentence)
    # vectorize the validation data set
    validation_unigram_mapping = cv_unigram.transform(validation_df_tagged_sentence)
    # vectorize the testing data set
    # train & predict MultinomialNB on unigram
    NLB = MultinomialNB()

    #tweak the model on validation set
    tunnedAlphaVal = tuneModel(NLB,train_unigram_mapping,train_df_tagged_tag,validation_unigram_mapping,validation_df_tagged_tag)
    NLB.alpha = tunnedAlphaVal
    print(" tunnedAlphaVal ", tunnedAlphaVal)
    test_unigram_mapping = cv_unigram.transform(test_df_tagged_sentence)
    model_unigrams = NLB.fit(train_unigram_mapping, train_df_tagged_tag)
    predictions_unigrams = model_unigrams.predict(test_unigram_mapping)
    score_val_unigrams = metrics.accuracy_score(test_df_tagged_tag, predictions_unigrams)
    print(stringVal, type, " ", score_val_unigrams)

def tuneModel(NLB,train_unigram_mapping,train_df_tagged_tag,validation_unigram_mapping,validation_df_tagged_tag):
    finalAlphaVal=0
    currMaxScore = 0;
    i =0
    while i<=5:
        NLBCurrent = MultinomialNB()
        NLBCurrent.alpha = i;
        model = NLBCurrent.fit(train_unigram_mapping, train_df_tagged_tag)
        predictions = model.predict(validation_unigram_mapping)
        score_val_unigrams = metrics.accuracy_score(validation_df_tagged_tag, predictions)
        if score_val_unigrams>currMaxScore:
            currMaxScore = score_val_unigrams
            finalAlphaVal = i
        i+=0.5
    return finalAlphaVal
def NLBBigramV2(type, train_df_tagged_sentence, train_df_tagged_tag, test_df_tagged_sentence, test_df_tagged_tag):
    # bigrams:vectorize the training data set
    cv_digrams = CountVectorizer(ngram_range=(2, 2))
    train_digrams_mapping = cv_digrams.fit_transform(train_df_tagged_sentence)
    # bigrams:vectorize the testing data set
    test_bigrams_mapping = cv_digrams.transform(test_df_tagged_sentence)
    # bigrams:train & predict MultinomialNB on bigram
    NLB_bigrams = MultinomialNB()
    model_digrams = NLB_bigrams.fit(train_digrams_mapping, train_df_tagged_tag)
    predictions_bigrams = model_digrams.predict(test_bigrams_mapping)
    score_val_bigrams = metrics.accuracy_score(test_df_tagged_tag, predictions_bigrams)
    print("score_val_bigrams ", type, " ", score_val_bigrams)


def NLBUnigramsBigramV2(type, train_df_tagged_sentence, train_df_tagged_tag, test_df_tagged_sentence,
                        test_df_tagged_tag):
    # unigrams+bigrams with stopwords vectorization
    # unigrams+bigrams:vectorize the training data set
    cv_unigrams_digrams = CountVectorizer(ngram_range=(1, 2))
    train_unigrams_digrams_mapping = cv_unigrams_digrams.fit_transform(train_df_tagged_sentence)
    # unigrams+bigrams:vectorize the testing data set
    test_unigrams_bigrams_mapping = cv_unigrams_digrams.transform(test_df_tagged_sentence)
    # unigrams+bigrams:train & predict MultinomialNB on unigram
    NLB_unigrams_bigrams = MultinomialNB()
    model_unigrams_digrams = NLB_unigrams_bigrams.fit(train_unigrams_digrams_mapping,
                                                      train_df_tagged_tag)
    predictions_unigrams_bigrams = model_unigrams_digrams.predict(test_unigrams_bigrams_mapping)
    score_val_unigrams_bigrams = metrics.accuracy_score(test_df_tagged_tag,
                                                        predictions_unigrams_bigrams)
    print("score_val_unigrams_bigrams ", type, " ", score_val_unigrams_bigrams)


# with stopwords
train_df_tagged_sentence = column(train_df_tagged, 1);
train_df_tagged_tag = column(train_df_tagged, 0);
validation_df_tagged_sentence = column(validation_df_tagged, 1);
validation_df_tagged_tag = column(validation_df_tagged, 0);
test_df_tagged_sentence = column(test_df_tagged, 1);
test_df_tagged_tag = column(test_df_tagged, 0);
# unigram with stopwords
NLBUnigramV3WithPickle("uni", train_df_tagged_sentence, train_df_tagged_tag, test_df_tagged_sentence,test_df_tagged_tag,validation_df_tagged_sentence,validation_df_tagged_tag)
# NLBUnigramV3WithTunning("withstopwords", train_df_tagged_sentence, train_df_tagged_tag, test_df_tagged_sentence,test_df_tagged_tag,validation_df_tagged_sentence,validation_df_tagged_tag)

# bigrams with stopwords
# NLBBigramV2("withstopwords",train_df_tagged_sentence,train_df_tagged_tag ,test_df_tagged_sentence,test_df_tagged_tag)
# unigrams+bigrams with stopwords
# NLBUnigramsBigramV2("withstopwords",train_df_tagged_sentence,train_df_tagged_tag ,test_df_tagged_sentence,test_df_tagged_tag)

# # # #without stopwords
# # # Join various path components for the input files
# path_To_train_withoutstopwords_csv =os.path.join(tempPath,"labels-train_ns.csv")
# path_To_test_withoutstopwords_csv =os.path.join(tempPath,"labels-test_ns.csv")
# path_To_val_withoutstopwords_csv =os.path.join(tempPath,"labels-val_ns.csv")
# # #read file inputs into project
# train_string_list_withoutstopwords = retriveTextFromFile(path_To_train_withoutstopwords_csv)
# validation_string_list_withoutstopwords = retriveTextFromFile(path_To_val_withoutstopwords_csv)
# test_string_list_withoutstopwords = retriveTextFromFile(path_To_test_withoutstopwords_csv)
# # #create 2d dataframe[tag,senetence]
# train_df_tagged_withoutstopwords = generateTwoDArray(train_string_list_withoutstopwords)
# test_df_tagged_withoutstopwords = generateTwoDArray(test_string_list_withoutstopwords)
# validation_df_tagged_withoutstopwords = generateTwoDArray(validation_string_list_withoutstopwords)
#
# train_df_tagged_sentence_withoutstopwords = column(train_df_tagged_withoutstopwords, 1);
# train_df_tagged_tag_withoutstopwords = column(train_df_tagged_withoutstopwords, 0);
# test_df_tagged_sentence_withoutstopwords = column(test_df_tagged_withoutstopwords, 1);
# test_df_tagged_tag_withoutstopwords = column(test_df_tagged_withoutstopwords, 0);
# validation_df_tagged_sentence_withoutstopwords = column(validation_df_tagged_withoutstopwords, 1);
# validation_df_tagged_tag_withoutstopwords = column(validation_df_tagged_withoutstopwords, 0);
#
#
# # # #unigram without stopwords
# NLBUnigramV2("withoutstopwords",train_df_tagged_sentence_withoutstopwords,train_df_tagged_tag_withoutstopwords,test_df_tagged_sentence_withoutstopwords,test_df_tagged_tag_withoutstopwords,validation_df_tagged_sentence_withoutstopwords,validation_df_tagged_tag_withoutstopwords)
# # #bigrams without stopwords
# NLBBigramV2("withoutstopwords",train_df_tagged_sentence_withoutstopwords,train_df_tagged_tag_withoutstopwords,test_df_tagged_sentence_withoutstopwords,test_df_tagged_tag_withoutstopwords)
# #unigrams+bigrams without stopwords
# NLBUnigramsBigramV2("withoutstopwords",train_df_tagged_sentence_withoutstopwords,train_df_tagged_tag_withoutstopwords,test_df_tagged_sentence_withoutstopwords,test_df_tagged_tag_withoutstopwords)





####delete this:
cv_uni_path = "./cv_uni.pkl"
mnb_uni_model_path = "./mnb_uni.pkl"
def generateArray():
    inference_filepath = './infer_test.txt'

    def retriveTextFromFile(filePath):
        with open(filePath) as f:
            posFileString = f.readlines()
            print("retrieved data from file ", filePath)
            return posFileString

    inference_string_list = retriveTextFromFile(inference_filepath)
    # for word in list:
    listOfWordsTraining = []
    i = 0
    while i < len(inference_string_list):
        word = inference_string_list[i]
        wordWithOutNewLineDelimiter = word.rstrip('\n')
        wordWithOutSpecialChars = removeSpecailCharFromString(wordWithOutNewLineDelimiter)
        listOfWordsTraining.append( wordWithOutSpecialChars)
        i += 1;
    return listOfWordsTraining

def NLBUnigram():
    cv_unigram = pickle.load(open(cv_uni_path,'rb'))
    model_mnb_uni = pickle.load(open(mnb_uni_model_path, 'rb'))
    makePredictionV2SameTime(cv_unigram,model_mnb_uni)

def makePredictionV2SameTime(cv,model):
    inference_mapping = cv.transform(inference_list)
    predictions = model.predict(inference_mapping)
    printPredictions(predictions, "pickledmodel")

def printPredictions(predictions, type):
    for i in range(len(inference_list)):
        result_statement = type + inference_list[i] + str(" => ") + (str(predictions[i]))
        print(result_statement)
def NLBUnigramV2Testing(train_df_tagged_sentence,train_df_tagged_tag):
    cv_unigram = CountVectorizer(ngram_range=(1, 1));
    # vectorize the training data set
    stringVal = "score_val_unigrams "
    train_unigram_mapping = cv_unigram.fit_transform(train_df_tagged_sentence)
    inference_unigram_mapping = cv_unigram.transform(inference_list)
    NLB = MultinomialNB()
    # tuneModel(NLB)
    model_unigrams = NLB.fit(train_unigram_mapping, train_df_tagged_tag)
    predictions_unigrams = model_unigrams.predict(inference_unigram_mapping)
    printPredictions(predictions_unigrams,"originalmodelprediction")
inference_list = generateArray()
NLBUnigram()
NLBUnigramV2Testing(train_df_tagged_sentence,train_df_tagged_tag)
###delete this: