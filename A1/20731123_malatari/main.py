import re
import random
import csv
import sys
import os


#get text from file
def retriveTextFromFile(filePath):
  with open(filePath) as f:
    posFileString = f.readlines()
    return posFileString


#2 Remove the following special characters: !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n
#remove special char
def removeSpecailCharFromList(orgWordList):
  wordList = [];
  for curr in orgWordList:
    word = removeSpecailCharFromString(curr)
    wordList.append(word)
  return wordList;


def removeSpecailCharFromString(orgWord):
  return re.sub(r"[^a-zA-Z0-9]+", ' ', orgWord).strip()

#create hashset to store stopwords
def createHashSetFromWordList(wordList):
    wordHashSet=set()
    for curr in wordList:
      wordHashSet.add(curr)
    return wordHashSet

#tokenize wordlist
def tokenizeWordList(wordList):
  tokenizedWordList = [];
  for word in wordList:
    wordArray = word.split()
    tokenizedWordList.append(wordArray)
  return tokenizedWordList

#3 Create two versions of your dataset:
def removeStopWords(tokenizedWordList,wordHashSet):
  tokenizedWordListWithoutStopWords = []
  for wordArr in tokenizedWordList:
    wordArrWithoutStopWords = []
    for word in wordArr:
      if word not in wordHashSet:
        wordArrWithoutStopWords.append(word)
    tokenizedWordListWithoutStopWords.append(wordArrWithoutStopWords)
  return tokenizedWordListWithoutStopWords



#Create Output files
def writeToFileWithListV2(list,filename):
  outputFile = open(filename,"w")
  csvWriter = csv.writer(outputFile);
  for line in list:
    #lineString = .join(str(line))
    csvWriter.writerow(line)
  outputFile.close()




####creating relative paths in a manner that will run on all OS

# Join various path components for the output data folder
pathToDataFolderForOutputFiles = "./data"
pathTotokenizedWordList =os.path.join(pathToDataFolderForOutputFiles,  "out.csv")
pathTotokenizedWordListTraining =os.path.join(pathToDataFolderForOutputFiles,  "train.csv")
pathTotokenizedWordListValidation = os.path.join(pathToDataFolderForOutputFiles,  "val.csv")
pathTotokenizedWordListTesting = os.path.join(pathToDataFolderForOutputFiles,  "test.csv")

pathTotokenizedWordListWithoutStopWords = os.path.join(pathToDataFolderForOutputFiles,  "out_ns.csv")
pathTotokenizedWordListWithoutStopWordsTraining = os.path.join(pathToDataFolderForOutputFiles,  "train_ns.csv")
pathTotokenizedWordListWithoutStopWordsValidation = os.path.join(pathToDataFolderForOutputFiles,  "val_ns.csv")
pathTotokenizedWordListWithoutStopWordsTesting = os.path.join(pathToDataFolderForOutputFiles,  "test_ns.csv")

# Join various path components for the input files
pathToNegtxt = os.path.join(str(sys.argv[1]),  "neg.txt")
pathToPostxt = os.path.join(str(sys.argv[1]),  "pos.txt")
pathToStopWordstxt = os.path.join("./nltkstopwordslist.txt")

##### done creating paths

##start of script calls
#take in cmd line args
posFilePath = pathToNegtxt
negFilePath = pathToPostxt
stopWordsFilePath = pathToStopWordstxt



#read file inputs into project
posFileString = retriveTextFromFile(posFilePath)
negFileString = retriveTextFromFile(posFilePath)
stopWordsList = retriveTextFromFile(stopWordsFilePath)

#####2 Remove the following special characters: !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n

#remove specailchars from wordlists
wordListWithoutSpecChar = removeSpecailCharFromList(posFileString)
wordListWithoutSpecCharNeg = removeSpecailCharFromList(negFileString)
stopWordsWithOutSpecChar = removeSpecailCharFromList(stopWordsList)

#create hashset to store stopwords
stopWordsHashSet = createHashSetFromWordList(stopWordsWithOutSpecChar)
print('out' in stopWordsHashSet)



#####1 Tokenize the corpus
len(wordListWithoutSpecChar)
#combine postive and negative lists into wordList
wordListWithoutSpecChar.extend(wordListWithoutSpecCharNeg)
len(wordListWithoutSpecChar)
#tokenize wordlist
tokenizedWordList = tokenizeWordList(wordListWithoutSpecChar)


#####3 Create two versions of your dataset:
#(3.1) with stopwords

#(3.2) without stopwords.

#remove stopwords
#randomize shuffle of lists
random.shuffle(tokenizedWordList)
tokenizedWordListWithoutStopWords = removeStopWords(tokenizedWordList,stopWordsHashSet)

#####4 Randomly split your data into training (80%), validation (10%) and test (10%) sets
print(' tokenizedWordListWithoutStopWords len ', len(tokenizedWordListWithoutStopWords))

#split list w/o stopwords
tokenizedWordListWithoutStopWordsTraining =  tokenizedWordListWithoutStopWords[:int((len(tokenizedWordListWithoutStopWords)+1)*.8)]
print(' tokenizedWordListWithoutStopWordsTraining len ', len(tokenizedWordListWithoutStopWordsTraining))

tokenizedWordListWithoutStopWordsValidation =  tokenizedWordListWithoutStopWords[int((len(tokenizedWordListWithoutStopWords)+1)*.8):int((len(tokenizedWordListWithoutStopWords)+1)*.9)]
print(' tokenizedWordListWithoutStopWordsValidation len ', len(tokenizedWordListWithoutStopWordsValidation))

tokenizedWordListWithoutStopWordsTesting =  tokenizedWordListWithoutStopWords[int((len(tokenizedWordListWithoutStopWords)+1)*.9):int((len(tokenizedWordListWithoutStopWords)+1)*1)]
print(' tokenizedWordListWithoutStopWordsTesting len ', len(tokenizedWordListWithoutStopWordsTesting))


#split list w/ stopwords
print(' tokenizedWordList len ', len(tokenizedWordList))

tokenizedWordListTraining =  tokenizedWordList[:int((len(tokenizedWordList)+1)*.8)]
print(' tokenizedWordListTraining len ', len(tokenizedWordListTraining))

tokenizedWordListValidation =  tokenizedWordList[int((len(tokenizedWordList)+1)*.8):int((len(tokenizedWordList)+1)*.9)]
print(' tokenizedWordListValidation len ', len(tokenizedWordListValidation))

tokenizedWordListTesting =  tokenizedWordList[int((len(tokenizedWordList)+1)*.9):int((len(tokenizedWordList)+1)*1)]
print(' tokenizedWordListTesting len ', len(tokenizedWordListTesting))


#write to output files
writeToFileWithListV2(tokenizedWordList,pathTotokenizedWordList)
writeToFileWithListV2(tokenizedWordListTraining,pathTotokenizedWordListTraining)
writeToFileWithListV2(tokenizedWordListValidation,pathTotokenizedWordListValidation)
writeToFileWithListV2(tokenizedWordListTesting,pathTotokenizedWordListTesting)

writeToFileWithListV2(tokenizedWordListWithoutStopWords,pathTotokenizedWordListWithoutStopWords)
writeToFileWithListV2(tokenizedWordListWithoutStopWordsTraining,pathTotokenizedWordListWithoutStopWordsTraining)
writeToFileWithListV2(tokenizedWordListWithoutStopWordsValidation,pathTotokenizedWordListWithoutStopWordsValidation)
writeToFileWithListV2(tokenizedWordListWithoutStopWordsTesting,pathTotokenizedWordListWithoutStopWordsTesting)


##### end of script calls