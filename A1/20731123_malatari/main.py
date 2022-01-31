import re
import random
import csv
import sys


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
    lineString = type#.join(str(line))
    csvWriter.writerow(line)
  outputFile.close()





##start of script calls
#take in cmd line args
posFilePath = str(sys.argv[1]) + "/pos.txt"
negFilePath =  str(sys.argv[1]) + "/neg.txt"
stopWordsFilePath =  str(sys.argv[1]) + "/nltkstopwordslist.txt"

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
tokenizedWordListWithoutStopWords = removeStopWords(tokenizedWordList,stopWordsHashSet)

#####4 Randomly split your data into training (80%), validation (10%) and test (10%) sets
print(' tokenizedWordListWithoutStopWords len ', len(tokenizedWordListWithoutStopWords))
#split list w/o stopwords
random.shuffle(tokenizedWordListWithoutStopWords)
tokenizedWordListWithoutStopWordsTraining =  tokenizedWordListWithoutStopWords[:int((len(tokenizedWordListWithoutStopWords)+1)*.8)]
print(' tokenizedWordListWithoutStopWordsTraining len ', len(tokenizedWordListWithoutStopWordsTraining), ' pop top ',tokenizedWordListWithoutStopWordsTraining.pop(0))
random.shuffle(tokenizedWordListWithoutStopWords)
tokenizedWordListWithoutStopWordsValidation =  tokenizedWordListWithoutStopWords[int((len(tokenizedWordListWithoutStopWords)+1)*.8):int((len(tokenizedWordListWithoutStopWords)+1)*.9)]
print(' tokenizedWordListWithoutStopWordsValidation len ', len(tokenizedWordListWithoutStopWordsValidation), ' pop top ',tokenizedWordListWithoutStopWordsValidation.pop(0))
random.shuffle(tokenizedWordListWithoutStopWords)
tokenizedWordListWithoutStopWordsTesting =  tokenizedWordListWithoutStopWords[int((len(tokenizedWordListWithoutStopWords)+1)*.9):int((len(tokenizedWordListWithoutStopWords)+1)*1)]
print(' tokenizedWordListWithoutStopWordsTesting len ', len(tokenizedWordListWithoutStopWordsTesting), ' pop top ',tokenizedWordListWithoutStopWordsTesting.pop(0) )

#split list w/ stopwords
random.shuffle(tokenizedWordList)
print(' tokenizedWordList len ', len(tokenizedWordList))
tokenizedWordListTraining =  tokenizedWordList[:int((len(tokenizedWordList)+1)*.8)]
print(' tokenizedWordListTraining len ', len(tokenizedWordListTraining), ' pop top ',tokenizedWordListTraining.pop(0))
random.shuffle(tokenizedWordList)
tokenizedWordListValidation =  tokenizedWordList[int((len(tokenizedWordList)+1)*.8):int((len(tokenizedWordList)+1)*.9)]
print(' tokenizedWordListValidation len ', len(tokenizedWordListValidation), ' pop top ',tokenizedWordListValidation.pop(0))
random.shuffle(tokenizedWordList)
tokenizedWordListTesting =  tokenizedWordList[int((len(tokenizedWordList)+1)*.9):int((len(tokenizedWordList)+1)*1)]
print(' tokenizedWordListTesting len ', len(tokenizedWordListTesting), ' pop top ',tokenizedWordListTesting.pop(0) )



#write to output files
writeToFileWithListV2(tokenizedWordList,"./data/out.csv")
writeToFileWithListV2(tokenizedWordListTraining,"./data/train.csv")
writeToFileWithListV2(tokenizedWordListValidation,"./data/val.csv")
writeToFileWithListV2(tokenizedWordListTesting,"./data/test.csv")

writeToFileWithListV2(tokenizedWordListWithoutStopWords,"./data/out_ns.csv")
writeToFileWithListV2(tokenizedWordListWithoutStopWordsTraining,"./data/train_ns.csv")
writeToFileWithListV2(tokenizedWordListWithoutStopWordsValidation,"./data/val_ns.csv")
writeToFileWithListV2(tokenizedWordListWithoutStopWordsTesting,"./data/test_ns.csv")


##### end of script calls