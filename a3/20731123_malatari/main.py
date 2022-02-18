import sys
import os
import re
import nltk.tokenize as tok
from gensim.models import Word2Vec
import pickle
print("here")


def retriveTextFromFile(filePath):
  with open(filePath) as f:
    fileString = f.readlines()
    print("File data has been retrived")
    return fileString

def removeSpecailCharFromString(orgWord):
    orgWord = orgWord.rstrip('\n')
    return re.sub(r"[^a-zA-Z0-9]+", ' ', orgWord).strip()

def removeSpecailCharFromList(orgWordList):
    list = []
    for sentence in orgWordList:
        tok2 = tok.word_tokenize(removeSpecailCharFromString(sentence))
        list.append(tok2)
    print("list has been tokenized and special chars removed")
    return list

def pickleModel( model,filename):
    model_filename = os.path.join(".","data",filename)
    # "./data/"
    #filename = "pickled_model_" + type + ".sav"
    pickle.dump(model,open(model_filename,'wb'))
    print("model has been pickled")



##script calls start
# pathToPosFile = os.path.join(str(sys.argv[1]),"pos.txt")
# pathToNegFile = os.path.join(str(sys.argv[1]),"neg.txt")
pathToPosFile = "./pos.txt"
pathToNegFile = "./neg.txt"
posFileData = retriveTextFromFile(pathToPosFile)
negFileData = retriveTextFromFile(pathToNegFile)
posFileData.extend(negFileData)

##tokenize
listOfSentences = removeSpecailCharFromList(posFileData)
print(listOfSentences[0])



model = Word2Vec(sentences=listOfSentences, vector_size=100, window=5, min_count=1, workers=4)
#model = Word2Vec(sentences=listOfSentences, vector_size=100, window=5, min_count=1, workers=4)
sims = model.wv.most_similar('computer', topn=20)
print(sims)
#pickle model
pickleModel(model, "w2v.model.pkl")

