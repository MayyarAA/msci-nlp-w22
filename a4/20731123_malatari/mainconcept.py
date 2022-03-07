import gensim.models
import os
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

# get text from file
def retriveTextFromFile(filePath):
    with open(filePath) as f:
        posFileString = f.readlines()
        print("retrieved data from file ", filePath)
        return posFileString

def generateTwoDArray(list):
    # for word in list:
    listOfWordsTraining = []
    i = 0
    while i < len(list):
        word = list[i]
        tag = word[:3]
        tag_binary = 1 if tag == "pos" else 0
        #keep in csv form
        wordWithOutSpecialChars = word[4:].replace(',', ' ')
        wordWithOutSpecialChars = wordWithOutSpecialChars.rstrip('\n')
        listOfWordsTraining.append([tag_binary, wordWithOutSpecialChars])
        i += 1;
    return listOfWordsTraining

def column(matrix, i):
    return [row[i] for row in matrix]
## take in train,val,test split from a1
# read file inputs into project

# Join various path components for the input files
tempPathFirstDirectory = "."
tempPathSecondDirectory = "a1-input-data"
path_To_train_csv = os.path.join(tempPathFirstDirectory,tempPathSecondDirectory, "labels-train.csv")
path_To_test_csv = os.path.join(tempPathFirstDirectory,tempPathSecondDirectory, "labels-test.csv")
path_To_val_csv = os.path.join(tempPathFirstDirectory,tempPathSecondDirectory, "labels-val.csv")
train_string_list = retriveTextFromFile(path_To_train_csv)
test_string_list = retriveTextFromFile(path_To_test_csv)
validation_string_list = retriveTextFromFile(path_To_val_csv)

# create 2d list[tag,senetence]
train_df_tagged = generateTwoDArray(train_string_list)
print("finished generating training 2dlist")
test_df_tagged = generateTwoDArray(test_string_list)
print("finished generating testing 2dlist")
validation_df_tagged = generateTwoDArray(validation_string_list)
print("finished generating validation 2dlist")

##take in word2vec from a3
pathToNoPickledModel = os.path.join(".","a3models","w2v.model")
word2vecModel = Word2Vec.load(pathToNoPickledModel)


##declare model
model_sigmoid = Sequential()
model_ReLU = Sequential()
model_tanh = Sequential()
## add input layer word2vec
#embeddings = gensim.models.KeyedVectors.load_word2vec_format(pathToNoPickledModel,binary=True)

model_sigmoid.add(Embedding(input_dim = len(word2vecModel.wv)+1,output_dim=word2vecModel.vector_size,input_length=1000 ))
model_ReLU.add(Embedding(input_dim = len(word2vecModel.wv)+1,output_dim=word2vecModel.vector_size,input_length=1000 ))
model_tanh.add(Embedding(input_dim = len(word2vecModel.wv)+1,output_dim=word2vecModel.vector_size,input_length=1000 ))

## add hidden layer to model
model_sigmoid.add(Dense(64, input_dim=8, activation='sigmoid'))
model_ReLU.add(Dense(64, input_dim=8, activation='relu'))
model_tanh.add(Dense(64, input_dim=8, activation='tanh'))

#add cross entropy as the loss fcn
model_sigmoid.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ReLU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_tanh.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
##add L2-norm regularization

##add dropouts, diff rates

##tokenize training set
word_seq_training = [text_to_word_sequence(sent) for sent in column(train_df_tagged, 1)]
MAX_SENT_LEN = max(word_seq_training, key=len)
print(MAX_SENT_LEN)
tokenized_trainging_set = Tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
##padding added to set
padding_traingData  =pad_sequences(column(train_df_tagged, 1))

print(padding_traingData)
##train model
model_sigmoid.fit(padding_traingData,column(train_df_tagged, 0),epochs=50, verbose=0)
##test model

##save models
print("saving models started")
path_To_output = os.path.join("../../a2/20731123_malatari", "data")
path_to_output_sigmoid = os.path.join(path_To_output,"nn.sigmoid.model")
path_to_output_relu = os.path.join(path_To_output,"nn.relu.model")
path_to_output_tanh = os.path.join(path_To_output,"nn.tanh.model")
model_sigmoid.save(path_to_output_sigmoid)

print("saving models completed")
## add output layer softmax



##save models

