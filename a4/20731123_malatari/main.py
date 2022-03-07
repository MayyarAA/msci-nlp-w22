import gensim.models
import os
import pickle
import pandas as pd
import numpy as np
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
from keras.layers.embeddings import Embedding
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_SENT_LEN = 30
MAX_VOCAB_SIZE = 20000
LSTM_DIM = 128
EMBEDDING_DIM = 300
#BATCH_SIZE = 500
BATCH_SIZE = 32
#N_EPOCHS = 10

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
        listOfWordsTraining.append([ wordWithOutSpecialChars,tag_binary])
        i += 1;
    return listOfWordsTraining


def build_models(modelType):
    ##build model
    model_input_dim = 78099
    model_sigmoid = Sequential();
    # 145
    model_sigmoid.add(Embedding(input_dim=model_input_dim, output_dim=300, weights=[embeddings_matrix], trainable=False,
                                mask_zero=True))
    # model_sigmoid.add(Dense(64, input_dim=8, activation='sigmoid'))
    model_sigmoid.add(Dense(64, input_dim=8, activation=modelType, kernel_regularizer=l2(0.001)))
    model_sigmoid.add(Dropout(0.4))
    model_sigmoid.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001)))
    model_sigmoid.add(Flatten())
    model_sigmoid.summary()

    return model_sigmoid


##start of loading data
# Join various path components for the input files
tempPathFirstDirectory = "."
tempPathSecondDirectory = "a1-input-data"
path_To_train_csv = os.path.join(tempPathFirstDirectory,tempPathSecondDirectory, "labels-train.csv")
path_To_test_csv = os.path.join(tempPathFirstDirectory,tempPathSecondDirectory, "labels-test.csv")

path_To_val_csv = os.path.join(".", "testtxt.csv")
train_string_list = retriveTextFromFile(path_To_train_csv)
test_string_list = retriveTextFromFile(path_To_test_csv)
validation_string_list = retriveTextFromFile(path_To_val_csv)


list_trainging = generateTwoDArray(train_string_list)
list_testing = generateTwoDArray(test_string_list)
list_validation = generateTwoDArray(validation_string_list)
df_training = pd.DataFrame(list_trainging,columns=['X','Y'])
df_testing =  pd.DataFrame(list_testing,columns=['X','Y'])


def tokenize_padd_data_prep(df_training):
    ##tokenize training set
    word_seq_training = [text_to_word_sequence(sent) for sent in df_training["X"]]
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq_training])
    saveTokenizer(tokenizer)
    tokenized_trainging_set = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq_training])
    training_set = pad_sequences(tokenized_trainging_set,maxlen = MAX_SENT_LEN,padding='post', truncating='post')
    return training_set

def saveTokenizer(tok):
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def testModel(modelType,model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # test the model

    # model_sigmoid.fit(X,Y,batch_size=BATCH_SIZE, epochs=10,validation_data=(X,Y))
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=10, validation_data=(X_Testing, Y_Testing))
    score, acc = model.evaluate(X_Testing, Y_Testing, batch_size=BATCH_SIZE)
    scoreStringVal = "model => " + modelType + " score => " + str(score) + " accur => " + str(acc)
    print(scoreStringVal)

def saveModel(modelType,model):
    ##save models
    print("saving models started")
    path_To_output = os.path.join(".", "data")
    path_to_output = os.path.join(path_To_output, "nn_"+modelType+".model")
    model.save(path_to_output)
    model.save("modelType" + ".h5")

#MAX_SENT_LEN = len(max(word_seq_training, key=len))



## add input layer word2vec
##take in word2vec from a3
pathToNoPickledModel = os.path.join(".","a3models","w2v.model")
word2vecModel = Word2Vec.load(pathToNoPickledModel)
word2vecModel.wv.save_word2vec_format('word2vmodel.bin', binary=True)
W2V_DIR = os.path.join('.','word2vmodel.bin')
embeddings = gensim.models.KeyedVectors.load_word2vec_format(W2V_DIR, binary=True, limit=500000)
##############++++++++++++++++++++++++++++++++++++++>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>$$$$$$$$$$$$$$$$$$$$$$
#tokenizer = Tokenizer(num_words=1500000)
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts( df_training["X"])

sequences = tokenizer.texts_to_sequences(df_training["X"])
word_index = tokenizer.word_index
review_pad = pad_sequences(sequences)
label = df_training['Y'].values

num_words = len(word_index) + 1
embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(num_words, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    try:
        embeddings_vector = embeddings[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector


training_set_labels = df_training["Y"]
X = tokenize_padd_data_prep(df_training)
Y = np.asarray(training_set_labels)


testing_set_labels = df_testing["Y"]
X_Testing=tokenize_padd_data_prep(df_testing)
Y_Testing=np.asarray(testing_set_labels)





model_sigmoid = build_models("sigmoid")
model_relu = build_models("relu")
model_tanh = build_models("tanh")


testModel("sigmoid",model_sigmoid)
testModel("relu",model_relu)
testModel("tanh",model_tanh)


saveModel("sigmoid",model_sigmoid)
saveModel("relu",model_relu)
saveModel("tanh",model_tanh)
