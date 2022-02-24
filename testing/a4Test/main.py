from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

##import data
# load the dataset
dataset = loadtxt('/Users/mayyaral-atari/Desktop/work/uni/msci598/assignments/testing/a4Test/pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
print("parsed file")
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print("built model")

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("using binary_crossentropy as the lose fcn")

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
print("model trainging")

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))