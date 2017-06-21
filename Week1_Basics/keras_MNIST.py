# Sreela Kodali - MNIST on Keras
# June 20th, 2017  

from __future__ import print_function                                          
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.optimizers import RMSprop, SGD
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(7)

# Part 1: Load Data                                          
#load data
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

# extract sections of xtrain and xtest
xtrain = xtrain.reshape(60000, 784)
xtest = xtest.reshape(10000, 784)

# casting each array
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')

# making array values from 0 to 1 
xtrain /= 255
xtest /= 255

#printing n for each data set
print(xtrain.shape[0], 'train samples')
print(xtest.shape[0], 'test samples')

# converting class vectors to binary class matrices
ytrain = keras.utils.to_categorical(ytrain, 10)
ytest = keras.utils.to_categorical(ytest, 10)

#print(xtrain.shape)
#print(xtest.shape)
#print(ytrain.shape)
#print(ytest.shape)

## CONTROL PARAMETERS
l_r = 0.5
p = 0.6
d = 0.0
n_nodesL1 = 25
n_nodesL2 = 12
n_nodesL3 = 10

opt = SGD(lr=l_r, momentum=p, decay=d, nesterov=False)

# Part 2: Define Model
model = Sequential()
model.add(Dense(n_nodesL1, input_shape=(784,),  activation='relu'))
model.add(Dense(n_nodesL2, activation='relu'))
model.add(Dense(n_nodesL3, activation='sigmoid'))

# Part 3: Compile Model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

epoch = [3, 6, 9, 12, 15]
trainP = numpy.zeros(5)
testP = numpy.zeros(5)
i = 0
for ep in epoch:
	# Fit Model
	print('---EPOCH #:', ep, '---')
	model.fit(xtrain, ytrain, epochs=ep, batch_size=10, verbose=0, validation_data=(xtest, ytest))
	trainPerformance = model.evaluate(xtrain, ytrain, verbose=0)
	print('Epoch #:', ep, 'Train loss:', trainPerformance[0])
	print('Epoch #:', ep, 'Train accuracy:', trainPerformance[1])
	trainP[i] = trainPerformance[1]

	testPerformance = model.evaluate(xtest, ytest, verbose=0)
	print('Epoch #:', ep, 'Test loss:', testPerformance[0])
	print('Epoch #:', ep, 'Test accuracy:', testPerformance[1])
	testP[i] = testPerformance[1]
	i += 1
fig = plt.figure()
plt.plot(epoch, trainP, epoch, testP)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.figlegend((trainP, testP), ('Train Accuracy', 'Test Accuracy'), 'upper right')
plt.show()

# Bonus: Predictions; inputting data and getting corresponding predictions                             
# Note: in this scenario, using the same data
   
#predictions = model.predict(X)
#round predictions                                                                                     
#rounded = [round(x[0]) for x in predictions]
#print(rounded)