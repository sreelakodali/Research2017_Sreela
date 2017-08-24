# Sreela Kodali - MNIST on Keras
# June 21st, 2017  



from __future__ import print_function                                          
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l1, l2
import numpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
numpy.random.seed(7)

# Part 1: Load Data                                          
#load data
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

# reshape dataset from 28x28 images to sets of 784 pixels
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


# CONTROL PARAMETERS
l_r = 0.5
p = 0.6
d = 0.0
n_nodesL1 = 25
n_nodesL2 = 15
n_nodesL3 = 10
drate = 0.05
reg = 0.01
#kernel_regularizer=l2(reg))


#creating sgd optimizer
opt = SGD(lr=l_r, momentum=p, decay=d, nesterov=False)

# Part 2: Define Model
model = Sequential()
model.add(Dense(n_nodesL1, input_shape=(784,),  activation='relu'))
model.add(Dropout(drate))
model.add(Dense(n_nodesL2, activation='relu', bias_regularizer=l1(reg)))
model.add(Dropout(drate))
model.add(Dense(n_nodesL3, activation='sigmoid'))

# Part 3: Compile Model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#setting up variables for forloop to fit/evaluate models with different epochs
epoch = [3, 6, 9, 12, 15]
trainP = numpy.zeros(len(epoch))
testP = numpy.zeros(len(epoch))
i = 0

#fit model and evaluate train and test performance for x epochs
for x in epoch:
	print('---EPOCH #:', x, '---')
	model.fit(xtrain, ytrain, epochs=x, batch_size=10, verbose=0, validation_data=(xtrain, ytrain))
	trainPerformance = model.evaluate(xtrain, ytrain, verbose=0)
	#print('Epoch #:', x, 'Train loss:', trainPerformance[0])
	print('Epoch #:', x, 'Train accuracy:', trainPerformance[1])
	trainP[i] = trainPerformance[1]

	testPerformance = model.evaluate(xtest, ytest, verbose=0)
	#print('Epoch #:', x, 'Test loss:', testPerformance[0])
	print('Epoch #:', x, 'Test accuracy:', testPerformance[1])
	testP[i] = testPerformance[1]
	i += 1

#plot train and test accuracy by # of epochs
fig = plt.figure()
line1, = plt.plot(epoch, trainP, label='Train')
line2, = plt.plot(epoch, testP, label='Test')
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.legend(handles=[line1, line2])
plt.show()