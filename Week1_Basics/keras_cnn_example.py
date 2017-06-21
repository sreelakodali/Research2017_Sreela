# Sreela Kodali - First CNN on Keras
# June 21st, 2017

import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Loading image data from MNIST
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

#display image as a test
#from matplotlib import pyplot as plt
#plt.show(xtrain[0])

#Reshape input data with image's depth for theano
xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)

print(xtrain.shape)

# casting each array
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')

# making array values from 0 to 1 
xtrain /= 255
xtest /= 255

# converting class vectors to binary class matrices
ytrain = np_utils.to_categorical(ytrain, 10)
ytest = np_utils.to_categorical(ytest, 10)

print(ytrain.shape)

#dropping rates for convolution model and fully connected model
drate1 = 0.25
drate2 = 0.5

#Setting up the Model, firs adding convolution layers and then fully connected
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', data_format='channels_first', input_shape=(1,28,28)))
print('Shape of current model output:', model.output_shape)
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(drate1))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(drate2))
model.add(Dense(10, activation='softmax'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit model with training data
model.fit(xtrain, ytrain, batch_size=32, epochs=10, verbose=1)

#evaluate the performance
performance = model.evaluate(xtrain, ytrain, verbose=1)
print('Train accuracy:', performance[1])
performance = model.evaluate(xtest, ytest, verbose=1)
print('Test accuracy:', performance[1])