# Sreela Kodali - First Network on Keras
# June 19th, 2017

# Initialize random number generator

from keras.models import Sequential
from keras.layers import Dense
import numpy
# set random seed for reproducibility
numpy.random.seed(7)

# Part 1: Load Data

# loading the pima indians diabetes dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# separating into input (X) and output (Y)
X = dataset[:,0:8]
Y = dataset[:,8]

# Part 2: Define Model
model = Sequential()
model.add(Dense(12, input_dim=8,  activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Part 3: Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Part 4: Fit Model
model.fit(X, Y, epochs=150, batch_size=10)

# Part 5: Evaluate Model
# Note: using the same train data to evaluate --> train accuracy

#performance = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], performance[1]*100))

# Bonus: Predictions; inputting data and getting corresponding predictions
# Note: in this scenario, using the same data
predictions = model.predict(X)
#round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
