# LOGISTIC REGRESSION

import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
TERMINATION_SIZE = 0.0000001
REGULARIZATION = 0.001

# =========== THE HYPOTHESIS FUNCTION ============= #
def hypothesisFunction(x, theta):
	return sigmoidFunction(np.inner(theta, x))

# =========== THE SIGMOID FUNCTION ============ #
def sigmoidFunction(innerProduct):
	return 1/(1+np.exp(-1 * (innerProduct)))
	
# =========== GRADIENT DESCENT ============= #
def gradientDescent(x, y, theta):
	for j in range(0, NUMBER_OF_FEATURES+1):
		sum = 0
		for i in range(0, NUMBER_OF_TRAINING_POINTS):
			sum += (hypothesisFunction(x[i], theta) - y[i]) * x[i][j]
		regularizingWeight = LEARNING_RATE * (REGULARIZATION / NUMBER_OF_TRAINING_POINTS)
		theta[j] = theta[j] * (1- regularizingWeight) - (LEARNING_RATE / NUMBER_OF_TRAINING_POINTS) * sum
		#theta[j] = theta[j] - (LEARNING_RATE / NUMBER_OF_TRAINING_POINTS) * sum
	return theta
	
def costFunction(x, y, theta):
	sum = 0
	for i in range(0, NUMBER_OF_TRAINING_POINTS):
		h = hypothesisFunction(x[i], theta)
		sum = sum + y[i] * np.log(h) + (1-y[i]) * np.log(1-h)	
	regularizingWeight = (REGULARIZATION / 2) * np.sum(np.square(theta))
	return (1 / NUMBER_OF_TRAINING_POINTS) * (-sum + regularizingWeight)

def logisticRegression(x, y, theta):
	
	costNew = costFunction(x, y, theta)
	iterationCount = 0
	costFunctionArray = [costNew]
	
	while True and iterationCount<300000:
		costOld = costNew
		theta = gradientDescent(x, y, theta)
		costNew = costFunction(x, y, theta)
		costFunctionArray.append(costNew)
		
		print "Rd:%d\t cost(%f) Delta(%f)\t theta(%s)" % (iterationCount, costOld, costOld-costNew, theta)
		
		'''
		if np.absolute(costOld - costNew) <= TERMINATION_SIZE:
			break
			'''
		iterationCount += 1
		
	# VISUALIZATION START 
	np.savetxt('data_trainedParameters.txt', theta)
	
	plt.figure(1)
	plt.plot(costFunctionArray)
	plt.savefig('aa.jpg')
	
	
if __name__ == '__main__':
	# =========== GLOBALIZE SOME VARIABLES ============= #
	global NUMBER_OF_FEATURES, NUMBER_OF_TRAINING_POINTS
	
	# =========== READ IN INPUT DATA FROM AN EXTERNAL FILE ============ #
	inputDataSet = np.loadtxt('data_normalizedInputData.txt')
	
	# =========== CHOPPING THE INPUT DATA INTO FEATURES(x) AND TARGET (y) =========== #
	NUMBER_OF_FEATURES = len(inputDataSet[0]) - 1
	NUMBER_OF_TRAINING_POINTS = len(inputDataSet)
	
	print inputDataSet
	
	x = inputDataSet[:, range(0, NUMBER_OF_FEATURES)]
	x = np.append(np.ones((NUMBER_OF_TRAINING_POINTS, 1)), x, 1)
	y = inputDataSet[:, NUMBER_OF_FEATURES]
	
	# =========== INITIALIZING THE VALUE OF THETA =============== #
	theta = np.zeros(NUMBER_OF_FEATURES + 1)
	print "NUMBER OF FEATURES (n) = %d" % (NUMBER_OF_FEATURES)
	print "NUMBER OF TRAINING POINTES (m) = %d" % (NUMBER_OF_TRAINING_POINTS)

	logisticRegression(x, y, theta)