# LINEAR REGRESSION 

import numpy as np 
import matplotlib.pyplot as plt

# ================= LINEAR REGRESSION PARAMETERS ================ #
LEARNING_RATE = 0.001
TERMINATION_SIZE = 0.00000001
REGULARIZATION = 0.001

# =========== HYPOTHESIS FUNCTION =============== #
def hypothesisFunction(theta, x):
	return np.inner(theta, x)

# =========== GRADIENT DESCENT ============= #
def gradientDescent(theta, x, y):
	for j in range(0, NUMBER_OF_FEATURES+1):
		sum = 0
		for i in range(0, NUMBER_OF_TRAINING_POINTS):
			sum += (hypothesisFunction(theta, x[i]) - y[i]) * x[i][j]
		regularizingWeight = LEARNING_RATE * (REGULARIZATION / NUMBER_OF_TRAINING_POINTS)
		theta[j] = theta[j] * (1 - regularizingWeight) - (LEARNING_RATE / NUMBER_OF_TRAINING_POINTS) * sum
		#theta[j] = theta[j] - (LEARNING_RATE / NUMBER_OF_TRAINING_POINTS) * sum
	return theta

# =========== COST FUNCTION: SQUARE ERROR ============= #
def costFunction(theta, x, y):
	sum = 0.0
	for i in range(0, NUMBER_OF_TRAINING_POINTS):
		sum += ((hypothesisFunction(theta, x[i])) - y[i]) ** 2
	sum = sum + REGULARIZATION * np.sum(np.square(theta))
	sum = sum / (2.0 * NUMBER_OF_TRAINING_POINTS)
	return sum		

# =========== MAIN ALGORITHM ============== #
def linearRegression(theta, x, y):
	# =========== BEGIN WITH AN INITIAL COST FUNCTION AND THETA ============== #
	costNew = costFunction(theta, x, y)
	iterationCount = 0
	costFunctionArray = [costNew]

	# =========== PERFORM GRADIENT DESCENT UNTIL CONVERGE =========== #
	while iterationCount<10000:
		costOld = costNew
		theta = gradientDescent(theta, x, y)
		costNew = costFunction(theta, x, y)
		costFunctionArray.append(costNew)
		

		print "Rd:%d\t cost(%f) Delta(%f)\t theta(%s)" % (iterationCount, costOld, costOld-costNew, theta)
		#if (np.absolute(costOld - costNew) <= TERMINATION_SIZE):
		#	break
		iterationCount += 1
	
	np.savetxt('data_trainedParameters.txt', theta)
	
	plt.figure(1)
	plt.plot(costFunctionArray)
	plt.savefig('aa.jpg')
	'''
	plt.figure(2)
	plt.plot(x[:,1], hy, x[:,1], y, 'o')
	plt.show()
	'''

	# =========== VISUALIZE THE TRAINED RESULT ON A GRAPH ============ #
	print "The trained result: %s with cost %f" % (theta, costNew)

# ========== MAIN FUNCTION FOR HANDLING I/O AND INITIALIZING PRIMITIVES ========== #
def main():
	# =========== GLOBALIZE SOME VARIABLES ============= #
	global NUMBER_OF_FEATURES, NUMBER_OF_TRAINING_POINTS
	global plotLearning

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
	print "NUMBER OF TRAINING POINTS (m) = %d" % (NUMBER_OF_TRAINING_POINTS)

	# =========== CALL THE MAIN FUNCTION OF LINEAR REGRESSION =========== #
	linearRegression(theta, x, y)

if __name__ == '__main__':	
 	main()