# LOGISTIC REGRESSION

import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
PROBLEM_SIZE = 15
PROBLEM_DIMENSION = 3
TERMINATION_SIZE = 0.0000001
symbol = ['o', 'x']

# =========== THE HYPOTHESIS FUNCTION ============= #
def hypothesisFunction(x, theta):
	return sigmoidFunction(np.inner(theta, x))

# =========== THE SIGMOID FUNCTION ============ #
def sigmoidFunction(innerProduct):
	return 1/(1+np.exp(-1 * (innerProduct)))
	
# =========== GRADIENT DESCENT ============= #
def gradientDescent(x, y, theta):
	for j in range(0, PROBLEM_DIMENSION):
		sum = 0
		for i in range(0, PROBLEM_SIZE):
			sum += (hypothesisFunction(x[i], theta) - y[i]) * x[i][j]
		theta[j] = theta[j] - (LEARNING_RATE / PROBLEM_SIZE) * sum
	return theta
	
def costFunction(x, y, theta):
	sum = 0
	for i in range(0, PROBLEM_SIZE):
		h = hypothesisFunction(x[i], theta)
		sum = sum + y[i] * np.log(h) + (1-y[i]) * np.log(1-h)	
	return -sum / PROBLEM_SIZE

def logisticRegression(x, y, theta):
	
	costNew = costFunction(x, y, theta)
	iterationCount = 0
	costFunctionArray = [costNew]
	
	while True and iterationCount<100000:
		costOld = costNew
		theta = gradientDescent(x, y, theta)
		costNew = costFunction(x, y, theta)
		costFunctionArray.append(costNew)
		
		if iterationCount % 5000 == 0:
			print "Rd:%d\t cost(%f, %f)\t theta(%s)" % (iterationCount, costOld, costNew, theta)
		
		if np.absolute(costOld - costNew) <= TERMINATION_SIZE:
			break
		iterationCount += 1
		
	# VISUALIZATION START 
	
	# EQUATION OF THE DECISION BOUNDARY 
	hx = np.arange(0, x.max(axis=0)[1])
	hy = (-theta[0] - theta[1] * hx) / theta[2]
	plt.plot(hx, hy)
	
	# PLOT DATA POINTS 
	plt.scatter(x[:,1], x[:,2], c=y)
	plt.show()
	
if __name__ == '__main__':
	inputDataSet = np.array([[1,2,0],[2,5,0],[20,19,1],[3,4,0],[18,18,1],[15,17,1],[3,3,0],[6,1,0],[5,20,0],[15,15,1],[12,19,1],[10,9,1],[8,7,0],[17,15,1],[18,15,1]])
	
	x = inputDataSet[:, [0,1]]
	x = np.append(np.ones((PROBLEM_SIZE,1)), x, 1)
	
	print x
	y = inputDataSet[:, 2]
	
	# INITIALIZING THETA VALUES
	theta = np.array([-0.5, 0.3, -0.1])

	logisticRegression(x, y, theta)