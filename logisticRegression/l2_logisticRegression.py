from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

def main():
	# ========================================================================= #
	# ================== STEP 1. LOAD EXTERNAL INPUT DATA ===================== #
	# ========================================================================= #
	inputData = np.loadtxt('data_inputData.txt')
	testData = np.loadtxt('data_testData.txt')

	# ========================================================================= #
	# ================= STEP 2. PREPARE AND FORMATTING DATA =================== #
	# ========================================================================= #
	NUMBER_OF_FEATURES = len(inputData[0]) - 1
	NUMBER_OF_TRAINING_POINTS = len(inputData)

	x = inputData[:, range(0, NUMBER_OF_FEATURES)]
	y = inputData[:, NUMBER_OF_FEATURES]
	testX = testData[:, range(0, NUMBER_OF_FEATURES)]
	testY = testData[:, NUMBER_OF_FEATURES]

	# ========================================================================= #
	# ============== STEP 3. DECLARE PRIMITIVES BEFORE THE PARTY ============== #
	# ========================================================================= #
	minSquareError = np.inf
	targetAlpha = None
	alphas = np.logspace(-10, -2, 200)			
	alphas = [1]
	# ========================================================================= #
	# ==================== STEP 4. FIND THE BEST ALPHA ======================== #
	# ========================================================================= #
	for eachAlpha in alphas:
		clf = LogisticRegression(C=eachAlpha)								# LINEAR REGRESSION CONFIGURATION
		clf.fit(x, y)												# PERFORM FITTING 
		squareError = np.mean((clf.predict(x) - y)**2)		# CALCULATE SQUARE ERROR 
		if  (squareError <= minSquareError):						# OVERWRITE TARGET ALPHA
			minSquareError = squareError
			targetAlpha = eachAlpha

	# ========================================================================= #
	# ===== STEP 5. PERFORM FITTING WITH THE BEST ALPHA AND DO PREDICTION ===== #
	# ========================================================================= #
	clf = LogisticRegression(C=targetAlpha)
	clf.fit(x, y)

	predictedData = clf.predict(testX)
	squareError = (np.mean(predictedData - testY) ** 2)
	
	# ========================================================================= #
	# ======================== STEP 6. VISUALISATION ========================== #
	# ========================================================================= #
	print ("\n")
	print ("==============================================================")
	print (" LOGISTIC REGRESSION ")
	print ("==============================================================")
	print (" theta0: %.3g" % (clf.intercept_))
	print (" theta1~n: %s" % (clf.coef_))
	print (" Alpha: %0.3g" % (targetAlpha))
	print (" Square Error: %.3g" % (squareError))
	print ("==============================================================")
	print (" Predict results:")
	for eachPredictedData in predictedData:
		print (" %6g" % (eachPredictedData))
	print ("==============================================================")
	print ("\n")

if __name__ == '__main__':
	main()