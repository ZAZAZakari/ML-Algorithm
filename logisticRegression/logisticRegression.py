from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib

import numpy as np

def main():
	# ========================================================================= #
	# ================== STEP 1. LOAD EXTERNAL INPUT DATA ===================== #
	# ========================================================================= #
	trainingData = np.loadtxt('trainingSet.txt')
	
	# ========================================================================= #
	# ================= STEP 2. PREPARE AND FORMATTING DATA =================== #
	# ========================================================================= #
	NUMBER_OF_FEATURES = len(trainingData[0]) - 1
	NUMBER_OF_TRAINING_POINTS = len(trainingData)

	x = trainingData[:, range(0, NUMBER_OF_FEATURES)]
	y = trainingData[:, NUMBER_OF_FEATURES]
	
	# ========================================================================= #
	# ============== STEP 3. DECLARE PRIMITIVES BEFORE THE PARTY ============== #
	# ========================================================================= #
	minSquareError = np.inf
	targetAlpha = None
	alphas = np.logspace(-10, -2, 500)			
	
	# ========================================================================= #
	# ===== STEP 4. PERFORM FITTING WITH THE BEST ALPHA AND SAVE THE MODEL ==== #
	# ========================================================================= #
	clf = LogisticRegressionCV(Cs=alphas)
	clf.fit(x, y)
	squareError = clf.score(x,y)						# CALCULATE SQUARE ERROR 
	joblib.dump(clf, 'learntModel.pkl')
	
	# ========================================================================= #
	# ======================== STEP 6. VISUALISATION ========================== #
	# ========================================================================= #
	print ("\n")
	print ("==============================================================")
	print (" LOGISTIC REGRESSION - LEARNING ")
	print ("==============================================================")
	print (" theta0: %.3g" % (clf.intercept_))
	print (" theta1~n: %s" % (clf.coef_))
	print (" Alpha: %0.3g" % (clf.C_))
	print (" Accuracy: %.3g" % (squareError))
	print ("==============================================================")
	print ("\n")

if __name__ == '__main__':
	main()