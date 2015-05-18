from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np

def main():
	# ========================================================================= #
	# ================== STEP 1. LOAD EXTERNAL INPUT DATA ===================== #
	# ========================================================================= #
	validationData = np.loadtxt('validationSet.txt')

	# ========================================================================= #
	# ================= STEP 2. PREPARE AND FORMATTING DATA =================== #
	# ========================================================================= #
	NUMBER_OF_FEATURES = len(validationData[0]) - 1
	NUMBER_OF_TRAINING_POINTS = len(validationData)

	validationX = validationData[:, range(0, NUMBER_OF_FEATURES)]
	validationY = validationData[:, NUMBER_OF_FEATURES]
	
	# ========================================================================= #
	# ===================== STEP 5. PERFORM VALIDATION ======================== #
	# ========================================================================= #
	clf = joblib.load('learntModel.pkl')
	validationData = clf.predict(validationX)
	validationProb = clf.predict_proba(validationX)
	validationError = clf.score(validationX, validationY)
	
	# ========================================================================= #
	# ======================== STEP 6. VISUALISATION ========================== #
	# ========================================================================= #
	print ("\n")
	print ("==============================================================")
	print (" LOGISTIC REGRESSION - VALIDATION")
	print ("==============================================================")
	print (" theta0: %.3g" % (clf.intercept_))
	print (" theta1~n: %s" % (clf.coef_))
	print (" Square Error: %.3g" % (validationError))
	print ("==============================================================")
	print (" Predict results:")
	for i in range(0, len(validationData)):
		print (" %6g \t %6g" % (validationData[i], np.max(validationProb[i])))
	print ("==============================================================")
	print ("\n")

if __name__ == '__main__':
	main()