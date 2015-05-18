from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np

def main():
	# ========================================================================= #
	# ================== STEP 1. LOAD EXTERNAL INPUT DATA ===================== #
	# ========================================================================= #
	predictData = np.loadtxt('testSet.txt')

	# ========================================================================= #
	# ================= STEP 2. PREPARE AND FORMATTING DATA =================== #
	# ========================================================================= #
	NUMBER_OF_FEATURES = len(predictData[0])
	NUMBER_OF_TRAINING_POINTS = len(predictData)

	predictX = predictData[:, range(0, NUMBER_OF_FEATURES)]
	
	# ========================================================================= #
	# ===== STEP 3. PERFORM FITTING WITH THE BEST ALPHA AND DO PREDICTION ===== #
	# ========================================================================= #
	clf = joblib.load('learntModel.pkl')
	
	predictedData = clf.predict(predictX)
	predictedProb = clf.predict_proba(predictX)
	# ========================================================================= #
	# ======================== STEP 6. VISUALISATION ========================== #
	# ========================================================================= #
	print ("\n")
	print ("==============================================================")
	print (" LOGISTIC REGRESSION - PREDICTION")
	print ("==============================================================")
	print (" theta0: %.3g" % (clf.intercept_))
	print (" theta1~n: %s" % (clf.coef_))
	print ("==============================================================")
	print (" Predict results:")
	for i in range(0, len(predictedData)):
		print (" %6g \t %6g" % (predictedData[i], np.max(predictedProb[i])))
	print ("==============================================================")
	print ("\n")

if __name__ == '__main__':
	main()