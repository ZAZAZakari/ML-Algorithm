from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib
import numpy as np
import os

def train(trainingData, pklFile):
	# ========================================================================= #
	# =============== STEP 1. DEFINE OUTPUT LEARNT MODEL FILE ================= #
	# ========================================================================= #
	if (pklFile == ''):
		os.system('rm -rf learntModel & mkdir learntModel')
		pklFile = 'learntModel/learntModel.pkl'
	
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
	joblib.dump(clf, pklFile)
	
	return {"intercept": clf.intercept_, "coef":clf.coef_, "alpha":clf.C_, "accuracy":clf.score(x,y)}

def validate(validationData, pklFile):
	# ========================================================================= #
	# =============== STEP 1. DEFIEN INPUT LEARNT MODEL FILE ================== #
	# ========================================================================= #
	if (pklFile == ''):
		pklFile = 'learntModel/learntModel.pkl'
		
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
	clf = joblib.load(pklFile)
	validationData = clf.predict(validationX)
	validationProb = clf.predict_proba(validationX)
	
	return {"intercept":clf.intercept_, "coef":clf.coef_, "accuracy":clf.score(validationX, validationY), \
		    "validationData":validationData, "validationProb":validationProb}

def predict(predictData, pklFile):
	# ========================================================================= #
	# =============== STEP 1. DEFIEN INPUT LEARNT MODEL FILE ================== #
	# ========================================================================= #
	if (pklFile == ''):
		pklFile = 'learntModel/learntModel.pkl'
		
	# ========================================================================= #
	# ================= STEP 2. PREPARE AND FORMATTING DATA =================== #
	# ========================================================================= #
	NUMBER_OF_FEATURES = len(predictData[0])
	NUMBER_OF_TRAINING_POINTS = len(predictData)

	predictX = predictData[:, range(0, NUMBER_OF_FEATURES)]
	
	# ========================================================================= #
	# ===== STEP 3. PERFORM FITTING WITH THE BEST ALPHA AND DO PREDICTION ===== #
	# ========================================================================= #
	clf = joblib.load(pklFile)
	
	predictedData = clf.predict(predictX)
	predictedProb = clf.predict_proba(predictX)
	
	return {"intercept":clf.intercept_, "coef":clf.coef_, "predictedData":predictedData, \
	        "predictedProb":predictedProb}
	
if __name__ == '__main__':
	print ("\n==============================================================")
	print ("\nHi Dear, You cannot run this script directly")
	print ("Please have a look at experiment1.py\n")
	print ("==============================================================\n")	
	
# ============================================================================= #
# END OF THE CODE 
# ============================================================================= #