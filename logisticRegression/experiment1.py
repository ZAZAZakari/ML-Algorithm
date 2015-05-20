import numpy as np
import logisticRegression

def main():
	# ================================================== #
	# =============== STEP 1. TRAINING ================= #
	# ================================================== #
	trainingResult = logisticRegression.train(np.loadtxt('trainingSet.txt'), '')
	
	print ("\n")
	print ("==============================================================")
	print (" LOGISTIC REGRESSION - LEARNING ")
	print ("==============================================================")
	print (" theta0: %.3g" % trainingResult["intercept"])
	print (" theta1~n: %s" % trainingResult["coef"])
	print (" Alpha: %0.3g" % trainingResult["alpha"])
	print (" Accuracy: %.3g" % trainingResult["accuracy"])
	print ("==============================================================")
	print ("\n")
	
	# ================================================== #
	# =============== STEP 2. VALIDATE ================= #
	# ================================================== #
	validateResult = logisticRegression.validate(np.loadtxt('validationSet.txt'), '')
	
	print ("\n")
	print ("==============================================================")
	print (" LOGISTIC REGRESSION - VALIDATION")
	print ("==============================================================")
	print (" theta0: %.3g" % (validateResult["intercept"]))
	print (" theta1~n: %s" % (validateResult["coef"]))
	print (" Accuracy: %.3g" % (validateResult["accuracy"]))
	print ("==============================================================")
	print (" Predict results:")
	for i in range(0, len(validateResult["validationData"])):
		print (" %6g \t %6g" % (validateResult["validationData"][i], \
		                        np.max(validateResult["validationProb"][i])))
	print ("==============================================================")
	print ("\n")
	
	# ================================================== #
	# =============== STEP 3. PREDICT ================== #
	# ================================================== #
	predictResult = logisticRegression.predict(np.loadtxt('testSet.txt'), '')
	print ("\n")
	print ("==============================================================")
	print (" LOGISTIC REGRESSION - PREDICTION")
	print ("==============================================================")
	print (" theta0: %.3g" % (predictResult["intercept"]))
	print (" theta1~n: %s" % (predictResult["coef"]))
	print ("==============================================================")
	print (" Predict results:")
	for i in range(0, len(predictResult["predictedData"])):
		print (" %6g \t %6g" % (predictResult["predictedData"][i], \
		                        np.max(predictResult["predictedProb"][i])))
	print ("==============================================================")
	print ("\n")
	
if __name__ == '__main__':
	main()