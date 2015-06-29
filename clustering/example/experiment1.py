import numpy as np
import sys
import random

# ====================================================================== #
# =============== CONFIGURATIONS HERE PLEASE MODIFY ==================== #
# ====================================================================== #
LIBRARY_PATH = '/home/zazazakari/ML-Algorithm/clustering/lib/'
PROJECT_PATH = '/home/zazazakari/ML-Algorithm/clustering/example/'
PKL_PATH = '/home/zazazakari/ML-Algorithm/clustering/example/'

TRAINING_SET_FILE_NAME = PROJECT_PATH + 'trainingSet.txt'
PREDICTION_SET_FILE_NAME = PROJECT_PATH + 'testSet.txt'
LEARNT_MODEL_NAME = PKL_PATH + 'experiment1.pkl'

# ====== YOU NEED TO HAVE YOUR Xvfb RUNNING FOR VISUALISATION ======= #
VISUALISATION = False
VISUALISATION_FILE_NAME = PROJECT_PATH + 'experiment1.png'

# ====================================================================== #
# ====================================================================== #
sys.path.append(LIBRARY_PATH)
import clustering

def main():
	# ================================================== #
	# =============== STEP 1. TRAINING ================= #
	# ================================================== #
	#trainingData = np.random.rand(1000, 2)
	trainingData = np.loadtxt(TRAINING_SET_FILE_NAME)
	testData = np.loadtxt(PREDICTION_SET_FILE_NAME)
	
	# PARAMS: TRAINING_DATA, LEARNT MODEL, CLUSTER ALL?, NUMBER OF CLUSTERS 
	clusteredResult = clustering.train(trainingData, LEARNT_MODEL_NAME, True, 3)
	
	# ========== TAKE THE VALUE FROM clusteredResult AS YOU NEED ============ #
	# print (clusteredResult['numberOfClusters'])
	# print (clusteredResult['clusterCenters'])
	# print (clusteredResult['labels'])
	
	# PARAMS: TRAINING_DATA, LEARNT MODEL
	predictionResult = clustering.predict(testData, LEARNT_MODEL_NAME)
	
	#print predictionResult
	
	if (VISUALISATION):
		clustering.visualise(trainingData, clusteredResult, VISUALISATION_FILE_NAME)

	
if __name__ == '__main__':
	main()