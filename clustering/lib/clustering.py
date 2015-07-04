import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def train(trainingData, pklFile, clusteringAll, numberOfClusters=None):
	# ========================================================================= #
	# =============== STEP 1. DEFINE OUTPUT LEARNT MODEL FILE ================= #
	# ========================================================================= #
	if (pklFile == ''):
		os.system('rm -rf learntModel & mkdir learntModel')
		pklFile = 'learntModel/learntModel.pkl'
	
	# ========================================================================= #
	# =============== STEP 2. PERFORM CLUSTERING TO THE DATA ================== #
	# ========================================================================= #
	if (numberOfClusters == None):
		print "Running MeanShift Model..."
		bandwidth = estimate_bandwidth(trainingData)
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=clusteringAll)
		ms.fit(trainingData)
		joblib.dump(ms, pklFile)
		return {"numberOfClusters":len(ms.cluster_centers_), "labels": ms.labels_, "clusterCenters":ms.cluster_centers_}
	
	elif (numberOfClusters != None):
		print "Running K-Means Model..."
		kMeans = KMeans(init='k-means++', n_clusters=numberOfClusters)
		kMeans.fit(trainingData)
		joblib.dump(kMeans, pklFile)
		return {"numberOfClusters":len(kMeans.cluster_centers_), "labels": kMeans.labels_, "clusterCenters":kMeans.cluster_centers_}

def predict(testData, pklFile):
	if (pklFile == ''):
		pklFile = 'learntModel/learntModel.pkl'
	clustering = joblib.load(pklFile)
	predictedData = clustering.predict(testData)
	return predictedData
	
def visualise(inputData, clusteredResult, outputFile):
	import matplotlib.pyplot as plt
	from itertools import cycle
	from mpl_toolkits.mplot3d import Axes3D

	INPUT_DIMENSION = len(inputData[0])
	colors = cycle('rbgcmykrbgcmykrbgcmykrbgcmyk')

	figure = plt.figure()

	if (INPUT_DIMENSION == 2):
		for k, col in zip(range(clusteredResult['numberOfClusters']), colors):
			my_member = clusteredResult['labels'] == k
			plt.scatter(inputData[my_member,0], inputData[my_member,1], \
						c=col, marker='o', s=20, edgecolor='k')
			plt.scatter(clusteredResult['clusterCenters'][k,0], \
						clusteredResult['clusterCenters'][k,1], \
						c=col, marker='o', s=50, edgecolor='k')

	elif (INPUT_DIMENSION == 3):
		ax = figure.add_subplot(111, projection='3d')
		for k, col in zip(range(clusteredResult['numberOfClusters']), colors):
			my_member = clusteredResult['labels'] == k
			ax.scatter(inputData[my_member, 0], \
					   inputData[my_member, 1], \
					   inputData[my_member, 2], \
					   c=col, marker='o', s=20, edgecolor='k')
			ax.scatter(clusteredResult['clusterCenters'][k,0], \
					   clusteredResult['clusterCenters'][k,1], \
					   clusteredResult['clusterCenters'][k,2], \
					   c=col, marker='o', s=50, edgecolor='k')
	else:
		print ('The dataset cannot be visualised')

	#plt.savefig(outputFile)
	plt.show()

if __name__ == '__main__':
	print ("\n==============================================================")
	print ("\nHi Dear, You cannot run this script directly")
	print ("Please have a look at experiment1.py\n")
	print ("==============================================================\n")	