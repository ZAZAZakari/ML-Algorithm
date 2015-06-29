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

	plt.figure(1)
	colors = cycle('rbgcmykrbgcmykrbgcmykrbgcmyk')
	
	for k, col in zip(range(clusteredResult['numberOfClusters']), colors):
		my_member = clusteredResult['labels'] == k
		
		plt.plot(inputData[my_member, 0], inputData[my_member, 1], col + '.')
		plt.plot(clusteredResult['clusterCenters'][k,0], clusteredResult['clusterCenters'][k,1], \
		         'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
	plt.savefig(outputFile)

if __name__ == '__main__':
	print ("\n==============================================================")
	print ("\nHi Dear, You cannot run this script directly")
	print ("Please have a look at experiment1.py\n")
	print ("==============================================================\n")	