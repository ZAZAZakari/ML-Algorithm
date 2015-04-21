import numpy as np
import matplotlib.pyplot as plt
import time

PROBLEM_SIZE = 30
PROBLEM_DIMENSION = 2
NUM_CLUSTERS = 3

def squareDistance(xVector, uVector):
	diffVector = np.subtract(xVector, uVector)
	return np.inner(diffVector, diffVector)
	
def clusterAssignment(x, u, xColor):
	for i in range(0, len(x)):
		minCluster = {'value':10000000, 'index':-1}
		for j in range(0, len(u)):
			clusterDistance = squareDistance(x[i],u[j])
			if clusterDistance < minCluster['value']:
				minCluster['value'] = clusterDistance
				minCluster['index'] = j
		xColor[i] = minCluster['index']
	return xColor
	
def movingCentroid(x, u, xColor):
	for i in range(0, len(u)):
		cluster = {'count':0, 'centroid':np.zeros(PROBLEM_DIMENSION)}
		for j in range(0, len(x)):
			if xColor[j] == i:
				cluster['count'] += 1
				cluster['centroid'] = np.add(cluster['centroid'], x[j])
		
		if (cluster['count'] <> 0):
			u[i] = np.divide(cluster['centroid'], cluster['count'])
	return u

x = np.random.rand(PROBLEM_SIZE, PROBLEM_DIMENSION) * 100
xColor = np.zeros(PROBLEM_SIZE)
u = np.random.rand(NUM_CLUSTERS, PROBLEM_DIMENSION) * 100

plt.figure(1)
plt.scatter(x[:,0], x[:,1], c=xColor*0.2+0.2)
plt.scatter(u[:,0], u[:,1], color='yellow')
plt.ion()
plt.show()
	
for i in range(0, 50):	
	xColor = clusterAssignment(x, u, xColor)
	u = movingCentroid(x, u, xColor)
	plt.cla()
	plt.scatter(x[:,0], x[:,1], c=xColor*0.2+0.2)
	plt.scatter(u[:,0], u[:,1], color='yellow')
	plt.draw()
	time.sleep(0.5)