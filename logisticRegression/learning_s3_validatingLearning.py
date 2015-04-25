import numpy as np
import matplotlib.pyplot as plt

minMax = np.loadtxt('minMax.txt')
inputData = np.loadtxt('data_inputData.txt')
learnt_parameters = np.loadtxt('data_trainedParameters.txt')

NUMBER_OF_FEATURES = len(inputData[0])-1

print NUMBER_OF_FEATURES
print learnt_parameters
#learnt_parameters[1] = (learnt_parameters[1] - minMax[0]) / (minMax[1] - minMax[0])

for i in range(0, NUMBER_OF_FEATURES):
	learnt_parameters[i+1] = (learnt_parameters[i+1] - minMax[i][0]) / (minMax[i][1] - minMax[i][0])
#learnt_parameters[2] = (learnt_parameters[2] - minMax[1][0]) / (minMax[1][1] - minMax[1][0])
'''
learnt_parameters[1] = learnt_parameters[1] * (minMax[0][1] - minMax[0][0]) + minMax[0][0]
learnt_parameters[2] = learnt_parameters[2] * (minMax[1][1] - minMax[1][0]) + minMax[1][0]
learnt_parameters[3] = learnt_parameters[3] * (minMax[1][1] - minMax[2][0]) + minMax[2][0]
'''
print learnt_parameters


x0 = np.ones((31,1))
x1 = range(-1, 30)
x2 = np.power(x1, 2)
x3 = np.power(x1, 3)
#x4 = np.power(x1, 4)

x0 = np.hstack((x0, np.array([x1]).transpose()))
x0 = np.hstack((x0, np.array([x2]).transpose()))
x0 = np.hstack((x0, np.array([x3]).transpose()))
#x0 = np.hstack((x0, np.array([x4]).transpose()))

x = x0[:,1]
y = np.inner(x0, learnt_parameters)

plt.plot(inputData[:,0], inputData[:,NUMBER_OF_FEATURES], 'o',x, y)
plt.savefig('bb.jpg')