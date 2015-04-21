import numpy as np

def main():
	# =================== LOAD INPUT DATA FROM EXTERNAL TEXT FILE ==================== #
	inputData = np.loadtxt('data_inputData.txt')

	# ================== OBTAINING INPUT DATA DIMENSIONS =================== #
	NUMBER_OF_DATA = len(inputData)
	NUMBER_OF_FEATURES = len(inputData[0])-1

	# ================== DECLARING AN EMPTY NORMALIZED INPUT DATA =================== #
	normalizedInputData = np.ndarray(shape = (NUMBER_OF_DATA, 0))
	minMax = []

	# ================== LOOP THROUGH ALL THE FEATURES ================== #
	for i in range(0, NUMBER_OF_FEATURES):
		featureList = inputData[:,i]

		# ============== FOR EACH FEATURE, OBTAIN MIN AND MAX ============= #
		minV = min(featureList)
		maxV = max(featureList)

		minMax.append([minV, maxV])
		# ============== PERFORM NORMALIZATION ON THE FEATURE ============== #
		featureList = (featureList - minV) / (maxV - minV)
		
		# ============== APPEND VERTICALLY THE NORMALIZED FEATURE ================ #
		normalizedInputData = np.hstack((normalizedInputData, np.array([featureList]).transpose()))
	# ================= SAVE THE NORMALIZED DATA TO ANOTHER FILE ================= #
	normalizedInputData = np.hstack((normalizedInputData, np.array([inputData[:, NUMBER_OF_FEATURES]]).transpose()))
	np.savetxt('minMax.txt', minMax)
	np.savetxt('data_normalizedInputData.txt', normalizedInputData)

if __name__ == '__main__':
	main()