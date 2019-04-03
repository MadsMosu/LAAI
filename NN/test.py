import numpy as np
from matplotlib import pyplot

allTestLosses = np.load('./models/formula1/multi-hidden/training_errors.npy')


pyplot.figure()
testErrors = []
for error in allTestLosses:
    testErrors.append(error[-1])
pyplot.scatter([1,2,3,4,5,6,7,8,9,10], testErrors)
pyplot.title('Model errors')
pyplot.xlabel('Number of neurons')
pyplot.ylabel('Test error')
pyplot.xticks([1,2,3,4,5,6,7,8,9,10], [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
pyplot.savefig('./models/formula1/multi-hidden/training_errors.png')