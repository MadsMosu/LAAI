import numpy as np
import matplotlib.pyplot as plt

iterations = np.load('results/random/iterations.npy')
lrs = np.load('results/random/q_errors.npy')

plt.figure()
for lr, it in zip(lrs, iterations):
	plt.plot(lr, it, marker='o')
plt.title('Learning curves')
plt.ylabel('Quantization error')
plt.xlabel('Iteration')
plt.savefig('./results/random/lcurves.png')