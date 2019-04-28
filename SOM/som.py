from SomImpl import MiniSom
import time
import numpy as np
from pylab import text,show,cm,axis,figure,subplot,imshow,zeros
import matplotlib.pyplot as plt
import sys
import os
from sklearn import preprocessing

train_data = np.load('./data/train.npy')
train_labels = np.load('./data/labels.npy')

train_labels.reshape(-1,1)

som_grid_rows = 10
som_grid_columns = 10
epochs = 2
sigma = 0.5
lr = 0.2

train_data = train_data.reshape(-1,15*15)
scaler = preprocessing.MinMaxScaler()
train_data = scaler.fit_transform(train_data)

# train_data = train_data.astype('float32')
# train_data /= 255

print(train_data.shape)


experiments = [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000] #
allTimes = []
learning_curves = []
iterations = []

for num_iter in experiments:
	t0 = time.time()
	som = MiniSom(
			som_grid_rows, 
			som_grid_columns, 
			15*15, 
			sigma=sigma, 
			learning_rate=lr,
			neighborhood_function='gaussian')
	som.pca_weights_init(train_data)
	#som.random_weights_init(train_data)
	som.train_random(train_data.reshape(-1,15*15), num_iter, verbose=True)
	t1 = time.time() - t0
	#print(time)
	allTimes.append(time.time() - t0)


	q_error = []
	iter_x = []
	learning_curve = []
	for i in range(num_iter):
		percent = 100*(i+1)/num_iter
		rand_i = np.random.randint(len(train_data))
		som.update(train_data[rand_i], som.winner(train_data[rand_i]), i, num_iter)
		if (i) % 100 == 0:
			error = som.quantization_error(train_data)
			q_error.append(error)
			learning_curve.append(error)
			iter_x.append(i)
			sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')
		
	lcurve=np.vstack((iter_x,q_error)).T
	iterations.append(iter_x)
	learning_curves.append(learning_curve)
	#np.savetxt('./results/random/lcurve'+str(num_iter)+'.txt',lcurve)


	plt.figure(figsize=(8, 8))
	wmap = {}
	im = 0
	for x, t in zip(train_data, train_labels):  # scatterplot
		w = som.winner(x)
		wmap[w] = im
		plt. text(w[0]+.5,  w[1]+.5,  str(t),
				color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
		im = im + 1
	plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
	plt.savefig('results/euclidian/umatrix_'+str(num_iter)+'.png')


np.save('results/euclidian/q_errors', learning_curves)
np.save('results/euclidian/iterations', iterations)

plt.figure()
plt.bar([1,2,3,4,5,6,7,8,9,10], height=allTimes) #
plt.title('Computational times')
plt.xlabel('Number of iterations')
plt.ylabel('Time in seconds')
plt.xticks([1,2,3,4,5,6,7,8,9,10], experiments) #
plt.savefig('./results/euclidian/computation_times.png')


plt.figure(figsize=(12,8), dpi=100)
for lr, it in zip(learning_curves, iterations):
	plt.plot(it, lr, marker='.')
plt.title('Learning curves')
plt.ylabel('Quantization error')
plt.xlabel('Iteration')
plt.savefig('./results/euclidian/lcurves.png')





