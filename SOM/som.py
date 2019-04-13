from SomImpl import MiniSom
import time
import numpy as np
from pylab import text,show,cm,axis,figure,subplot,imshow,zeros
import matplotlib.pyplot as plt
import sys
import os

train_data = np.load('./data/train.npy')
train_labels = np.load('./data/labels.npy')

train_labels.reshape(-1,1)

som_grid_rows = 10
som_grid_columns = 10
epochs = 2
sigma = 0.5
lr = 0.2

train_data = train_data.astype('float32')
train_data /= 255

train_data = train_data.reshape(-1,15*15)


experiments = [10,20,30,40,50,60,70,80,90,100] #
allTimes = []
learning_curves = []

for num_iter in experiments:
	t0 = time.time()
	som = MiniSom(
			som_grid_rows, 
			som_grid_columns, 
			15*15, 
			sigma=sigma, 
			learning_rate=lr)
	#som.pca_weights_init(data)
	som.random_weights_init(train_data)
	som.train_random(train_data.reshape(-1,15*15), num_iter*train_data.shape[0], verbose=True)
	t1 = time.time() - t0
	print(time)
	allTimes.append(time.time() - t0)
	#print('Time elapsed: %.2fs' % (time))


	q_error = []
	iter_x = []
	for i in range(num_iter):
		percent = 100*(i+1)/num_iter
		rand_i = np.random.randint(len(train_data))
		som.update(train_data[rand_i], som.winner(train_data[rand_i]), i, num_iter)
		if (i+1) % 100 == 0:
			error = som.quantization_error(train_data)
			q_error.append(error)
			learning_curves.append(error)
			iter_x.append(i)
			sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')
		
	lcurve=np.vstack((iter_x,q_error)).T
	learning_curves.append(lcurve)
	np.savetxt('./results/random/lcurve'+str(num_iter)+'.txt',lcurve)

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
	plt.savefig('results/random/umatrix_'+str(num_iter)+'.png')


	# plt.figure(figsize=(10, 10), facecolor='white')
	# cnt = 0
	# for j in reversed(range(20)):  # images mosaic
	# 	for i in range(20):
	# 		plt.subplot(20, 20, cnt+1, frameon=False,  xticks=[],  yticks=[])
	# 		if (i, j) in wmap:
	# 			plt.imshow([wmap[(i, j)]],
	# 					cmap='Greys', interpolation='nearest')
	# 		else:
	# 			plt.imshow(np.zeros((8, 8)),  cmap='Greys')
	# 		cnt = cnt + 1
	# plt.tight_layout()
	# plt.savefig('resulting_images/som_digts_mosaic.png')
	# plt.show()

plt.figure()
plt.plot(learning_curves)
plt.title('Learning curves')
plt.ylabel('Quantization error')
plt.xlabel('Iteration')
plt.savefig('./results/random/lcurves.png')


plt.figure()
plt.bar([1,2,3,4,5,6,7,8,9,10], height=allTimes)
plt.title('Computational times')
plt.xlabel('Number of iterations')
plt.ylabel('Time in seconds')
plt.xticks([1,2,3,4,5,6,7,8,9,10], experiments)
plt.savefig('./results/random/computation_times.png')




