from minisom import MiniSom
import time
import numpy as np
from pylab import text,show,cm,axis,figure,subplot,imshow,zeros
import matplotlib.pyplot as plt

data = np.load('./data/data.npy')
#test_data = np.load('./data/test_data.npy')

som_grid_rows = 10
som_grid_columns = 10
epochs = 1500
sigma = 0.5
lr = 0.01

allTimes = []
distanceTimes = []

t0 = time()
som = MiniSom(
        som_grid_rows, 
        som_grid_columns, 
        15*15, 
        sigma=sigma, 
        learning_rate=lr)
som.train_random(data.reshape(-1,15*15), data.shape[0]*epochs)
time = time() - t0
allTimes.append(time)
print('Time elapsed: %.2fs' % (time))

#compute nearest training sample for each SOM unit
t0 = time()
wmap = {}
qerrors = np.empty((som_grid_rows,som_grid_columns))
qerrors.fill(np.nan)
for im,x in enumerate(X.reshape(-1,15*15)):
    (i,j) = som.winner(x)
    qe = np.linalg.norm(x-som.weights[i,j])
    if np.isnan(qerrors[i,j]) or qe<qerrors[i,j]:
        wmap[(i,j)] = im
        qerrors[i,j] = qe
time = time() - t0
distanceTimes.append(time)
print('Time elapsed: %.2fs' % (time()-t0))


figure(1)
for j in range(som_grid_columns): # images mosaic
	for i in range(som_grid_rows):
		if (i,j) in wmap:
			text(i+.5, j+.5, str(y[wmap[(i,j)]]), 
                 color=cm.Dark2(y[wmap[(i,j)]]/9.), 
                 fontdict={'weight': 'bold', 'size': 11})
ax = axis([0,som.weights.shape[0],0,som.weights.shape[1]])

figure(facecolor='white')
cnt = 0
for j in reversed(range(som_grid_columns)):
	for i in range(som_grid_rows):
		subplot(som_grid_columns,som_grid_rows,cnt+1,frameon=False, xticks=[], yticks=[])
		if (i,j) in wmap:
			imshow(X[wmap[(i,j)]])
		else:
			imshow(zeros((15,15)))
		cnt = cnt + 1

figure(facecolor='white')
cnt = 0
for j in reversed(range(som_grid_columns)):
	for i in range(som_grid_rows):
		subplot(som_grid_columns,som_grid_rows,cnt+1,frameon=False, xticks=[], yticks=[])
		imshow(som.weights[i,j].reshape(15,15))
		cnt = cnt + 1





