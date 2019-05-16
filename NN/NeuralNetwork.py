import keras
from keras.layers import Dense, Activation, BatchNormalization
from keras.models import load_model, save_model
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot
pyplot.rcParams.update({'figure.max_open_warning': 0})
import os
import os.path
import sys
from TimeHistory import TimeHistory
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import axes3d
import math

#load all data
f1_values = np.load('./formula1-data/values.npy')
f1_train = np.load('./formula1-data/traindata/training-data.npy')
f1_val = np.load('./formula1-data/validationdata/validation-data.npy')
f1_test = np.load('./formula1-data/testdata/test-data.npy')

f1_labels = np.load('./formula1-data/labels.npy')
f1_train_labels = np.load('./formula1-data/traindata/training-labels.npy')
f1_val_labels = np.load('./formula1-data/validationdata/validation-labels.npy')
f1_test_labels = np.load('./formula1-data/testdata/test-labels.npy')

f2_values = np.load('./formula2-data/values.npy')
f2_train = np.load('./formula2-data/traindata/training-data.npy')
f2_val = np.load('./formula2-data/validationdata/validation-data.npy')
f2_test = np.load('./formula2-data/testdata/test-data.npy')

f2_labels = np.load('./formula2-data/labels.npy')
f2_train_labels = np.load('./formula2-data/traindata/training-labels.npy')
f2_val_labels = np.load('./formula2-data/validationdata/validation-labels.npy')
f2_test_labels = np.load('./formula2-data/testdata/test-labels.npy')

f1_values = np.reshape(f1_values,(-1,1))
f1_train = np.reshape(f1_train,(-1,1))
f1_val = np.reshape(f1_val,(-1,1))
f1_test = np.reshape(f1_test,(-1,1))

f1_labels = np.reshape(f1_labels,(-1,1))
f1_train_labels = np.reshape(f1_train_labels,(-1,1))
f1_val_labels = np.reshape(f1_val_labels,(-1,1))
f1_test_labels = np.reshape(f1_test_labels,(-1,1))

f2_values = np.reshape(f2_values,(-1,2))
f2_train = np.reshape(f2_train,(-1,2))
f2_val = np.reshape(f2_val,(-1,2))

f2_labels = np.reshape(f2_labels,(-1,1))
f2_train_labels = np.reshape(f2_train_labels,(-1,1))
f2_val_labels = np.reshape(f2_val_labels,(-1,1))

def formula1(xList):
    labels = [] 
    for x in xList:
        formula = np.sin(2*np.pi*x)+np.sin(5*np.pi*x)
        labels.append(formula)
    return np.array(labels)

def formula2(xyList):
    labels = []
    for (x,y) in xyList:
        formula = np.exp(-(x**2+y**2)/0.1)
        labels.append(formula)
    return np.array(labels)

def save_summary(dir, model):
    stdout_summary = sys.stdout
    with open(dir, 'w') as f:
        sys.stdout = f
        model.summary()
    sys.stdout = stdout_summary


batch_size = 5
epochs = 1000
formula = '1'
model_type = 'multi-hidden'

experimental_neurons = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#experimental_neurons = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

allTimes = []
allTrainingLosses = []
allTrainingAccs = []
allTestLosses = []
allTestAcc = []

for n in experimental_neurons:
    model = Sequential()
    model.add(Dense(units=n, input_dim=1))
    model.add(Activation(activation='sigmoid'))
    model.add(BatchNormalization())

    model.add(Dense(units=n))
    model.add(Activation(activation='sigmoid'))
    model.add(BatchNormalization())

    model.add(Dense(units=1))
    model.add(Activation(activation='linear'))

    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9)#
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])

    #checkpoint = keras.callbacks.ModelCheckpoint(os.path.abspath('./models/formula'+formula+'/'+model_type+'/model_n'+str(n)+'.h5'), monitor='mean_squared_error', verbose=0, save_best_only=True, mode='auto')
    time_callback = TimeHistory()
    history = model.fit(f1_train, f1_train_labels,
        #batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(f1_val, f1_val_labels),
        callbacks=[time_callback],
        shuffle=True,
        #validation_split=0.25
    )
    time = time_callback.times

    #loadedModel = load_model(os.path.abspath('./models/formula'+formula+'/'+model_type+'/model_n'+str(n)+'.h5'))
    loss = model.evaluate(f1_test, f1_test_labels, verbose=1)
    print('Test loss:', loss[0])
    allTestLosses.append(loss[0])
    allTrainingLosses.append(history.history['loss'])
    allTimes.append(time)

    # save_model(model, './models/formula'+formula+'/'+model_type+'/model_n'+str(n)+'.h5')
    np.save('./models/formula'+formula+'/'+model_type+'/times', allTimes)
    np.save('./models/formula'+formula+'/'+model_type+'/training_errors', allTrainingLosses)
    np.save('./models/formula'+formula+'/'+model_type+'/test_errors', allTestLosses)
    np.save('./models/formula'+formula+'/'+model_type+'/test_accs', allTestAcc)

    #save model evaluation error and acc
    np.save('./models/formula'+formula+'/'+model_type+'/evaluation/model_n'+str(n)+'_error', loss)

        #save summary of model
    save_summary('./models/formula'+formula+'/'+model_type+'/summaries/model_n'+str(n)+'_summary.txt', model)

    #plot test prediction vs actual
    f1_values = np.sort(f1_values)
    # f2_labels = formula1(f2_values)
    predictions = model.predict(f1_values).reshape(-1,1)
    pyplot.figure()
    pyplot.plot(predictions)
    pyplot.plot(f1_labels)
    pyplot.title('Model prediction - n = '+str(n))
    pyplot.ylabel('y')
    pyplot.xlabel('x')
    pyplot.legend(['Prediction', 'Actual'], loc='upper-left')
    pyplot.savefig('./models/formula'+formula+'/'+model_type+'/model_n'+str(n)+'_predictionVsActual.png')


    #plot error
    pyplot.figure()
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model error - n = '+str(n))
    pyplot.ylabel('Error')
    pyplot.xlabel('Epoch')
    pyplot.legend(['train', 'validation'], loc='upper-left')
    pyplot.savefig('./models/formula'+formula+'/'+model_type+'/model_n'+str(n)+'_error.png')


#make graphs
pyplot.figure()
testErrors = []
for error in allTrainingLosses:
    testErrors.append(error[-1])
pyplot.scatter([1,2,3,4,5,6,7,8,9,10], testErrors)
pyplot.title('Model errors')
pyplot.xlabel('Number of neurons')
pyplot.ylabel('Training error')
pyplot.xticks([1,2,3,4,5,6,7,8,9,10], experimental_neurons)
pyplot.savefig('./models/formula'+formula+'/'+model_type+'/training_errors.png')



pyplot.figure()
pyplot.scatter([1,2,3,4,5,6,7,8,9,10], allTestLosses)
pyplot.title('Model errors')
pyplot.xlabel('Number of neurons')
pyplot.ylabel('Test error')
pyplot.xticks([1,2,3,4,5,6,7,8,9,10], experimental_neurons)
pyplot.savefig('./models/formula'+formula+'/'+model_type+'/test_errors.png')

pyplot.figure()
averages = []
for time in allTimes:
    average = np.average(time)
    averages.append(average)
pyplot.bar([1,2,3,4,5,6,7,8,9,10], height=averages)
pyplot.title('Computational times')
pyplot.xlabel('Number of neurons')
pyplot.ylabel('Average epoch computation time/seconds')
pyplot.xticks([1,2,3,4,5,6,7,8,9,10], experimental_neurons)
pyplot.savefig('./models/formula'+formula+'/'+model_type+'/computation_times.png')








