import math
import numpy as np
from sklearn.model_selection import train_test_split
import os

def generateFormula1Values():
    values = []
    x = -1
    while (x <= 1):
        x = round(x, 3)
        values.append(x)
        x += 0.002
    return np.array(values)

def generateFormula2Values():
    values =[]
    x = y = -1
    while (x <= 1 and y <= 1):
        x = round(x, 2)
        y = round(y, 2)
        values.append([x, y])
        x += 0.05
        y += 0.05  
    return np.array(values)

def formula1(xList):
    labels = [] 
    for x in xList:
        formula = math.sin(2*math.pi*x)+math.sin(5*math.pi*x)
        labels.append(formula)
    return np.array(labels)



def formula2(xyList):
    labels = []
    for (x,y) in xyList:
        formula = math.exp(-(x**2+y**2)/0.1)
        labels.append(formula)
    return np.array(labels)


#generate data values
formula1Values = generateFormula1Values()
formula2Values = generateFormula2Values()


#shuffle data
#np.random.shuffle(formula1Values)
#np.random.shuffle(formula2Values)


#generate labels
formula1Labels = formula1(formula1Values)
formula2Labels = formula2(formula2Values)

#generate 600 samples for training set
train1, rest1, train1labels, rest1labels = train_test_split(formula1Values, formula1Labels, test_size=0.3, shuffle=False)
test1, val1, test1labels, val1labels = train_test_split(formula1Values, formula1Labels, test_size=0.5, shuffle=False)

#generate (42,2) samples for training set
train2, rest2, train2labels, rest2labels = train_test_split(formula2Values, formula2Labels, test_size=0.3, shuffle=False)
test2, val2, test2labels, val2labels = train_test_split(formula2Values, formula2Labels, test_size=0.5, shuffle=False)

np.save('formula1-data/values', formula1Values)
np.save('formula1-data/labels', formula1Labels)

np.save('formula1-data/traindata/training-data',train1)
np.save('formula1-data/traindata/training-labels',train1labels)

np.save('formula1-data/validationdata/validation-data',val1)
np.save('formula1-data/validationdata/validation-labels',val1labels)

np.save('formula1-data/testdata/test-data',test1)
np.save('formula1-data/testdata/test-labels',test1labels)


np.save('formula2-data/values', formula2Values)
np.save('formula2-data/labels', formula2Labels)

np.save('formula2-data/traindata/training-data',train2)
np.save('formula2-data/traindata/training-labels',train2labels)

np.save('formula2-data/validationdata/validation-data',val2)
np.save('formula2-data/validationdata/validation-labels',val2labels)

np.save('formula2-data/testdata/test-data',test2)
np.save('formula2-data/testdata/test-labels',test2labels)






            