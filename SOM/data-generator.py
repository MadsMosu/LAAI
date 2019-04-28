import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.image import resize_images, ResizeMethod
from keras.datasets import mnist
import cv2

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#train_images = mnist.train.images
#train_labels = mnist.train.labels

train = train_images[:1000]
train_labels = train_labels[:1000]

print(train.shape)

def resize(mnist):
     train_data = []
     for img in mnist:
          new_img = cv2.resize(img, dsize=(15, 15),interpolation=cv2.INTER_CUBIC)
          train_data.append(new_img)
     return train_data

train = np.array(resize(train))
#train = train.resize(-1,15*15)
print(train.shape)



np.save('./data/train', train)
np.save('./data/labels', train_labels)


