import numpy as np
import tensorflow as tf
import cv2

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

def resize(mnist):
     train_data = []
     for img in mnist:
            resized_img = cv2.resize(img, (15, 15))
            train_data.append(resized_img)
     return train_data

data = np.array(resize(mnist.train._images))
#data = np.array(resize(mnist.test._images))

print(data)
print(data.shape)

np.save('./data/data', data)
#np.save('./data/test_data', test_data)


