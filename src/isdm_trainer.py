# before proceeding further.
from __future__ import print_function
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import isdm
import time

data_set = read_data_sets('', False)
self_data = data_set.train
training_data = data_set.train
validation_data = data_set.validation
testing_data = data_set.test
isdm.set_dimensionality(784)

memory = isdm.NPIntegerSDM(100000)
print(self_data.next_batch(100)[0].shape)
start = time.time()
input_data = self_data.next_batch(50)
print(input_data[1])
number = [isdm.MCRVector.random_vector() for i in range(10)]
input_collector = dict()
for image, label in zip(input_data[0], input_data[1]):
    input_image = np.array(image.flatten()*15, dtype=np.int8)
    input_vector = isdm.MCRVector(input_image)
    try:
        current = input_collector[label]
    except KeyError:
        input_collector[label] = input_vector
    else:
        input_collector[label] = current+input_vector

bins = np.bincount(input_data[1])

for key in range(10):
    memory.write(input_collector[key])
    print('.', end='')

print("Time taken to store", time.time()-start)

fig = plt.figure(figsize=(2, 2))

test, test_labels = self_data.next_batch(100)

print(test_labels)
k = 0
for i, (image, label) in enumerate(zip(test, test_labels)):
    if label == k:
        k += 1
        input_image = isdm.MCRVector(np.array(image.flatten()*15, dtype=np.int8))
        out = memory.read(input_image)
        cleaned_image = out.dims/15
        ax = fig.add_subplot(2, 2, k)
        ax.imshow(cleaned_image.reshape(28, 28), cmap=cm.gray)
        if k == 4:
            break

plt.show(block=True)
