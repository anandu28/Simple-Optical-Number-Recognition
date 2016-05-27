from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
NROWS = 28
NCOLS = 28
SUBSET_SIZE = 5000
EPOCHS_PER_SUBSET = 3
random.seed()

class Image:
    def __init__(self, image, label):
        self.label = label;
        self.image = image;

    def as1dArray(self):
        ret = []
        for row in range(0,NROWS):
            ret += self.image[row]
        return ret

def convertLabelToOutputArray(label):
    output = []
    for i in range(0,10):
        output.append(1 if i == label else 0)
    return output

def getNumericalResult(results):
    max = results[0]
    maxindex = 0
    for i in range(1, len(results)):
        if results[i] > max:
            max = results[i]
            maxindex = i
    return maxindex

def simplify(input):
    return float(input) / 255

def random_subset(start_set, size):
    subset = []
    for n in range(0, size):
        subset.append(start_set[random.randint(0, len(start_set) - 1)])
    return subset

labels = []
images = []
#load data
train_labels = open("data/train-labels.idx1-ubyte", 'rb')
train_images = open("data/train-images.idx3-ubyte", 'rb')

train_labels.read(8)
for i in range(0,60000):
    labels.append(ord(train_labels.read(1)))

train_images.read(16)
for i in range(0,60000):
    if i % 6000 == 0:
        print ceil((float(i) / 60000) * 100), "% loaded"
    imagedata = []
    for row in range(0,NROWS):
        row = []
        for col in range(0,NCOLS):
            row.append(ord(train_images.read(1)))
        imagedata.append(row)
    image = Image(imagedata, labels[i])
    images.append(image)

net = buildNetwork(NROWS * NCOLS, 10)
print "Loading test data"

test_labels = []
test_images = []

#load data
test_label_file = open("data/test-labels.idx1-ubyte", 'rb')
test_image_file = open("data/test-images.idx3-ubyte", 'rb')

test_label_file.read(8)
for i in range(0,10000):
    test_labels.append(ord(test_label_file.read(1)))

test_image_file.read(16)
for i in range(0,10000):
    if i % 1000 == 0:
        print ceil((float(i) / 10000) * 100), "% loaded"
    imagedata = []
    for row in range(0,NROWS):
        row = []
        for col in range(0,NCOLS):
            row.append(ord(test_image_file.read(1)))
        imagedata.append(row)
    image = Image(imagedata, test_labels[i])
    test_images.append(image)

trainer = BackpropTrainer(net, None)

maxSuccess = 0

while maxSuccess < 75:
    print "Training epochs"
    ds = SupervisedDataSet(NROWS * NCOLS, 10)
    for image in random_subset(images, SUBSET_SIZE):
        ds.addSample(map(simplify, image.as1dArray()), convertLabelToOutputArray(image.label))
    trainer.setData(ds)
    trainer.trainEpochs(EPOCHS_PER_SUBSET)
    #trainer.trainUntilConvergence()
    print "Beginning testing"
    hits = 0
    misses = 0

    for image in test_images:
        result = net.activate(map(simplify, image.as1dArray()))
        numResult = getNumericalResult(result)
        if numResult == int(image.label):
            hits += 1
        else:
            misses += 1

    print "Simulation complete"
    successRate = ceil((float(hits)/(hits + misses)) * 100)
    if successRate > maxSuccess:
        maxSuccess = successRate
    print "Success Rate: ",successRate , "%"

print "Sufficient accuracy achieved!"
for image in test_images:
    result = net.activate(map(simplify, image.as1dArray()))
    print "Target response: ", image.label
    numResult = getNumericalResult(result)
    print "Response gotten: ", numResult
    data = [map(simplify, row) for row in image.image]
    plt.imshow(data)
    plt.show()
