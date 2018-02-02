""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""

import random 
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
import numpy as np
import pylab as plt
from scipy import misc
import re
import collections
import os
from tensorflow.python.client import device_lib
import resnet50
import signal
import sys
from os import listdir
from os.path import isfile, join
import cv2
import pickle
import time
import h5py
from keras.layers import Dense, Activation, Flatten, Merge
from keras.models import Model, Sequential


os.environ["CUDA_VISIBLE_DEVICES"]="1"
print device_lib.list_local_devices()

def writeListToTxt(list, name):
    f = open(name,'w') 
    for l in list:
        f.write(l+"\n")
    f.close


def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def one_hot(n, i):
    a = np.zeros(n)
    a[i] = float(1)
    return np.array(a)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([1-10]+)', key) ] 
    return sorted(l, key = alphanum_key)

datasetRoot = "/home/rovit01/Escritorio/sequences_cat3/Test/img/"
#datasetRoot = "/home/rovit01/Escritorio/sequences_cat3/test_cat3/cat3/"
#datasetRoot = "/home/rovit01/Escritorio/sequences_cat3/s5/cat3_test/"

def readDataset(path):
    f = open(path, 'r')
    X = []
    Y = []
    for line in f:
        try:
            toks = line.split()[0].split(";")
            #pathToImg = datasetRoot + toks[0]
            pathToImg = toks[0]
            #print pathToImg
            img = np.reshape(cv2.resize(misc.imread(pathToImg), (224, 224)), (224,224,3))
            X.append(img)
            Y.append(one_hot(10, int(toks[1])-1))
        except:
            None

    return np.array(X), np.array(Y)

testMan = "/home/rovit01/Escritorio/sequences_cat3/Test/test.txt"
#testMan = "/home/rovit01/Escritorio/sequences_cat3/test_cat3/test_cat3.txt"
#testMan = "/home/rovit01/Escritorio/sequences_cat3/s5/cat3_s5_test.txt"

test_x, test_Y = readDataset(testMan)
#print test_x.shape, test_Y.shape

model = resnet50.ResNet50()
opt = SGD()
model.compile(loss= 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.load_weights("/home/rovit01/Escritorio/sequences_cat3/resnet50/snaps/ex7_ep_73.hdf5")
#model.summary()
#print len(test_x)
batch_size = 1
loss = model.evaluate(np.array(test_x), np.array(test_Y), batch_size=batch_size, verbose=1)
print "Test Loss ", loss

aciertos = 0
for ex,Y in zip(test_x,test_Y):
    out = model.predict([np.array(ex).reshape(1,224,224,3)])

    print "Predicted:",out[0].argmax(),"GT:",Y.argmax()

    if out[0].argmax() == Y.argmax():
        aciertos = aciertos +1

print "Test Hits ", float(aciertos), "/",float(len(test_Y))