# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:13:23 2017

@author: Adam
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import dilation
from skimage.measure import label, regionprops
from scipy import ndimage

from skimage.filters import threshold_adaptive
from scipy.misc import imresize
from sklearn.datasets import fetch_mldata

izrazi = np.zeros([10045, 784], 'uint8')

mnist = fetch_mldata('MNIST original')
data   = mnist.data / 255.0
labels = mnist.target.astype('int')
train_rank = 10000
train_subset = np.random.choice(data.shape[0], train_rank)
train_data = data[train_subset]
train_labels = labels[train_subset]

#for i in range(1, 10001):
#    slika = train_data[i].reshape(784)
#    izrazi[i] = slika
#    plt.imshow(train_data[i], 'gray')

for i in range (1,11):
    plus = imread("slike/"+"plus"+str(i)+".png")
    grayPlus = rgb2gray(plus) * 255
    binar = 1 - (grayPlus > 30)
    binar = dilation(binar)
    labeled = label(binar)
    regions = regionprops(labeled)
    for region in regions:
        bbox = region.bbox          
        img_crop =  grayPlus[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        #imgResize = imresize(img_crop, [28,28])
        #imgReshape = imgResize.reshape(784)
        #izrazi[9990+x-1]=z
        plt.imshow(img_crop, 'gray')
        
for i in range (1,11):
    minus = imread("slike/"+"minus"+str(i)+".png")
    grayMinus = rgb2gray(minus) * 255
    binar = 1 - (grayMinus > 30)
    binar = dilation(binar)
    labeled = label(binar)
    regions = regionprops(labeled)
    for region in regions:
        bbox = region.bbox          
        img_crop =  grayMinus[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        imgResize = imresize(img_crop, [28,28])
        #imgReshape = imgResize.reshape(784)
        #izrazi[9990+x-1]=z
        plt.imshow(img_crop, 'gray')
        
for i in range (1,11):
    puta = imread("slike/"+"puta"+str(i)+".png")
    grayPuta = rgb2gray(puta) * 255
    binar = 1 - (grayPuta > 30)
    binar = dilation(binar)
    labeled = label(binar)
    regions = regionprops(labeled)
    for region in regions:
        bbox = region.bbox          
        img_crop =  grayPuta[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        imgResize = imresize(img_crop, [28,28])
        #imgReshape = imgResize.reshape(784)
        #izrazi[9990+x-1]=z
        plt.imshow(img_crop, 'gray')
        
for i in range (1,11):
    deljenje = imread("slike/"+"deljenje"+str(i)+".png")
    grayDeljenje = rgb2gray(deljenje) * 255
    binar = 1 - (grayDeljenje > 30)
    binar = dilation(binar)
    labeled = label(binar)
    regions = regionprops(labeled)
    for region in regions:
        bbox = region.bbox          
        img_crop =  grayDeljenje[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        imgResize = imresize(img_crop, [28,28])
        #imgReshape = imgResize.reshape(784)
        #izrazi[9990+x-1]=z
        plt.imshow(img_crop, 'gray')
        
for i in range (1,11):
    koren = imread("slike/"+"koren"+str(i)+".png")
    grayKoren = rgb2gray(koren) * 255
    binar = 1 - (grayKoren > 30)
    binar = dilation(binar)
    labeled = label(binar)
    regions = regionprops(labeled)
    for region in regions:
        bbox = region.bbox          
        img_crop =  grayKoren[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        imgResize = imresize(img_crop, [28,28])
        #imgReshape = imgResize.reshape(784)
        #izrazi[9990+x-1]=z
        plt.imshow(img_crop, 'gray')

#img = imread('slika1.png')
#gray = rgb2gray(img)*255
#img_gr = gray > 0.5
#img_gray_tr = 1-threshold_adaptive(img_gr, block_size=75, offset=0.04)
#plt.imshow(img_gray_tr, 'gray')
#
#labeled = label(img_gray_tr)
#regions = regionprops(labeled)
#
#print('Broj regiona: {}'.format(len(regions)))
#plt.imshow(labeled ,'gray')
#
#objects = ndimage.find_objects(labeled)
##for i in objects:
#loc = objects[5]
#plt.imshow(img_gray_tr[loc], 'gray')
#print(loc[0].start, loc[0].stop)
