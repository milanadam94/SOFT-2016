# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:13:23 2017

@author: Jelena
"""
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold='nan')
import math
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import dilation
from skimage.measure import label, regionprops
from scipy.misc import imresize
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors

def proces(self,rhs):
    
    cifre = np.zeros([5050, 784], 'uint8')
    
    print "skida mnist"
    mnist = fetch_mldata('MNIST original')
    data   = mnist.data
    labels = mnist.target.astype('int')
    train_rank = 5000
    train_subset = np.random.choice(data.shape[0], train_rank)
    train_data = data[train_subset]
    train_labels = labels[train_subset]
    
    br0 = br1 = br2 = br3 = br4 = br5 = br6 = br7 = br8 = br9 = 0
    
    print "broji brojeve iz mnist-a"
    for i in range(1, 5001):
        if(train_labels[i-1] == 0):
            br0 = br0 + 1
        if(train_labels[i-1] == 1):
            br1 = br1 + 1
        if(train_labels[i-1] == 2):
            br2 = br2 + 1
        if(train_labels[i-1] == 3):
            br3 = br3 + 1
        if(train_labels[i-1] == 4):
            br4 = br4 + 1
        if(train_labels[i-1] == 5):
            br5 = br5 + 1
        if(train_labels[i-1] == 6):
            br6 = br6 + 1
        if(train_labels[i-1] == 7):
            br7 = br7 + 1
        if(train_labels[i-1] == 8):
            br8 = br8 + 1
        if(train_labels[i-1] == 9):
            br9 = br9 + 1
    
    #PROVERITI PETLJE KAKO UZIMAJU BR* i A*
    print "dodaje 0"
    #dodaje sve nule
    a1 = 1
    for i in range(1, 5001):
        #if a1 < 200: da li ograniciti na fiksno?
        if(train_labels[i-1] == 0):
            cifre[a1-1]=train_data[i-1]
            a1 = a1 + 1
    
    print "dodaje 1"
    a2 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 1):
            cifre[br0+a2]= train_data[i-1]
            a2 = a2 + 1
    
    print "dodaje 2"
    a3 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 2):
            cifre[br0+br1+a3]=train_data[i-1]
            a3 = a3 + 1
    
    print "dodaje 3"
    a4 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 3):
            cifre[br0+br1+br2+a4]=train_data[i-1]
            a4 = a4 + 1
    
    
    a5 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 4):
            cifre[br0+br1+br2+br3+a5]=train_data[i-1]
            a5 = a5 + 1
    
    a6 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 5):
            cifre[br0+br1+br2+br3+br4+a6]=train_data[i-1] 
            a6 = a6 + 1
    
    a7 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 6):
            cifre[br0+br1+br2+br3+br4+br5+a7]=train_data[i-1]
            a7 = a7 + 1
    
    a8 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 7):
            cifre[br0+br1+br2+br3+br4+br5+br6+a8]=train_data[i-1]
            a8 = a8 +1
    
    a9 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 8):
            cifre[br0+br1+br2+br3+br4+br5+br6+br7+a9]=train_data[i-1]
            a9 = a9 + 1
    
    a10 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 9):
            cifre[br0+br1+br2+br3+br4+br5+br6+br7+br8+a10]=train_data[i-1]
            a10 = a10 + 1
     
        
    for i in range (1, 5001):
        cifre[i] = cifre[i]*(-1)+255
    
    print "poceo znakove"
    #probati sa dva for-a
    
    for j in range(1, 11):
        slika = imread("slike/1/"+str(j)+".png")
        toGray = rgb2gray(slika) * 255
        negate = 1 - (toGray > 30)
        negate = dilation(negate)
        labeled = label(negate)
        regions = regionprops(labeled)
        for region in regions:
            bbox = region.bbox          
            img_crop =  toGray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            imgResize = imresize(img_crop, [28,28])
            imgReshape = imgResize.reshape(784)
            cifre[5000+j-1] = imgReshape
            
    for j in range(1, 11):
        slika = imread("slike/2/"+str(j)+".png")
        toGray = rgb2gray(slika)
        negate = 1 - (toGray > 30)
        negate = dilation(negate)
        labeled = label(negate)
        regions = regionprops(labeled)
        for region in regions:
            bbox = region.bbox          
            img_crop =  toGray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            imgResize = imresize(img_crop, [28,28])
            imgReshape = imgResize.reshape(784)
            cifre[5010+j-1] = imgReshape
            
    for j in range(1, 11):
        slika = imread("slike/3/"+str(j)+".png")
        toGray = rgb2gray(slika)
        negate = 1 - (toGray > 30)
        negate = dilation(negate)
        labeled = label(negate)
        regions = regionprops(labeled)
        for region in regions:
            bbox = region.bbox          
            img_crop =  toGray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            imgResize = imresize(img_crop, [28,28])
            imgReshape = imgResize.reshape(784)
            cifre[5020+j-1] = imgReshape
            
    for j in range(1, 11):
        slika = imread("slike/4/"+str(j)+".png")
        toGray = rgb2gray(slika)
        negate = 1 - (toGray > 30)
        negate = dilation(negate)
        labeled = label(negate)
        regions = regionprops(labeled)
        for region in regions:
            bbox = region.bbox          
            img_crop =  toGray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            imgResize = imresize(img_crop, [28,28])
            imgReshape = imgResize.reshape(784)
            cifre[5030+j-1] = imgReshape
    
    for j in range(1, 11):
        slika = imread("slike/5/"+str(j)+".png")
        toGray = rgb2gray(slika)
        negate = 1 - (toGray > 30)
        negate = dilation(negate)
        labeled = label(negate)
        regions = regionprops(labeled)
        for region in regions:
            bbox = region.bbox          
            img_crop =  toGray[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            imgResize = imresize(img_crop, [28,28])
            imgReshape = imgResize.reshape(784)
            cifre[5040+j-1] = imgReshape
    
    
    KNN = NearestNeighbors(1, 'auto').fit(cifre)
    
    print "slika za obradu"
    izrazSlika = imread(rhs)
    #izrazSlika = imread("problem.png")
    toGray = rgb2gray(izrazSlika)*255
    negate=1-(toGray>30)   
    negate=dilation(negate)
    #plt.imshow(negate,'gray')
    labeled=label(negate)
    regions=regionprops(labeled)
    upored=toGray.astype('uint8')
    
    
    list = {}
    regions.sort
    for region in regions:
        bbox=region.bbox  
        img_crop =  upored[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        test=imresize(img_crop,[28,28])
        test=test.reshape(1,784)
        rez=KNN.kneighbors(test, 5)
        list[bbox]=rez[1][0][0]
        
    def sortByKey():
        result=sorted(list.items(), key = lambda t : t[0][1])
        return result
    
    
    result=sortByKey()
    resultList=np.asarray(result)
    odnos=0
    end=0
    izraz=""
    
    #collections.OrderedDict(sorted(streetno.items()))
    for x, broj in resultList:
        if(odnos==0):
            odnos=x[2]-x[0]
        if((x[2]-x[0])-odnos<0 and broj<4990):
            izraz+="**"
        if(end!=0 and end<x[1]):
            izraz+=')'
            end=0
        if broj < br0:
            izraz+=str(0)
        elif broj < br0+br1:
            izraz+=str(1)
        elif broj < br0+br1+br2:
            izraz+=str(2)
        elif broj < br0+br1+br2+br3:
            izraz+=str(3)
        elif broj < br0+br1+br2+br3+br4:
            izraz+=str(4)
        elif broj < br0+br1+br2+br3+br4+br5:
            izraz+=str(5)
        elif broj < br0+br1+br2+br3+br4+br5+br6:
            izraz+=str(6)
        elif broj < br0+br1+br2+br3+br4+br5+br6+br7:
            izraz+=str(7)
        elif broj < br0+br1+br2+br3+br4+br5+br6+br7+br8:
            izraz+=str(8)
        elif broj < br0+br1+br2+br3+br4+br5+br6+br7+br8+br9:
            izraz+=str(9)
        elif broj < 5010:
            izraz+='/'
        elif broj < 5020:
            izraz+='math.sqrt('
            end=x[3]
            if(odnos==x[2]-x[0]):
                odnos=0
        elif broj < 5030:
            izraz+='-'
        elif broj < 5040:
            izraz+='+'
        elif broj < 5050:
            izraz+='*'
            
        
    if(end!=0):
        izraz+=')'        
    print izraz
    
    return eval(izraz)

    
