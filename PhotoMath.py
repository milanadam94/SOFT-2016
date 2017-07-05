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
    
    cifre = np.zeros([5150, 784], 'uint8')
    handwritten = np.zeros([100, 784], 'uint8')
    
    mnist = fetch_mldata('MNIST original')
    data   = mnist.data
    labels = mnist.target.astype('int')
    train_rank = 5000
    train_subset = np.random.choice(data.shape[0], train_rank)
    train_data = data[train_subset]
    train_labels = labels[train_subset]
    
    br0 = br1 = br2 = br3 = br4 = br5 = br6 = br7 = br8 = br9 = 0
    
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
    
    
    for i in range(0, 10):
        for j in range(1, 11):
            slika = imread("slike/"+str(i)+"/"+str(j)+".png")
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
                handwritten[i*10+j-1] = imgReshape
    
    
    #dodaje sve nule iz mnist
    a1 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 0):
            cifre[a1-1]=train_data[i-1]*(-1)+255
            a1 = a1 + 1
    for i in range(1, 11):
        cifre[a1+i-2]=handwritten[i-1]
    
    
    a2 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 1):
            cifre[br0+a2+9]= train_data[i-1]*(-1)+255
            a2 = a2 + 1      
    for i in range(1, 11):
        cifre[br0+a2+10+i-2]=handwritten[10+i-1]
      
        
    a3 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 2):
            cifre[br0+br1+a3+19]=train_data[i-1]*(-1)+255
            a3 = a3 + 1
    for i in range(1, 11):
        cifre[br0+br1+a3+20+i-2]=handwritten[20+i-1]
       
           
    a4 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 3):
            cifre[br0+br1+br2+a4+29]=train_data[i-1]*(-1)+255
            a4 = a4 + 1
    for i in range(1, 11):
        cifre[br0+br1+br2+a4+30+i-2]=handwritten[30+i-1]
        
           
    a5 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 4):
            cifre[br0+br1+br2+br3+a5+39]=train_data[i-1]*(-1)+255
            a5 = a5 + 1       
    for i in range(1, 11):
        cifre[br0+br1+br2+br3+a5+30+i-2]=handwritten[40+i-1]
        
         
    a6 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 5):
            cifre[br0+br1+br2+br3+br4+a6+49]=train_data[i-1]*(-1)+255
            a6 = a6 + 1
    for i in range(1, 11):
        cifre[br0+br1+br2+br3+br4+a6+40+i-2]=handwritten[50+i-1]
    
    
    a7 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 6):
            cifre[br0+br1+br2+br3+br4+br5+a7+59]=train_data[i-1]*(-1)+255
            a7 = a7 + 1
    for i in range(1, 11):
        cifre[br0+br1+br2+br3+br4+br5+a7+50+i-2]=handwritten[60+i-1]
        
         
    a8 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 7):
            cifre[br0+br1+br2+br3+br4+br5+br6+a8+69]=train_data[i-1]*(-1)+255
            a8 = a8 +1
    for i in range(1, 11):
        cifre[br0+br1+br2+br3+br4+br5+br6+a8+60+i-2]=handwritten[70+i-1]
        
        
    a9 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 8):
            cifre[br0+br1+br2+br3+br4+br5+br6+br7+a9+69]=train_data[i-1]*(-1)+255
            a9 = a9 + 1
    for i in range(1, 11):
        cifre[br0+br1+br2+br3+br4+br5+br6+br7+a9+70+i-2]=handwritten[80+i-1]
        
        
    a10 = 1
    for i in range(1, 5001):
        if(train_labels[i-1] == 9):
            cifre[br0+br1+br2+br3+br4+br5+br6+br7+br8+a10+79]=train_data[i-1]*(-1)+255
            a10 = a10 + 1
    for i in range(1, 11):
        cifre[br0+br1+br2+br3+br4+br5+br6+br7+br8+a10+80+i-2]=handwritten[90+i-1]
        
    
    pom = 0
    for i in range(11, 15):
        for j in range(1, 11):
            slika = imread("slike/"+str(i)+"/"+str(j)+".png")
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
                cifre[5100+pom*10+j-1] = imgReshape
        pom = pom + 1
    
    
    KNN = NearestNeighbors(1, 'auto').fit(cifre)
    
    izrazSlika = imread(rhs)
    #izrazSlika = imread("problem.png")
    toGray = rgb2gray(izrazSlika)*255
    negate=1-(toGray>30)   
    negate=dilation(negate)
    #plt.imshow(negate,'gray')
    labeled=label(negate)
    regions=regionprops(labeled)
    cast=toGray.astype('uint8')
    
    
    list = {}
    regions.sort
    for region in regions:
        bbox=region.bbox  
        img_crop =  cast[bbox[0]:bbox[2],bbox[1]:bbox[3]]
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
        if((x[2]-x[0])-odnos<0 and broj<5090):
            izraz+="**"
        if(end!=0 and end<x[1]):
            izraz+=')'
            end=0
        if broj < br0+10:
            izraz+=str(0)
        elif broj < br0+br1+20:
            izraz+=str(1)
        elif broj < br0+br1+br2+30:
            izraz+=str(2)
        elif broj < br0+br1+br2+br3+40:
            izraz+=str(3)
        elif broj < br0+br1+br2+br3+br4+50:
            izraz+=str(4)
        elif broj < br0+br1+br2+br3+br4+br5+60:
            izraz+=str(5)
        elif broj < br0+br1+br2+br3+br4+br5+br6+70:
            izraz+=str(6)
        elif broj < br0+br1+br2+br3+br4+br5+br6+br7+80:
            izraz+=str(7)
        elif broj < br0+br1+br2+br3+br4+br5+br6+br7+br8+90:
            izraz+=str(8)
        elif broj < br0+br1+br2+br3+br4+br5+br6+br7+br8+br9+100:
            izraz+=str(9)
        elif broj < 5110:
            izraz+='/'
        elif broj < 5120:
            izraz+='math.sqrt('
            end=x[3]
            if(odnos==x[2]-x[0]):
                odnos=0
        elif broj < 5130:
            izraz+='-'
        elif broj < 5140:
            izraz+='+'
        elif broj < 5150:
            izraz+='*'
            
    
    if(end!=0):
        izraz+=')'        
    #print izraz
    #
    #print eval(izraz)
    
    return eval(izraz)

    
