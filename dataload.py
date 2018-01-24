#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:27:41 2017

@author: aidan
"""
import pickle
import matplotlib.pyplot as plt

def datload(filename):
    f=open(filename, "rb")
    dlist=pickle.load(f,encoding='bytes')

    while 1:
        try:
            dlist.append(pickle.load(f))
        except (EOFError):
            break
    
    f.close()
    return dlist

    

d1=datload("checkcoup10_dkm50_prec60.pickle")

xlis=[]
tlist=[]

for jj in range(len(d1)):
    xlis.append(d1[jj][0])
    tlist.append(d1[jj][1][0]-d1[jj][1][3])

plt.plot(xlis,tlist)
plt.show()