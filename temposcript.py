#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:26:32 2018

@author: aidan
"""

from sysclass import temposys
from numpy import array,exp
from lineshapes import eta_all, numint
import matplotlib.pyplot as plt
import pickle

def datload(filename):
    f=open(filename, "rb")
    dlist=[[],[]]

    while 1:
        try:
            dlist[0].append(pickle.load(f,encoding='bytes'))
        except (EOFError):
            break
    
    f.close()
    return dlist

sigz=array([[1,0],[0,-1]])
sigx=array([[0,1],[1,0]])
sigy=array([[0,-1j],[1j,0]])
sigp=0.5*(sigx+1j*sigy)
sigm=0.5*(sigx-1j*sigy)
idd=array([[1,0],[0,1]])

dt=0.125
dkmax=10
prec=50



eps=0
v=1
hamil=eps*sigz+v*sigx


system=temposys(2)
system.set_filename('example')
system.set_hamiltonian(lambda t: hamil)
system.add_dissipation([[0,sigp],[0,sigm]])
system.set_state(0.5*(idd+sigz))

system.convergence_params(dt,dkmax,prec)

def eta(t):
    return eta_all(t,0.2,1.00001,7.5,0,0.5*0.1)

system.add_bath([[sigz,eta]])



system.prep()
system.prop(100)

system.convergence_check([5,10,20],[30,40,50],200)
system.convergence_checkplot(sigz)

#dat=system.getopdat(sigz)
#plt.plot(dat[0],dat[1])
#plt.show()

#system.plotopdat(sigz)

'''
def specd(w):
    return 0.01*w**3*exp(-(w/2)**2)


eta=lambda t: numint(t,0.523681,specd,ct=1)
system=temposys(2)
system.set_filename('prestest')
#system.add_operator([[optime,sigm],[optime+10,sigm]])
system.convergence_params(dt,dkmax,prec)
system.set_state(0.5*(idd+sigz))
system.set_hamiltonian(lambda t: hamil)
system.add_dissipation([[0.1,sigp],[0.15,sigm]])
system.add_bath([[0.5*(idd+sigz),eta]])
system.mod=0
system.ttcorr_convergence_check(sigm,sigp,1000,[15,30,60],[30,40,50],4000)
system.ttcorr_convergence_plot(0.5*(idd+sigz))
'''