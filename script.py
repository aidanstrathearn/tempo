#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:26:32 2018

@author: aidan
"""

from temposys_class import temposys, numint
from numpy import array, exp
from mpmath import besselj,sinc,cos
import matplotlib.pyplot as plt

###############################################################
####### EXAMPLE 1 - Ohmic Spin-1/2 SBM (Fig 2) ################
###############################################################

#set the dimension of the problem and
#define spin-1/2 operators using numpy arrays
#all operator/matrix type objects that go into TEMPO should be numpy arrays
hilbert_dim=2
Sz=0.5*array([[1,0],[0,-1]])
Sx=0.5*array([[0,1],[1,0]])
Sy=0.5*array([[0,-1j],[1j,0]])
idd=0.5*array([[1,0],[0,1]])

#Define coupling a, temperature T, cutoff frequency wc, and driving v (this is \Omega in the paper)
a=0.7
T=0
wc=5
v=1

#Define the spectral density function
def j1(w):
    return 2*a*w*exp(-w/wc)

#Numerically integrate the correlation function C(t), Eq.14 in Methods, to get the lineshape
#Whithin temposys_class finite differences are taken on this lineshape to obtain the coefficients
#given by Eq. 13 in Methods
def eta(t):
    return numint(t,T,j1)

#set the three convergence parameters: discretization timestep, memory length, and svd precision
#values chosen here are an example and produce something close to the a=07 dynamics in Fig. 2 but
#not as converged
dt=0.25
dkmax=50
prec=50

#define the free system hamiltonian and initialise in spin up state
hamiltonian=v*Sx
in_state=idd+Sz

#initialse a tempo system with specified dimension
system=temposys(hilbert_dim)
#name the system, all output files labelled with this name
system.set_filename('ohmic_spinhalf')
#set the free system hamiltonian to be used and the initial state of the spin
system.set_hamiltonian(hamiltonian)
system.set_state(in_state)

#attach the bath to the system, with lineshape eta and coupled to Sz spin operator
system.add_bath([[Sz,eta]])
system.convergence_params(dt,dkmax,prec)

#prepare the system ready to propagater ---
#---- must always be the last thing you do before propagating!
system.prep()

#propagate and collect reduced state data over 125 timesteps
system.prop(125)

#output data for evolution of observable Sz and plot
dat=system.getopdat(Sz)
plt.plot(dat[0],dat[1])
plt.show()


#==============================================================================
# #define spin-1/2 operators using numpy arrays
# #all operator/matrix type objects that go into TEMPO should be numpy arrays
# Sz=0.5*array([[1,0],[0,-1]])
# Sx=0.5*array([[0,1],[1,0]])
# Sy=0.5*array([[0,-1j],[1j,0]])
# idd=0.5*array([[1,0],[0,1]])
# 
# a=0.5*0.4
# T=0.2
# wc=7.5
# v=2
# 
# def j1(w):
#     return a*w*exp(-w/wc)
# 
# def eta(t):
#     return numint(t,T,j1)
# 
# dt=0.25
# dkmax=10
# prec=50
# 
# hilbert_dim=2
# hamiltonian=v*Sx
# in_state=idd+Sz
# 
# system=temposys(hilbert_dim)
# system.set_filename('makri')
# system.set_hamiltonian(hamiltonian)
# system.set_state(in_state)
# 
# system.add_bath([[Sz,eta]])
# system.convergence_params(dt,dkmax,prec)
# 
# system.prep()
# system.prop(100)
# 
# dat=system.getopdat(Sz)
# plt.plot(dat[0],dat[1])
# plt.show()
#==============================================================================


#==============================================================================
# 
# a=0.5
# T=0.5
# wc=0.5
# D=1
# R=20
# def j2(w):
#     return 2*a*(w**D)/(wc**(D-1))*exp(-w/wc)*(1-[cos(R*w),besselj(0,R*w),sinc(R*w)][D-1])
# 
#==============================================================================