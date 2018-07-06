#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:26:32 2018

@author: aidan
"""

from temposys_class import temposys, numint
from numpy import array, exp,log, diag, zeros,sqrt,insert,dot
from mpmath import besselj,sinc,cos,loggamma
import matplotlib.pyplot as plt
from time import time

#####################################
######### Short Guide ###############
#####################################
#This code will solve for dynamics of the reduced system with full system dynamics governed by Eq.
#with the bath lineshape function eta(t) as defined in Eq.
#All operators must be in a basis where the operator O is diagonal
#
#For a reduced system with:
#--hilbert space dimension d
#--(diagonal) bath coupling operator O=A
#--initial state reduced density matrix p0
#
#the set-up is:
#system=temposys(d)
#system.set_hamiltonian(H_0)
#system.set_state(p0)
#system.add_bath([A,eta(t)])
#
#then set the convergence parameters
#system.convergence_params(Del,pp,K)
#with:
#--discretisation timestep Del
#--precision paramter, pp, for performings svds a la Eq.(2) such that lambda_c=lambda_{max}*10**(-pp/10)
#--memory cutoff length K
#
#Now all that is left is to prepare the system
#system.prep()
#and to propagate the ADT n timesteps
#system.prop(n)
#
#After propagation, o obtain for expectation of operator V in the form data=[times_list,data_list]
#data=system.getopdat(V)
#
#

####################################################
############ The Spin Boson Model ##################
####################################################
#This function sets up a tempo system for the spin-(S/2) SBM, Eq.(4),
#for a bath at temperature T and with spectral density Jw as defined in Eq.(5)
#and initial spin state rho
#it also returns spin operators since we might want to find their expectations
def spin_boson(S,Om,rho,T,Jw):   
    #first define spin operators  
    Sz=[S/2]
    while len(Sz)<S+1: Sz.append(Sz[-1]-1)
    Sz=diag(Sz)

    Sx=zeros((S+1,S+1))
    for jj in range(len(Sx)-1):
        Sx[jj][jj+1]=0.5*sqrt(0.5*S*(0.5*S+1)-(0.5*S-jj)*(0.5*S-jj-1))
    Sx=Sx+Sx.T

    #now set up the system
    #Hilbert space dimension=S+1
    system=temposys(S+1)
    system.set_hamiltonian(Om*Sx)
    system.set_state(rho)
    #attach the bath - using inbuilt numerical integrator to find eta(t), which is 
    #the correlation function in Eq.(14), C(t), integrated twice over time from 0 to t
    system.add_bath([Sz,system.num_eta(T,Jw)])     
    return system, Sz, Sx

#==============================================================================
# ###########################################################################################
# ######### Example 1 - Fig.2 in Makri & Makarov J. Chem. Phys 102, 4600, (1995) with dkmax=70
# ###########################################################################################
# ######### takes ~90secs to run on EliteBook with i7 Core and 16GB RAM
# ###########################################################################################
# 
# #First we set up the system for a spin-1/2 and set values taking account of
# #factors of 0.5 between pauli and spin operators
# s=1
# Om=2
# rho=array([[1,0],[0,0]])
# 
# #set the kondo parameter a, the cutoff frequency wc,the temperature T=1/beta
# #and define the ohmic spectral density Jw
# a=0.1
# wc=7.5
# T=0.2
# def Jw(w):
#     return 2*a*w*exp(-w/wc)
# 
# #now set timestep and kmax
# Del=0.25
# dkmax=10
# 
# #set up the spin boson model and get spin operators
# sbm,sz,sx=spin_boson(s,Om,rho,T,Jw)
# 
# #propagate for 100 steps using three different values of svd truncation precision
# #and plot operator expectations to check for convergence
# #can see convergence with pp=30 i.e. lambda_c=0.001*lambda_max
# t0=time()
# for pp in [20,30,40]:
#     sbm.convergence_params(Del,dkmax,pp)
#     sbm.prep()
#     sbm.prop(100)
#     datz=sbm.getopdat(2*sz)
#     datx=sbm.getopdat(sx)
#     plt.plot(datz[0],datz[1])
#     plt.plot(datx[0],datx[1])
# print('total time: '+str(time()-t0))
# plt.show()
#==============================================================================


###########################################################################################
######### Example 2 - Fig.3(a) in TEMPO paper for lowest 3 couplings
###########################################################################################
######### takes ~110secs to run on EliteBook with i7 Core and 16GB RAM
###########################################################################################

#Set up the spin size and initial up state
s=2
Om=1
rho=array([[1,0,0],[0,0,0],[0,0,0]])

#set cutoff frequency and temperature
wc=5
T=0

#now set timestep and kmax - the timestep is larger and the cutoff tau_c smaller than
#used in paper so less converged -- better convergence needed for scaling analysis
Del=5/20
K=10
pp=40

#propagate system for 120 steps at lowest three values of coupling and plot Sz data
t0=time()
for a in [0.15]:
    def j1(w):
        return 2*a*w*exp(-w/wc)
    sbm,sz,sx=spin_boson(s,Om,rho,T,j1)
    sbm.convergence_params(Del,K,pp)
    sbm.prep()
    sbm.prop(120)
    datz=sbm.getopdat(sz)
    plt.plot(datz[0],datz[1])
print('total time: '+str(time()-t0))
plt.show()



