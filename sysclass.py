#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:56:29 2018

@author: aidan
"""
from numpy import array, expand_dims, kron, eye, dot, append, ones, arange, outer, zeros
from time import time
from mpmath import exp
from scipy.linalg import expm
from MpsMpo_block_level_operations import mps_block, mpo_block
from MpsMpo_site_level_operations import mpo_site
from lineshapes import eta_all
import matplotlib.pyplot as plt

class temposys(object):
    def __init__(self,hilbert_dim):
        #initialise system object with hilbert space dimension = hilbert_dim
        self.dim=hilbert_dim
        
        self.ham=lambda t: eye(self.dim)    #the reduced system hamiltonian is now a function of time
        self.lindbds=[]                     #list of lindblads and rates: [[rate1,oper1],[rate2,oper2],...]
        self.intparam=[]                    #lists of interaction op eigenvalues and bath etas: [[eigs1,eta1],[eigs2,eta2],..]
        self.state=eye(self.dim)            #reduced system density matrix
        
        self.dkmax=0                        #maximum length of the mps
        self.dt=0                           #size of timestep
        self.prec=0                         #precision used in svds
        self.mod=0                          #maximum point to use new quapi coeffs up to
        self.point=0                        #current point in time system is at
        
        self.dat=[]                         #data list: [[time1,state1],[time2,state2],...]
        self.dfile='temp'
        self.mps=mps_block(0,0,0)           #blank mps block
        self.mpo=mpo_block(0,0,0)           #blank mpo block
        
    def getcoeffs(self):
        #goes through eta functions in intparam creating makri coeffs from them
        for el in self.intparam:
            #tb is list of values of eta evaulated at integer timestep
            tb=list(map(el[1],list(arange(0,(self.dkmax+2)*self.dt,self.dt)))) #tb is array of et evaluated at integer timesteps
            #etab takes finite differences on tb to create coeffs
            etab=[tb[1]]
            for jj in range(1,self.dkmax+1): etab.append(tb[jj+1]-2*tb[jj]+tb[jj-1])
            #if loops evaluates the eta's and coeffs at further timesteps if needed for new quapi
            if self.mod>self.dkmax:
                tb=append(tb,array(
                        map(el[1],list(arange((self.dkmax+2)*self.dt,(self.mod+3)*self.dt,self.dt)))
                        ))
                etab=append(etab,array(
                        map(lambda jj: tb.diff(jj)-tb.diff(self.dkmax-1),range(self.dkmax+1,self.mod+1))
                        ))
            #coeffs are appended to element in intparam: [[eigs1,eta1,coeffs1],[eigs2,eta2,coeffs2],...]   
            el.append(etab)
        print('coeffs saved')
        
    def itab(self,dk):
        #function to store the influence function I_dk as a table
        
        #initialise matrix
        iffac=ones((self.dim**2,self.dim**2))
        
        #loop through all baths multiplyin in the corresponding matrix
        for el in self.intparam:
            #picking out the correct eta coeff
            eta_dk=el[2][dk+(self.point-dk)*int((self.mod and self.point)>self.dkmax)]
            #creating arrays of differences and sums of operator eigenvalues
            difs,sums=array([]),array([])
            for s in el[0]: difs,sums=append(difs,s-array(el[0])),append(sums,s+array(el[0]))
            #creating I_dk matrix and multiplying into iffac (couldnt get exp() to work here so used numerical value....)
            iffac=iffac*2.7182818284590452353602874713527**(outer(eta_dk.real*difs-1j*eta_dk.imag*sums,-difs))
        
        if dk==0: return iffac.diagonal() #I_0 is a funtion of one varibale only so is converted to vector
        else: return iffac
    
    def sysprop(self,j):
        #function to create a system propagator over time self.dt/2
        
        #we are using the convention where vec(rho) is rows stacked on vertically
        #and A.rho.B -> Ax(B.T).vec(rho)
        #first create hamiltonian commutation superoperator
        liou=-1j*(kron(self.ham(j*self.dt),eye(self.dim)) - kron(eye(self.dim),self.ham(j*self.dt).conj()))
        #then loop through list of lindblads to create dissipators
        for li in self.lindbds:
            liou=liou+li[0]*(kron(li[1],li[1].conj())
                -0.5*(kron(dot(li[1].conj(),li[1]),eye(self.dim))+kron(eye(self.dim),dot(li[1].conj().T,li[1]).T)))
        #and we are propagating initial state from the RIGHT     
        return expm(liou*self.dt/2).T 
    
    def temposite(self,dk):
        #converts 2-leg I_dk table into a 4-leg tempo mpo_site object
        iffac=self.itab(dk)
        #if dk==1 then multiply in system propagator and I_0
        if dk==1: iffac=iffac*self.itab(0)*dot(self.sysprop(self.point-1),self.sysprop(self.point))
        
        #initialise 4-leg tensor that will become mpo_site and loop through assigning elements
        tab=zeros((self.dim**2,self.dim**2,self.dim**2,self.dim**2),dtype=complex)
        for i1 in range(self.dim**2):
            for a1 in range(self.dim**2):
                tab[i1][i1][a1][a1]=iffac[i1][a1]
        
        
        if dk>=self.dkmax or dk==self.point:
            #if at an end site then sum over east leg index and replace with 1d dummy leg
            return mpo_site(tens_in=expand_dims(dot(tab,ones(self.dim**2)),-1))
        else:        
            return mpo_site(tens_in=tab)
                     
    def prep(self):
        #prepares system to be propagated once params have been set
        self.dat.append([0,self.state])     #store initial state in data
        self.point=1                        #move to first point
        self.getcoeffs()                    #calculate the makri coeffs
        
        #propagte initial state half a timestep with sys prop and multiply in I_0 to get initial 1-leg ADT           
        self.mps.insert_site(0,expand_dims(expand_dims(
                dot(self.state.reshape(self.dim**2),self.sysprop(0))*self.itab(0)
                                    ,-1),-1))
        
        #append first site to mpo to give a length=1 block
        self.mpo.append_mposite(self.temposite(1))
    
    def getstate(self):
        #function to return the reduced system physical density matrix at time=self.point*self.dt
        return dot(self.mps.readout3(),self.sysprop(self.point-1)).reshape((self.dim,self.dim))
    
    def prop(self,ksteps=1):
        #propagates the system for ksteps - system must be prepped first
        for k in range(ksteps):
            
            t0=time()
            #find current time physical state and store this in self.dat
            self.state=self.getstate()
            self.dat.append([self.point*self.dt,self.state])
            #contract and grow the mps using the mpo
            self.mps.contract_with_mpo(self.mpo,prec=self.prec,trunc_mode='accuracy')
            self.mps.insert_site(0,expand_dims(eye(self.dim**2),1))
            #move the system forward a point
            self.point=self.point+1
            #replace dk=1 tempo site - only really necessary for time dependent hamiltonians
            self.mpo.data[0]=self.temposite(1)
            
            if self.point<self.dkmax+1:
                #this is the growth stage: mpo has end site updated and a new site appended
                self.mpo.data[-1]=self.temposite(self.point-1)
                self.mpo.append_mposite(self.temposite(self.point))
            elif (self.mod>self.dkmax) or (self.point==self.dkmax+1):
                #beyond the growth stage we either use new quapi and carry on updating end site
                #or update it just once at point=kmax+1 if using old quapi
                self.mpo.data[-1]=self.temposite(self.dkmax)
                self.mps.contract_end()       
            else:
                #if not using new quapi and point>kmax just contract away end site of mpo
                self.mps.contract_end()
            print("propagated " +str(k+1)+" of "+str(ksteps)+"  Time: "+str(time()-t0))
           
    def opdat(self,op):
        td=[[],[]]
        for da in self.dat:
            td[0].append(da[0])
            td[1].append(dot(op,da[1]).trace().real)
        return td    
       

########
########
#example code for how to run it

#defining operators        
sigz=array([[1,0],[0,-1]])
sigx=array([[0,1],[1,0]])
sigy=array([[0,-1j],[1j,0]])
sigp=0.5*(sigx+1j*sigy)
sigm=0.5*(sigx-1j*sigy)
idd=array([[1,0],[0,1]])  

#initialise a 2d tempo system and set params   
system=temposys(2)
#hamiltonina - a function of time, in this case constant
system.ham=lambda t: sigx
#initial state is excited
system.state=0.5*(idd+sigz)
#couples to sigma_z with single ohmic bath 
system.intparam.append([[1,-1],lambda t: eta_all(t,0.2,1.000001,7.5,0,0.5*0.01*10) ])
#timestep, memory length and truncation precision
system.dt=0.25
system.dkmax=10
system.prec=0.00001

#prep the system for propagation
system.prep()
#propagate the system 100 timesteps
system.prop(100)
   
#generate data for obseravble sigz and plot 
dat=system.opdat(sigz)
plt.plot(dat[0],dat[1])
plt.show()

     
        
        
        
        