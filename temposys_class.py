#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:15:33 2018
@author: aidan
"""
from numpy import cos,sin,inf,ascontiguousarray,unique,array, expand_dims, kron, eye, dot, ones, outer, zeros, shape, dtype, void
from mpmath import coth
from scipy.integrate import quad
from pickle import dump
from time import time
from scipy.linalg import expm
from mpsmpo_class import mps_block, mpo_block, mpo_site

class temposys(object):
    def __init__(self,hilbert_dim=2):
        #initialise system object with hilbert space dimension d=hilbert_dim
        self.dim=hilbert_dim
        
        #initialise the Hamiltonian commutator superoperator
        #acts on d^2 size vector so has dimension (d^2,d^2)
        self.ham=zeros((self.dim**2,self.dim**2)) 
        
        #intialise list that will contain all info about baths and couplings
        #an element will contain:
        #--commutator/anticommutator superoperators of system operator coupled to bath
        #--the lineshape function, eta(t), of the bath
        #--the makri coefficients eta_{kk'}, given by second order finite differences on eta(t)       
        self.intparam=[]                    
        
        #seld.deg will contain degeneracy info of tempo site tensors:
        #for vertical and horizontal legs of tensor separately:
        #--the number of unique elements
        #--the positions in the multidimensional array of a single occurance of each unique element
        #--an array the same size as the full tensor, whose elements are the positions in the list 
        #of unique elements so that the full multidimensional array can be reconstructed
        self.deg=[]
        
        #keeping track of how many bath we attach to system - any degeneracy has to be common to all baths
        self.nbaths=0
        
        #instantaneous reduced system density matrix - vectorised
        self.state=array(self.dim**2)
        
        #initial reduced system density matrix - stored separate from instantaneous state
        self.istate=array(self.dim**2)   
        
        #memory length/maximum length of the mps
        self.dkmax=0
        
        #sets the precision of the singluar value truncation
        #singular values will be discarded if x/x_{max}<10^(-0.1*self.prec)
        #!!!!note that self.prec>150 gives precision comparable to intrinsic numerical error
        #so shouldn't really make it that high!!!!
        self.prec=0                         
        
        #list containing info about computation time and size of tensors at each propagation step:
        #--time to contract mps/mpo
        #--list of bond dimensions along mps
        #--total number of elemenets of mps
        self.diagnostics=[]
        
        #keeping track of which point of propagation the system is at
        self.point=0                        
        
        #the discretisation timestep
        self.dt=0                           
        
        #list of times and vectorised reduced density matrices output by tempo at those times
        self.statedat=[[],[]]
        
        #name of the system which will label all files the system outputs
        self.name='temp'
    
        self.mps=mps_block(0,0,0)           #blank mps block
        self.mpo=mpo_block(0,0,0)           #blank mpo block
       
    def set_filename(self,name_string):
        #sets the name of the system
        if type(name_string)==str:
            self.name=name_string
        else:
            print('filename needs to be a string')
            
    def checkdim(self,op_array):
        #checks that input arrays have the correct dimensions
        if shape(op_array)==(self.dim,self.dim):
            return 0
        else:
            print('input operator has wrong dims: '+str(shape(op_array)))

    def set_state(self,state_array):
        #sets the initial state of the system
        self.checkdim(state_array)
        self.istate=state_array.reshape(self.dim**2)
    
    def set_hamiltonian(self,ham):
        #sets the hamiltonian of the system
        self.checkdim(ham)
        self.ham= -1j*(kron(ham,eye(self.dim)) - kron(eye(self.dim),ham.conj()))
        
    def add_bath(self,bath_list):
        #attaches a bath to the system
        #takes a list whose elements have the form:
        #[diagonal hilbert space system operator coupled to bath, eta(t) of bath]
        for el in bath_list:
            self.checkdim(el[0])
            #constructing commutating and anticommutating operators for each coupling
            comm=kron(el[0].diagonal(),ones(self.dim))-kron(ones(self.dim),el[0].diagonal())
            acomm=kron(el[0].diagonal(),ones(self.dim))+kron(ones(self.dim),el[0].diagonal())
            
            #if the timestep has been set already then calculate Makri coeffs, else leave blank
            if self.dt>0: 
                self.intparam.append([[comm,acomm],el[1],self.getcoeffs(el[1])])
            else: self.intparam.append([[comm,acomm],el[1],[]])
            self.nbaths=self.nbaths+1
    
    def find_degeneracy(self):
        #finds degeneracy in tempo site tensor based on degeneracy in system coupling superoperators
        
        clis=zeros((2*self.nbaths,self.dim**2))
        ii=0
        for el in self.intparam:
            clis[ii]=el[0][0]
            clis[ii+self.nbaths]=el[0][1]
            ii=ii+1
            
        b = ascontiguousarray(clis[:self.nbaths].T).view(dtype((void, (clis[:self.nbaths].T).dtype.itemsize * (clis[:self.nbaths].T).shape[1])))
        uh=unique(b,return_index=True,return_inverse=True)
        print('W/E degen: '+str(len(el[0][0]))+' to '+str(len(uh[0])))
        b = ascontiguousarray(clis.T).view(dtype((void, (clis.T).dtype.itemsize * (clis.T).shape[1])))
        uv=unique(b,return_index=True,return_inverse=True)
        print('N/S degen: '+str(len(el[0][0]))+' to '+str(len(uv[0])))
        self.deg=[[len(uh[0]),uh[1],uh[2]],[len(uv[0]),uv[1],uv[2]]]
     
    def convergence_params(self,dt_float,dkmax_int,truncprec_int):
        #sets the convergence parameters and calculates Makri coefficients is baths have already been added
        self.dkmax=dkmax_int
        self.prec=truncprec_int
        self.dt=dt_float
        for el in self.intparam: el[2]=self.getcoeffs(el[1])

    def getcoeffs(self,eta_function):
        #calculates makri coeffs by taking second order finite derivatives of an eta(t) function
        
        #tb is discretized eta(t) in form of a list
        tb=list(map(eta_function,array(range(self.dkmax+2))*self.dt))
        etab=[tb[1]]
        for jj in range(1,self.dkmax+1):
            etab.append(tb[jj+1]-2*tb[jj]+tb[jj-1])
        return etab

    def sysprop(self,j):
        #constructs system propagator at timestep j - the j dependence is pointless currently 
        #but is built to allow for time dependent hamiltonians in future
        return expm(self.ham*self.dt/2).T            
                  
    def itab(self,dk):
        #function to store the influence function I_dk as a table
        #loop through all baths multiplyin in the corresponding matrix
        vec1=zeros(self.dim**2)
        vec2=zeros(self.dim**2)
        for el in self.intparam:
            #picking out the correct eta coeff
            vec1=vec1+el[0][0]
            eta_dk=el[2][dk]
            #print(eta_dk)
            vec2=vec2+eta_dk.real*el[0][0]+1j*eta_dk.imag*el[0][1]
        if dk>1:
            vec1=array([vec1[i] for i in self.deg[0][1]])
            vec2=array([vec2[i] for i in self.deg[1][1]])
        
        iffac=2.7182818284590452353602874713527**(outer(vec2,-vec1))
   
        if dk==0: return iffac.diagonal() #I_0 is a funtion of one varibale only so is converted to vector
        else: return iffac
    
    def temposite(self,dk):
        #converts 2-leg I_dk table into a 4-leg tempo mpo_site object
        iffac=self.itab(dk)
        #print(iffac)
        if dk==1:
            iffac=(iffac*self.itab(0))
            iffac=iffac*dot(self.sysprop(self.point-1),self.sysprop(self.point))
            #initialise 4-leg tensor that will become mpo_site and loop through assigning elements
            tab=zeros((self.deg[1][0],self.dim**2,self.dim**2,self.deg[0][0]),dtype=complex)
            for i1 in range(self.dim**2):
                for a1 in range(self.dim**2):
                    tab[self.deg[1][2][i1]][i1][a1][self.deg[0][2][a1]]=iffac[i1][a1]
        
        else:
            tab=zeros((self.deg[1][0],self.deg[1][0],self.deg[0][0],self.deg[0][0]),dtype=complex)
            for i1 in range(self.deg[1][0]):
                for a1 in range(self.deg[0][0]):
                    tab[i1][i1][a1][a1]=iffac[i1][a1]

        if dk>=self.dkmax or dk==self.point:
            #if at an end site then sum over east leg index and replace with 1d dummy leg
            return mpo_site(tens_in=expand_dims(dot(tab,ones(tab.shape[3])),-1))
        else:        
            return mpo_site(tens_in=tab)
    
    def getstate(self):
        self.state=dot(self.mps.readout(),self.sysprop(self.point-1))
                         
    def prep(self):
        #prepares system to be propagated once params have been set
        self.find_degeneracy()
        self.mps=mps_block(0,0,0)           #blank mps block
        self.mpo=mpo_block(0,0,0)           #blank mpo block
        self.state=self.istate
        self.statedat=[[0],[self.state],[self.dkmax,self.prec]]     #store initial state in data
        self.point=1                        #move to first point

        #propagte initial state half a timestep with sys prop and multiply in I_0 to get initial 1-leg ADT           
        self.mps.insert_site(0,expand_dims(expand_dims(
                dot(self.state,self.sysprop(0))*self.itab(0)
                                    ,-1),-1))
        self.getstate()
        
        #append first site to mpo to give a length=1 block
        self.mpo.append_mposite(self.temposite(1))
    
    def prop(self,kpoints=1,savemps=False):
        #propagates the system for ksteps - system must be prepped first
        for k in range(kpoints):
            
            t0=time()
            #find current time physical state and store this in self.dat
            self.statedat[0].append(self.point*self.dt)
            self.statedat[1].append(self.state)
            
            #contract and grow the mps using the mpo
            self.mps.contract_with_mpo(self.mpo,prec=10**(-0.1*self.prec),trunc_mode='accuracy')
            self.mps.insert_site(0,expand_dims(eye(self.dim**2),1))
            #move the system forward a point
            self.point=self.point+1
            self.getstate()
            #replace dk=1 tempo site - only really necessary for time dependent hamiltonians
            self.mpo.data[0]=self.temposite(1)
            if self.point<self.dkmax+1:
                #this is the growth stage: mpo has end site updated and a new site appended
                self.mpo.data[-1]=self.temposite(self.point-1)
                self.mpo.append_mposite(self.temposite(self.point)) 
            else:
                #if not using new quapi and point>kmax just contract away end site of mpo
                self.mps.contract_end()
            print("point:" +str(self.point)+' time:'+str(time()-t0)+' dkm:'+str(self.dkmax)+' pp:'+str(self.prec))        
            print('max dim: '+str(max(self.mps.bonddims()))+' tot size: '+str(self.mps.totsize()))
            
            self.diagnostics.append([time()-t0,self.mps.bonddims(),self.mps.totsize()])
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
 
    def savesys(self):
        dump(self,open(self.name+"_sys_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
        
    def getopdat(self,op):
        od=[]
        for da in self.statedat[1]:
            od.append(dot(op,da.reshape((self.dim,self.dim))).trace().real)
        return [self.statedat[0],od]
    
def fo(w,T,t):
    if T==0: return w**(-2)*((1-cos(w*t))+1j*(sin(w*t)-w*t))
    else: return w**(-2)*(coth(w/(2*T))*(1-cos(w*t))+1j*(sin(w*t)-w*t))
    return fo

def numint(t,T,nin): 
    if t == 0:
        eta = 0
    else:
        numir = quad(lambda w: nin(w)*(fo(w,T,t).real),0,inf)
        numii = quad(lambda w: nin(w)*(fo(w,T,t).imag),0,inf)
        eta = numir[0]+1j*numii[0]
    return eta