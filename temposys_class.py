#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:15:33 2018

@author: aidan
"""
from numpy import exp,swapaxes,reshape,diag,cos,sin,inf,ascontiguousarray,unique,array, expand_dims, kron, eye, dot, ones, outer, zeros, shape, dtype, void
from mpmath import coth
from scipy.integrate import quad
from pickle import dump
from time import time
from scipy.linalg import expm
from mpsmpo_class import mps_block, mpo_block
from pathos.multiprocessing import ProcessingPool as Pool
import sys

class bath(object):
    #the reason for a separate class for baths is in anticipation of multiple baths
    def __init__(self,op,Jw=0,T=0,eta=None):
        #initialise with the reduced system operator the bath couples to 'op'
        #and either: a spectral density 'Jw' and temperature 'T'
        #or an analytic eta function 'eta'
        
        #get the dimension of the reduced system
        self.dim=op.shape[0]
        #construct commutator and anticommutator superoperators from 'op'
        self.comm=kron(op,eye(self.dim)) - kron(eye(self.dim),op.T)
        self.acomm=kron(op,eye(self.dim)) + kron(eye(self.dim),op.T)
        #from comm and acomm get the degeneracy info for West/East and North/South legs separately
        self.WEdeg=self.row_degeneracy([self.comm.diagonal()])
        self.NSdeg=self.row_degeneracy([self.comm.diagonal(),self.acomm.diagonal()])
        
        #either use the spectral density and temperature to numerically obtain eta function
        #or use the input eta function
        if Jw!=0:
            self.num_eta(T,Jw)
        else:
            self.eta_fun=eta
        #initialse the discretisation timestep
        self.dt=0
        
        #initialise list of discrete eta(t) - eta_list=[eta(0),eta(dt),eta(2 dt),..]
        self.eta_list=[]
        
        #################################################################
        #########Calculation of coefficients defined in Eq.(13)############
        #############################################################
        #we define the function eta(t)=\int_0^t dt'  \int_0^t' dt'' C(t'')
        #with C(t) as defined in Eq.(14)
        #Then Eq.(13) top is written: 
        #eta_dk=( eta(dt (dk+1))-eta(dt dk) ) - ( eta(dt dk)-eta(dt (dk-1)) )
        #and Eq.(13) bottom:
        #eta_0=eta(dt)
        #Thus we find eta_dk by calculating eta(t), either numerically or analytically
        #at a finite number of timesteps and taking finite differences between them.
        #An alternative, perhaps numerically easier way to get them is to recognise
        #that actually eta_dk=C(dt dk)dt^2 in the dt->0 limit
    
    def num_eta(self,T,Jw,subdiv=1000):
        #function that numerically calculates lineshape eta(t) for a given bath at temperature T,
        #initially in thermal equilibrium  whith correlation function given by Eq.(14)
        #Important Note: If the integration fails to converge it will only give warnings and
        #going ahead with the propagation can cause blow up of mps bond dimensions - hence
        #the subdivisions optional argument which is set high
        #Because the max subdivisions is set high the integrals can be slow
        #and we might need to do a couple hundred of them - hence the parallelisation in
        #self.discretise()
        
        #eta function defintion: eta(t)=\int_0^t dt'  \int_0^t' dt'' C(t'')
        #Time integration over Eq.(14) can be done anayltically leaving an improper
        #intergal over \omega which we perform numerically
        
        #define real and imaginary parts separately and also take temperature=0
        #as separate special case
        def intRe(t): 
            return quad(lambda w: w**(-2)*Jw(w)*(1-cos(w*t)),0,inf,limit=subdiv)[0]      
        def intReT(t): 
            return quad(lambda w: w**(-2)*Jw(w)*(1-cos(w*t))*coth(w/(2*T)),0,inf,limit=subdiv)[0]
        def intIm(t): 
            return quad(lambda w: w**(-2)*Jw(w)*(sin(w*t)-w*t),0,inf,limit=subdiv)[0]
        
        def eta(t):      
            if T==0: 
                return intRe(t)+1j*intIm(t)
            else:
                return intReT(t)+1j*intIm(t)
            
        self.eta_fun=eta
    
    def discretise(self,dt,kmax=0):
        #function to evaluate eta(t) at a discrete set of points with timestep dt
        #and calculating eta(k dt) up to k=kmax+2 (this is to allow kmax=0 to correspond to using no memory cutoff)
        ctime=time()
        #check that the timestep is one that has already been used - because we might have already calculated some eta(k*dt)
        #for a given timestep and want to avoid reevaluting integrals
        #if a new timestep then reset both internal timestep and the eta list
        if self.dt != dt:
            print('setting bath timestep')
            self.dt=dt
            self.eta_list=[]
        
        #if len(self.eta_list)>kmax+2:
        #    return 0
        print('discretising...')
        
        #going to use multiprocessing since we might need hundreds of numerical integrals that
        #need carried out to high precision
        #using Pool from module pathos because it uses dill, not pickle, so can deal with locally
        #defined functions
        with Pool() as pool:
            try:
                #if the pool is already running then reset it - if already use it retains previous results
                #and we shoudl clear them before going on
                pool.restart()
            except(AssertionError): 
                pass
            #evaluate the eta function at a discrete set of points using imap
            #if there are already entries in the list then start from the next
            #required eta(k*dt) and calculate up to kmax+2
            ite=list(pool.imap(self.eta_fun,array(range(len(self.eta_list),kmax+3))*self.dt))
            #close the pool
            pool.close()
            pool.join()
            pool.clear()
        #get the list of values   
        for el in ite:
            self.eta_list.append(el)
        ##### For the non-parallel version replace the 'with Pool()' section with: ite=list(map(self.eta_fun,array(range(self.dkmax+2))*self.dt))'
        print('time: '+str(round(-ctime+time(),2)))
               
    def row_degeneracy(self,matrix):
        #finds degenerate rows of a matrix
        #needed instead of just the function 'unique' to find common degeneracy in comms and acomms
        mat=array(matrix)
        #some magic here to get a vector v which has same degeneracy structure as rows of matrix mat
        #I lifted this straight from a stackexchange thread
        v = ascontiguousarray(mat.T).view(dtype((void, (mat.T).dtype.itemsize * (mat.T).shape[1])))
        #find degeneracy in v - 'unique' returns:
        #[list of unique values,positions in v of the a single instance of each unique val,list same length of v with each element a position in the list of unique vals to map back to v]
        un=unique(v,return_index=True,return_inverse=True)
        #dont explicitly need the unique vals (arent correct anyway since we converted mat to v) - just how many there are
        return [len(un[0]),array(un[1]),un[2]]
    
    def I_dk(self,dk,unique=False):
        #creates the rank-2 tensor I_dk(j,j') of Eq.(11) but without free propagator when dk=1
        #acheives this by taking outer product of two rank-1 vectors to create Eq.(12) as rank-2 tensor
        #then exponentiate each element
        
        #these rank-2 tensors are used to contruct the rank-4 b tensors in Eq.(22)

        #Note this requires O to be diagonal as it just takes diagonal components
        Om=self.comm.diagonal()
        Op=self.acomm.diagonal()

        if dk==0:
            #for dk=0 then I_0(j,j) is really just a function of 1 variable - so
            #a rank-1 tensor we need - form vector by taking element by element product
            #and exponential all elements
            eta_dk=self.eta_list[1]
            Idk=exp(-Om*(eta_dk.real*Om+1j*eta_dk.imag*Op))
            if unique:
                #if just needing the unique value then insert calculated array positions
                #of a single instance of the uniques to reduce I_dk to its uniques
                Idk=Idk[self.NSdeg[1]]
        else:
            #find eta_dk by taking finite difference on the discetised eta(t)
            eta_dk=self.eta_list[dk+1]-2*self.eta_list[dk]+self.eta_list[dk-1]
            #need rank-2 tensor so take outer product of vectors and exponeniate each element
            Idk=exp(-outer(eta_dk.real*Om+1j*eta_dk.imag*Op,Om))
            if unique:
                #find the array of only unique values if so desired
                Idk=(Idk[self.NSdeg[1]].T)[self.WEdeg[1]].T
        
        return Idk

            
class temposys(object):
    def __init__(self,hilbert_dim=2):
        #initialise system object with hilbert space dimension d=hilbert_dim
        self.dim=hilbert_dim
        
        #initialise the Hamiltonian commutator superoperator
        #acts on d^2 size vector so has dimension (d^2,d^2)
        self.ham=zeros((self.dim**2,self.dim**2)) 
        
        #self.b will be a bath object which is used to biuld TEMPO tensors
        self.b=None                   

        
        #instantaneous reduced system density matrix - vectorised
        self.state=array(self.dim**2)
        
        #initial reduced system density matrix - stored separate from instantaneous state
        self.istate=array(self.dim**2)   
        
        self.freeprop=array((self.dim**2,self.dim**2))
        
        #memory length/maximum length of the mps, this is K in the paper
        #the initial value dkmax=0 corresponds to special case of using no memory cutoff
        self.dkmax=0
        
        #sets the precision of the singluar value truncation
        #singular values will be discarded if x/x_{max}<10^(-0.1*self.prec)
        #!!!!note that self.prec>150 gives precision comparable to intrinsic numerical error
        #so shouldn't really make it that high!!!!
        self.prec=0                         

        #keeping track of which point of propagation the system is at
        self.point=0                        
        
        #the discretisation timestep, this is \Delta in the paper
        self.dt=0                           
        
        #list of times and vectorised reduced density matrices output by tempo at those times
        self.statedat=[[],[]]
        
        #name of the system which will label all files the system outputs
        self.name='temp'
        
    
    def comm(self,op):
        #constructs commutator superoperator of Hilbert space operator op
        return kron(op,eye(self.dim)) - kron(eye(self.dim),op.T)   
    def acomm(self,op):
        #constructs anticommutator superoperator of Hilbert space operator op
        return kron(op,eye(self.dim)) + kron(eye(self.dim),op.T)
       
    def set_filename(self,name_string):
        #sets the name of the system
        if type(name_string)==str:
            self.name=name_string
        else:
            print('filename needs to be a string')
           
    def checkdim(self,op_array):
        #checks that op_array is a hilbert space operator
        if shape(op_array)==(self.dim,self.dim):
            return 0
        else:
            print('input operator has wrong dims: '+str(shape(op_array)))
            sys.exit()

    def set_state(self,state_array):
        #sets the initial state of the system
        self.checkdim(state_array)
        self.istate=state_array.reshape(self.dim**2)
    
    def set_hamiltonian(self,ham):
        #constructs reduced system liouvillian and stores as self.ham
        self.checkdim(ham)
        self.ham= -1j*self.comm(ham)
        #if discretized then construct the free propagator
        #note dt/2 due to symmetrized trotter splitting
        #also note the transpose .T -- we are using the nonstanard convention of
        #propagating row vectors (instead of column), multiplying matrices in from the right
        if self.dt>0: self.freeprop=expm(self.ham*self.dt/2).T 
        #if statement is so that 'set_hamiltonian' and 'convergence_params' commute
                              
    def get_state(self):
        #readout reduced state from ADT by summing over indices - built in function of mps object
        #also propagate reduced state a final half timestep under free propagation due to symmetric trotter splitting
        self.state=dot(self.mps.readout(),self.freeprop)
        #append current time and state to the data list
        self.statedat[0].append(self.point*self.dt)
        self.statedat[1].append(self.state)
    
    
    def convergence_params(self,dt=0,dk=0,prec=0):
        #sets the convergence parameters
        #only set dk and prec if they are integers
        if type(dk)==int and dk>0: 
            self.dkmax=dk
        if type(prec)==int and prec>0: 
            self.prec=prec
        if dt>0:
            #if setting dt then also calculate freeprop
            self.dt=dt
            self.freeprop=expm(self.ham*self.dt/2).T
    
    def add_bath(self,op,Jw=0,T=0,eta=None):
        #add a bath specifying what operator 'op' it couples to and either
        #a spectral density and temperature Jw_T=[Jw,T] ot an eta function
        self.checkdim(op)
        self.b=bath(op,Jw,T,eta)                        
    
    def getopdat(self,op):
        self.checkdim(op)
        #extracts data for time evolution of expectation of hilbert space operator op
        #initialise data list
        od=[]
        #loop through reduced density matrix data converting back into hilbert space and taking expectation
        for da in self.statedat[1]:
            od.append(dot(op,da.reshape((self.dim,self.dim))).trace().real)
        #retrun list of times and data list
        return [self.statedat[0],od]
        
    
    def b_tensor(self,dk):
        #converts rank-2 I_dk tensor into a 4-leg b tensor Eq.(22) taking account of degeneracy   
        if dk==1:
            #if dk=1 then multiply in free propagator - note we also include I_0 here instead of b_0 like in Methods section    
            iffac=(self.b.I_dk(1)*self.b.I_dk(0))*dot(self.freeprop,self.freeprop)
            
            #initialise 4-leg tensor dimensions based on degeneracy
            #for dk=1 we can only use the degeneracy/partial summing technique on South and East legs (see mpsmpo_class.py)
            tab=zeros((self.b.NSdeg[0],self.dim**2,self.dim**2,self.b.WEdeg[0]),dtype=complex)
            #loop through assigning elements of 4-leg from elements the 2-leg
            for i1 in range(self.dim**2):
                for a1 in range(self.dim**2):
                    tab[self.b.NSdeg[2][i1]][i1][a1][self.b.WEdeg[2][a1]]=iffac[i1][a1]
        
        else:
            #start by constructing an array out of the unique elements in itab
            tab=self.b.I_dk(dk,unique=True)
            #combine the 2 legs(axes) into vector and create matrix with this vector as diagonal
            tab=diag(reshape(tab,(self.b.NSdeg[0]*self.b.WEdeg[0])))
            #reshape the matrix into a 4-leg
            tab=reshape(tab,(self.b.NSdeg[0],self.b.WEdeg[0],self.b.NSdeg[0],self.b.WEdeg[0]))
            #put axes in the right place
            tab=swapaxes(tab,1,2)
            
        if dk==self.point or (dk==self.dkmax and self.dkmax>0):
            #if at an end site then sum over external leg and replace with 1d dummy leg
            return expand_dims(dot(tab,ones(tab.shape[3])),-1)
        else:        
            return tab
                         
    def prep(self):
        print('preparing temposys for propagation')
        #prepares system to be propagated once params have been set
        
        #initialise mps and mpo - mps needs the truncation precision its going to be kept at
        #prespecified - this can be changed during propagation if thats what youre into
        self.mps=mps_block(prec=10**(-0.1*self.prec))           
        self.mpo=mpo_block()          
        
        #set initial instantaneous state and list of states and times that will be calculated
        self.state=self.istate
        self.statedat=[[0],[self.state],[self.dkmax,self.prec]]
        
        #discretise the bath eta(t) function, for dkmax=0 it calculates
        #eta(k dt) for k=0,1,2 - just enough to get eta_dk for dk=0,1 which we need below
        self.b.discretise(self.dt,self.dkmax)
        #propagate initial state, multiply in I_0 and insert into
        #mps to create initial rank-1 ADT, as in Eq.(17)
        #Note only propagating init state dt/2 due to symmetric trotter splitting
        #and using expand dims to turn 1-leg init state into 3-leg tensor with 2 1d dummy indices
        self.mps.insert_site(0,expand_dims(expand_dims(
                dot(self.state,self.freeprop)*self.b.I_dk(0)
                                    ,-1),-1))
        
        #system now prepped at point 1
        self.point=1  
        #insert site to mpo object to give 1-site TEMPO
        self.mpo.insert_site(0,self.b_tensor(1))
        
        #get the reduced state at point 1
        self.get_state()
        
    def prop(self,kpoints=1):
        #if no dkmax was set then no memory cutoff used and we need to calculate
        #eta(k*dt) up to N+1 where N is the point the system ends up at after
        #propagating kpoints
        if self.dkmax==0:
            self.b.discretise(self.dt,self.point+kpoints+1)
        print('propagating')
        ptime=time()
        #propagates the system for kpoints steps - system must be prepped first
        for k in range(kpoints):
            #print('dims: '+str(self.mps.bonddims()))
            t0=time()          
            #contract ADT with TEMPO performing svds and truncating 
            self.mps.contract_with_mpo(self.mpo)
            #self.mps.contract_with_mpo(self.mpo)
            self.mps.insert_site(0,expand_dims(eye(self.dim**2),1))            
            #move the system forward a point and get the state data
            self.point=self.point+1
            self.get_state()
          
            if self.point<self.dkmax+1 or self.dkmax==0:
                #while  in the growth stage we need to update the 'K'th site to give it an extra leg
                self.mpo.sites[-1].update(self.b_tensor(self.point-1))
                #and then append the new end site
                self.mpo.insert_site(self.point-1,self.b_tensor(self.point))
            else:
                #after the growth stage the TEMPO remains the same at each step of propagation
                #but we now need to contract one leg of the ADT as described in paper
                self.mps.contract_end()
            #print out the current point and time it took to contract
            print(str(self.point)+'/'+str(self.point+kpoints-k-1)+'  time: '+str(round(time()-t0,2)))
            #dump the data for the reduced state to a pickle file
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"_prec"+str(self.prec)+".pickle",'wb'))
        print('prop time: ' +str(round(time()-ptime,2)))  

    
 