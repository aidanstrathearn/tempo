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
from mpsmpo_class import mps_block, mpo_block
from pathos.multiprocessing import ProcessingPool as Pool


class temposys(object):
    def __init__(self,hilbert_dim=2):
        #initialise system object with hilbert space dimension d=hilbert_dim
        self.dim=hilbert_dim
        
        #initialise the Hamiltonian commutator superoperator
        #acts on d^2 size vector so has dimension (d^2,d^2)
        self.ham=zeros((self.dim**2,self.dim**2)) 
        
        #intialise list that will contain all info about bath coupling
        #--intparam[0]=commutator/anticommutator superoperators of system operator coupled to bath
        #--intparam[1]=the lineshape function, eta(t), of the bath
        #--intparam[2]=the makri coefficients eta_{kk'}, given by second order finite differences on eta(t)       
        self.intparam=[]                    
        
        #self.deg will contain degeneracy info of tempo site tensors:
        #for vertical and horizontal legs of tensor separately:
        #--the number of unique elements
        #--the positions in the multidimensional array of a single occurance of each unique element
        #--an array the same size as the full tensor, whose elements are the positions in the list 
        #of unique elements so that the full multidimensional array can be reconstructed
        self.deg=[]
        
        #instantaneous reduced system density matrix - vectorised
        self.state=array(self.dim**2)
        
        #initial reduced system density matrix - stored separate from instantaneous state
        self.istate=array(self.dim**2)   
        
        self.freeprop=array((self.dim**2,self.dim**2))
        #memory length/maximum length of the mps, this is K in the paper
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
    
    def convergence_params(self,dt_float,dkmax_int,truncprec_int):
        #sets the convergence parameters
        self.dkmax=dkmax_int
        self.prec=truncprec_int
        #calculates Makri coefficients if baths have already been added
        if len(self.intparam)>1 and self.dt != dt_float:
            self.dt=dt_float
            self.intparam[2]=self.getcoeffs(self.intparam[1])
        #sets free propagator
        self.freeprop=expm(self.ham*self.dt/2).T 
                          
    def getopdat(self,op):
        #extracts data for time evolution of expectation of hilbert space operator op
        #initialise data list
        od=[]
        #loop through reduced density matrix data converting back into hilbert space and taking expectation
        for da in self.statedat[1]:
            od.append(dot(op,da.reshape((self.dim,self.dim))).trace().real)
        #retrun list of times and data list
        return [self.statedat[0],od]
    
    def num_eta(self,T,Jw,subdiv=1000):
        #function that numerically calculates lineshape eta(t) for a given bath at temperature T,
        #initially in thermal equilibrium  whith correlation function given by Eq.(14)
        #Important Note: If the integration fails to converge it will only give warnings and
        #going ahead with the propagation can cause blow up of mps bond dimensions - hence
        #the subdivisions optional argument which is set high
        #Because the max subdivisions is set high the integrals can be slow
        #and we might need to do a couple hundred of them - hence the parallelisation in
        #self.getcoeffs()
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
            
        return eta


    def getcoeffs(self,eta_function):
        print('calculating eta_dk coefficients')
        ctime=time()
        #calculates makri coeffs by taking second order finite differences of an eta(t) function
        #this is not how we define them in the paper but is equivalent
        #
        #eta function defintion: eta(t)=\int_0^t dt'  \int_0^t' dt'' C(t'')
        #with C(t) as defined in Eq.(14)
        #Then Eq.(13) top is written: 
        #eta_dk=( eta(dt (dk+1))-eta(dt dk) ) - ( eta(dt dk)-eta(dt (dk-1)) )
        #and Eq.(13) bottom:
        #eta_0=eta(dt)

        #tb is discretized eta(t) in form of a list tb=[eta(dt),eta(2 dt),eta(3 dt), ...]
        
        #using Pool from module pathos because it uses dill, not pickle, so can deal with locally
        #defined functions
        with Pool() as pool:
            try:
                #if the pool is already running then reset it - if already use it retains previous results
                #and we shoudl clear them before going on
                pool.restart()
            except(AssertionError): 
                pass
            #evaluate the eta function at a discrete set of points in parallel
            ite=pool.imap(eta_function,array(range(self.dkmax+2))*self.dt)
            #close the pool
            pool.close()
            pool.join() 
        #get the list of values   
        tb=list(ite)      
        ##### For the non-parallel version use: tb=list(map(eta_function,array(range(self.dkmax+2))*self.dt))'
        
        #initial list of eta_dk's: first coeff is just eta_0=eta(dt), this is Eq.(13) bottom
        etab=[tb[1]]
        #now take finite differences on tb to get coeffs
        for jj in range(1,self.dkmax+1): etab.append(tb[jj+1]-2*tb[jj]+tb[jj-1])
        print('coefficient/integration time: '+str(round(-ctime+time(),2)))
        return etab
               
    def add_bath(self,b_list):
        #attaches a bath to the system
        #b_list should have form [hilbert space operator coupled to bath,eta(t) of bath]    
        self.intparam=[[self.comm(b_list[0]).diagonal(),self.acomm(b_list[0]).diagonal()],b_list[1],[]]
        #print(b_list[1](1))
        #if the timestep has been set already then calculate Makri coeffs, else leave blank
        #if statement is to make 'add_bath', 'set_hamiltonian' and 'convergence_params' commute
        if self.dt>0: 
            self.intparam[2]=self.getcoeffs(b_list[1])
        #find degeneracy which allows for partial summing of tensor network
        #for degeneracy in 'alpha' legs of 4-leg b tensor only need to find degeneracy in commutator superoperator
        #for degeneracy in 'j' legs need to find common degeneracy in commutators and anticommutators
        self.deg=[self.row_degeneracy(self.intparam[0][:1]),self.row_degeneracy(self.intparam[0])]

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
        return [len(un[0]),un[1],un[2]]
    
    def itab(self,dk):
        #creates the rank-2 tensor I_dk(j,j') of Eq.(11) but without free propagator when dk=1
        #acheives this by taking outer product of two rank-1 vectors to create Eq.(12) as rank-2 tensor
        #then exponentiate each element
        #pick out tthe correct eta_dk and the commutator (Om) and acommutator (Op)
        eta_dk=self.intparam[2][dk]
        [Om,Op]=self.intparam[0]
        #take outer product of vectors and exponentiate- element-by-element NOT matrix exponential
        #yes thats the number e, I think it looks prettier written like this...
        iffac=2.7182818284590452353602874713527**(outer(eta_dk.real*Om+1j*eta_dk.imag*Op,-Om))
        #I_0 is a funtion of one varibale - a vector - so take the diagonal of full matrix
        if dk==0: return iffac.diagonal()
        else: return iffac
    
    def tempotens(self,dk):
        #converts rank-2 itab tensor into a 4-leg tempo mpo_site object taking account of degeneracy
        
        #initialise tensor
        iffac=self.itab(dk)
        if dk==1:
            #if dk=1 then multiply in free propagator - note we also include I_0 here instead of b_0 like in Methods section
            iffac=(iffac*self.itab(0))         
            iffac=iffac*dot(self.freeprop,self.freeprop)
            
            #initialise 4-leg tensor dimensions based on degeneracy
            #for dk=1 we can only use the degeneracy/partial summing technique on legs not connected to freeprop
            tab=zeros((self.deg[1][0],self.dim**2,self.dim**2,self.deg[0][0]),dtype=complex)
            #loop through assigning elements of 4-leg from elements the 2-leg
            for i1 in range(self.dim**2):
                for a1 in range(self.dim**2):
                    tab[self.deg[1][2][i1]][i1][a1][self.deg[0][2][a1]]=iffac[i1][a1]
        
        else:
            #initialise 4-leg tensor dimensions based on degeneracy
            tab=zeros((self.deg[1][0],self.deg[1][0],self.deg[0][0],self.deg[0][0]),dtype=complex)
            #loop through assigning elements of 4-leg from elements the 2-leg
            for i1 in range(self.dim**2):
                for a1 in range(self.dim**2):
                    tab[self.deg[1][2][i1]][self.deg[1][2][i1]][self.deg[0][2][a1]][self.deg[0][2][a1]]=iffac[i1][a1]

        if dk>=self.dkmax or dk==self.point:
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
        
        #create initial rank-1 ADT, as in Eq.(17), as an mps object
        #Note only propagating init state dt/2 due to symmetric trotter splitting
        #and using expand dims to turn 1-leg init state into 3-leg tensor with 2 1d dummy indices
        self.mps.insert_site(0,expand_dims(expand_dims(
                dot(self.state,self.freeprop)*self.itab(0)
                                    ,-1),-1))
        
        #append insert site to mpo object to give 1-site TEMPO
        self.mpo.insert_site(0,self.tempotens(1))
        
        #system now prepped at point 1
        self.point=1  
        #get the reduced state at point 1
        self.get_state()
        
    def prop(self,kpoints=1):
        print('propagating')
        ptime=time()
        #propagates the system for kpoints steps - system must be prepped first
        for k in range(kpoints):       
            t0=time()          
            #contract ADT with TEMPO performing svds and truncating 
            self.mps.contract_with_mpo(self.mpo)            
            #move the system forward a point and get the state data
            self.point=self.point+1
            self.get_state()
          
            if self.point<self.dkmax+1:
                #while  in the growth stage we need to update the 'K'th site to give it an extra leg
                self.mpo.sites[-1].update(self.tempotens(self.point-1))
                #and then append the new end site
                self.mpo.insert_site(self.point-1,self.tempotens(self.point))
            else:
                #after the growth stage the TEMPO remains the same at each step of propagation
                #but we now need to contract one leg of the ADT as described in paper
                self.mps.contract_end()
            #print out the current point and time it took to contract
            print(str(self.point)+'/'+str(kpoints)+'  time: '+str(round(time()-t0,2)))          
            #obtain mps info of current ADT
            self.diagnostics.append([time()-t0,self.mps.bonddims(),self.mps.totsize()])
            #dump the data for the reduced state to a pickle file
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"_prec"+str(self.prec)+".pickle",'wb'))
        print('prop time: ' +str(round(time()-ptime,2)))