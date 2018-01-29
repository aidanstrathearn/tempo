#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:56:29 2018

@author: aidan
"""
from numpy import array, expand_dims, kron, eye, dot, append, ones, outer, zeros, shape
from numpy.fft import fft, fftfreq, fftshift
from matplotlib.pyplot import plot, show, legend, subplot
from pickle import dump
from time import time
from mpmath import exp
from scipy.linalg import expm
from MpsMpo_block_level_operations import mps_block, mpo_block
from MpsMpo_site_level_operations import mpo_site

class temposys(object):
    def __init__(self,hilbert_dim):
        #initialise system object with hilbert space dimension = hilbert_dim
        self.dim=hilbert_dim
        
        self.ham=lambda t: zeros((self.dim**2,self.dim**2))    #the reduced system hamiltonian is now a function of time
        self.diss=zeros((self.dim**2,self.dim**2))                    #list of lindblads and rates: [[rate1,oper1],[rate2,oper2],...]
        self.intparam=[]                    #lists of interaction op eigenvalues and bath etas: [[eigs1,eta1],[eigs2,eta2],..]
        self.ops=[]
        self.state=array(self.dim**2)           #reduced system density matrix
        self.istate=array(self.dim**2)           #reduced system density matrix
        self.dkmax=0                        #maximum length of the mps
               
        self.prec=0                         #precision used in svds
        self.ntot=1
        
        self.mod=0                          #maximum point to use new quapi coeffs up to
        self.point=0                        #current point in time system is at
        self.dt=0                           #size of timestep
        self.statedat=[[],[]]
        self.corrdat=[[],[]]                         #data list: [[time1,state1],[time2,state2],...]
        self.name='temp'

    
    def set_filename(self,name_string):
        if type(name_string)==str:
            self.name=name_string
        else:
            print('filename needs to be a string')
            
    def checkdim(self,op_array):
        if shape(op_array)==(self.dim,self.dim):
            return 0
        else:
            print(op_array)
            print('input operator has wrong dims: '+str(shape(op_array)))
            #exit()
    
    def set_endtime(self,tfinal_float,ntot_integer):
        self.ntot=ntot_integer
        self.dt=tfinal_float/ntot_integer
        
    def set_state(self,state_array):
        self.checkdim(state_array)
        self.istate=state_array.reshape(self.dim**2)
        
    def set_hamiltonian(self,ham_function):
        self.checkdim(ham_function(0))
        self.ham=lambda t: -1j*(kron(ham_function(t),eye(self.dim)) - kron(eye(self.dim),ham_function(t).conj()))
        return 0
    
    def add_dissipation(self,lind_list):
        for el in lind_list:
            self.checkdim(el[1])
            self.diss=self.diss+el[0]*(kron(el[1],el[1].conj())
                -0.5*(kron(dot(el[1].conj().T,el[1]),eye(self.dim))+kron(eye(self.dim),dot(el[1].conj().T,el[1]).T)))
    
    def add_bath(self,bath_list):
        for el in bath_list:
            self.checkdim(el[0])
            if self.dt>0: self.intparam.append([el[0].diagonal(),el[1],self.getcoeffs(el[1])])
            else: self.intparam.append([el[0].diagonal(),el[1],[]])
    
    def add_operator(self,op_list):
        print('adding operator')
        for el in op_list:
            self.checkdim(el[1])
            self.ops.append([el[0],kron(el[1],eye(self.dim)).T])
          
        
    def convergence_params(self,dt_float,dkmax_int,truncprec_int):
        self.dkmax=dkmax_int
        self.prec=truncprec_int
        self.dt=dt_float
        for el in self.intparam: el[2]=self.getcoeffs(el[1])
            
    def getcoeffs(self,eta_function):
        #tb is list of values of eta evaulated at integer timestep
        tb=list(map(eta_function,array(range(self.dkmax+2))*self.dt))
        #etab takes finite differences on tb to create coeffs
        etab=[tb[1]]+list(map(lambda jj: tb[jj+1]-2*tb[jj]+tb[jj-1],range(1,self.dkmax+1)))
        #if 
        if self.mod>self.dkmax:
            tb=tb+list(map(eta_function,array(range(self.dkmax+2,self.mod+3))*self.dt))             
            etab=list(etab)+list(map(lambda jj: tb[jj+1]-tb[jj]-tb[self.dkmax]+tb[self.dkmax-1],range(self.dkmax+1,self.mod+2)))
        return etab
        
    def sysprop(self,j): 
        return expm((self.ham(j*self.dt)+self.diss)*self.dt/2).T         
              
    def itab(self,dk):
        #function to store the influence function I_dk as a table
        
        #initialise matrix
        iffac=ones((self.dim**2,self.dim**2))
        #loop through all baths multiplyin in the corresponding matrix
        for el in self.intparam:
            #picking out the correct eta coeff
            eta_dk=el[2][dk+(self.point-dk)*int((self.mod and self.point)>self.dkmax and dk>1)]
            #creating arrays of differences and sums of operator eigenvalues
            difs,sums=array([]),array([])
            for s in el[0]: difs,sums=append(difs,s-array(el[0])),append(sums,s+array(el[0]))
            #creating I_dk matrix and multiplying into iffac (couldnt get exp() to work here so used numerical value....)
            iffac=iffac*2.7182818284590452353602874713527**(outer(eta_dk.real*difs+1j*eta_dk.imag*sums,-difs))
        
        if dk==0: return iffac.diagonal() #I_0 is a funtion of one varibale only so is converted to vector
        else: return iffac
    
    def temposite(self,dk):
        #converts 2-leg I_dk table into a 4-leg tempo mpo_site object
        iffac=self.itab(dk)
        #if dk==1 then multiply in system propagator and I_0
        if dk==1:
            iffac=iffac*self.itab(0)
            if len(self.ops)>0 and self.ops[0][0]==self.point:
                print('using op')
                iffac=iffac*dot(self.sysprop(self.point-1),dot(self.ops[0][1],self.sysprop(self.point)))
                del self.ops[0]
            else:
                iffac=iffac*dot(self.sysprop(self.point-1),self.sysprop(self.point))
        
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
        #datfile=open(self.name+".pickle",'wb')
        #prepares system to be propagated once params have been set
        self.mps=mps_block(0,0,0)           #blank mps block
        self.mpo=mpo_block(0,0,0)           #blank mpo block
        self.state=self.istate
        self.statedat=[[0],[self.state],[self.dkmax,self.prec]]     #store initial state in data
        self.point=1                        #move to first point
        #self.getcoeffs()                    #calculate the makri coeffs
        
        #propagte initial state half a timestep with sys prop and multiply in I_0 to get initial 1-leg ADT           
        self.mps.insert_site(0,expand_dims(expand_dims(
                dot(self.state,self.sysprop(0))*self.itab(0)
                                    ,-1),-1))
        
        self.getstate()
        
        #append first site to mpo to give a length=1 block
        self.mpo.append_mposite(self.temposite(1))
        
        
        
    
    def getstate(self):
        self.state=dot(self.mps.readout3(),self.sysprop(self.point-1))
        if len(self.ops)>0 and self.ops[0][0]==self.point: 
            self.state=dot(self.state,self.ops[0][1])
    
    def prop(self,kpoints=1):
        avgt=0
        #propagates the system for ksteps - system must be prepped first
        for k in range(kpoints):
            
            t0=time()
            #find current time physical state and store this in self.dat
            #self.getstate()
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
            elif (self.mod>self.dkmax) or (self.point==self.dkmax+1):
                #beyond the growth stage we either use new quapi and carry on updating end site
                #or update it just once at point=kmax+1 if using old quapi
                self.mpo.data[-1]=self.temposite(self.dkmax)
                self.mps.contract_end()       
            else:
                #if not using new quapi and point>kmax just contract away end site of mpo
                self.mps.contract_end()
            avgt=(avgt*(self.point-2)+time()-t0)/(self.point-1)
            print("point:" +str(k+1)+' avg time:'+str(avgt)+' dkm:'+str(self.dkmax)+' pp:'+str(self.prec))
            #print('avg time: '+str(avgt))
        
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
     
    def convergence_scan(self,dkm_list,prec_list,ntot):
        self.convdat=[]
        for pp in prec_list:
            self.convdat.append([])
            for dk in dkm_list:
                self.convergence_params(self.dt,dk,pp)
                self.prep()
                self.prop(ntot)
                self.convdat[-1].append(self.statedat)
    
    def ttcorr(self,op1_array,op2_array,j1_int,j2_int):
            self.corrdat=[[],[],[self.dkmax,self.prec]]
            self.add_operator([[j1_int,op1_array]])
            self.checkdim(op2_array)
            self.prep()
            self.prop(j2_int)
            for jj in range(j1_int,j2_int+1):
                self.corrdat[0].append(self.statedat[0][jj]-j1_int*self.dt)
                self.corrdat[1].append(self.observe(op2_array,self.statedat[1][jj]) ) 
            self.statedat[0]=self.statedat[0][:j1_int]
            self.statedat[1]=self.statedat[1][:j1_int]
            return 0  
    
    def ttcorr_convergence_check(self,op1_array,op2_array,j1_int,dkm_list,prec_list,ntot):
        self.convergence_params(self.dt,dkm_list[-1],prec_list[-1])
        self.ttcorr(op1_array,op2_array,j1_int,ntot)
        self.convdat=[[self.statedat],[self.statedat]]
        self.corr_convdat=[[self.corrdat],[self.corrdat]]
        self.spec_convdat=[[self.getspectrum()],[self.getspectrum()]]
        for pp in prec_list[:-1]:
            self.convergence_params(self.dt,dkm_list[-1],pp)
            self.ttcorr(op1_array,op2_array,j1_int,ntot)
            self.convdat[0].append(self.statedat)
            self.corr_convdat[0].append(self.corrdat)
            self.spec_convdat[0].append(self.getspectrum())
        for kk in dkm_list[:-1]:
            self.convergence_params(self.dt,kk,prec_list[-1])
            self.ttcorr(op1_array,op2_array,j1_int,ntot)
            self.convdat[1].append(self.statedat)
            self.corr_convdat[1].append(self.corrdat)
            self.spec_convdat[1].append(self.getspectrum())
    
    def ttcorr_convergence_plot(self,op):
        subplot(231)
        for ppdat in self.convdat[0]:
            opdat=[]
            for rhvec in ppdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(ppdat[0],opdat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(234)
        for kkdat in self.convdat[1]:
            opdat=[]
            for rhvec in kkdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(kkdat[0],opdat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
        legend()
        
        subplot(232)
        for ppdat in self.corr_convdat[0]:
            redat=[]
            imdat=[]
            for datum in ppdat[1]:
                redat.append(datum.real)
                imdat.append(datum.imag)
            plot(ppdat[0],redat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(235)
        for kkdat in self.corr_convdat[1]:
            redat=[]
            imdat=[]
            for datum in kkdat[1]:
                redat.append(datum.real)
                imdat.append(datum.imag)
            plot(kkdat[0],redat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
        legend()
        
        subplot(233)
        for ppdat in self.spec_convdat[0]:
            redat=[]
            imdat=[]
            for datum in ppdat[1]:
                redat.append(datum.real)
                imdat.append(datum.imag)
            plot(ppdat[0],redat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(236)
        for kkdat in self.spec_convdat[1]:
            redat=[]
            imdat=[]
            for datum in kkdat[1]:
                redat.append(datum.real)
                imdat.append(datum.imag)
            plot(kkdat[0],redat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
        legend()
        
        show()
    
    def getspectrum(self):
        
        return [fftshift(fftfreq(len(self.corrdat[0]),self.dt)*2*3.1415926),fftshift(fft(self.corrdat[1]-self.corrdat[1][-1])),[self.dkmax,self.prec]]   
    
    def convergence_check(self,dkm_list,prec_list,ntot):
        self.convergence_params(self.dt,dkm_list[-1],prec_list[-1])
        self.prep()
        self.prop(ntot)
        self.convdat=[[self.statedat],[self.statedat]]
        for pp in prec_list[:-1]:
            self.convergence_params(self.dt,dkm_list[-1],pp)
            self.prep()
            self.prop(ntot)
            self.convdat[0].append(self.statedat)
        for kk in dkm_list[:-1]:
            self.convergence_params(self.dt,kk,prec_list[-1])
            self.prep()
            self.prop(ntot)
            self.convdat[1].append(self.statedat)
    
    def convergence_checkplot(self,op):
        subplot(221)
        for ppdat in self.convdat[0]:
            opdat=[]
            for rhvec in ppdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(ppdat[0],opdat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(222)
        for kkdat in self.convdat[1]:
            opdat=[]
            for rhvec in kkdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(kkdat[0],opdat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
        legend()
        
        subplot(224)
        for kkdat in self.convdat[1]:
            opdat=[]
            for rhvec in kkdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(kkdat[0],opdat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
        legend()
        show()
        
    def convergence_getdat(self,op):
        datgrid=[]
        for pplis in self.convdat:
            datgrid.append([])
            for dkdat in pplis:
                opdat=[]
                for rhvec in dkdat[1]:
                    opdat.append(self.observe(op,rhvec))
                datgrid[-1].append(opdat)
        return self.convdat[0][0][0], datgrid
    
    def convergence_plotdat(self,op):
        plts=[]
        for pplis in self.convdat:
            for dkdat in pplis:
                opdat=[]
                for rhvec in dkdat[1]:
                    opdat.append(self.observe(op,rhvec))
                pp,=plot(dkdat[0],opdat,label='dkm'+str(dkdat[2][0])+'pp'+str(dkdat[2][1]))
                plts.append(pp)
        legend(handles=plts)
        show()
                         
    def savesystem():
        
        return 0
    
    def savedata():
        return 0
    
    def getopdat(self,op):
        od=[]
        for da in self.statedat[1]:
            od.append(dot(op,da.reshape((self.dim,self.dim))).trace().real)
        return [self.statedat[0],od]
    
    def observe(self,op,rhovec):
        return dot(op,rhovec.reshape((self.dim,self.dim))).trace().real
    
    def plotopdat(self,op):
        od=[]
        for da in self.statedat[1]:
            od.append(dot(op,da.reshape((self.dim,self.dim))).trace().real)
        plot(self.statedat[0],od)
        show()
        return [self.statedat[0],od]
    
    def getcorrdat(self):
        return self.corrdat
    
    


'''
system=temposys(2)
system.set_hamiltonian(lambda t: array([[1,3],[6,8]]))
system.dt=1
print(system.ham(0))
print(shape(system.sysprop(2))  )
print(system.sysprop2(2)) 
print(system.sysprop(2))      

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
system.ham=lambda t: sigx*0
#initial state is excited
system.state=0.5*(idd+sigx)
#couples to sigma_z with single ohmic bath 
system.intparam.append([[1,-1],lambda t: eta_all(t,0.2,1.000001,7.5,0,0.5*0.01*10)])
system.lindbds.append([0.05*0,sigm])
system.lindbds.append([0.05*0,sigp])
#system.ops.append([150,idd])
#timestep, memory length and truncation precision
system.dt=0.15
system.dkmax=10
system.prec=0.000001
system.mod=20
#prep the system for propagation
system.prep()
#propagate the system 100 timesteps
system.prop(20)
#generate data for obseravble sigz and plot 
dat=system.opdat(sigx)
plt.plot(dat[0],dat[1])
plt.show()
'''
     
        
        
        
        