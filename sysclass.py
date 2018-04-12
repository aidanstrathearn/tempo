#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 08:15:31 2018

@author: aidan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:56:29 2018

@author: aidan
"""
from numpy import ascontiguousarray,unique,array, expand_dims, kron, eye, dot, ones, outer, zeros, shape
from numpy.fft import fft, fftfreq, fftshift
import numpy as np
from matplotlib.pyplot import plot, show, legend, subplot, figure, imshow, colorbar
from pickle import dump, load
from time import time
from scipy.linalg import expm
from MpsMpo_block_level_operations import mps_block, mpo_block
from MpsMpo_site_level_operations import mpo_site

class temposys(object):
    '''
    def __new__(cls, filepath=None, *args, **kwargs):
        if filepath:
            with open(filepath,'rb') as f:
               inst = load(f,encoding='bytes')
            if not isinstance(inst, cls):
               raise TypeError('Unpickled object is not of type {}'.format(cls))
        else:
            inst = super(temposys, cls).__new__(cls, *args, **kwargs)
        #return inst
    '''
    def __init__(self,hilbert_dim=2,file=None):
        if type(file)==str:
            try: 
                loaded=load(open(file,'rb'),encoding='bytes')
                print(dir(self))
                self.statedat=[]
            except (FileNotFoundError): print('problem loading system from file, initialising new sys')
        else:
            #initialise system object with hilbert space dimension = hilbert_dim
            self.dim=hilbert_dim
        
            #self.ham=lambda t: zeros((self.dim**2,self.dim**2))    #the reduced system hamiltonian is now a function of time
            self.ham2= zeros((self.dim**2,self.dim**2))
            self.diss=zeros((self.dim**2,self.dim**2))                    #list of lindblads and rates: [[rate1,oper1],[rate2,oper2],...]
            self.intparam=[]                    #lists of interaction op eigenvalues and bath etas: [[eigs1,eta1],[eigs2,eta2],..]
            self.deg=[]
            self.nbaths=0
            self.ops=[]
            self.state=array(self.dim**2)           #reduced system density matrix
            self.istate=array(self.dim**2)           #reduced system density matrix
            self.dkmax=0                        #maximum length of the mps
            self.mpsdims=[[],[]]
            self.prec=0                         #precision used in svds
            self.ntot=1
        
            self.mod=0                          #maximum point to use new quapi coeffs up to
            self.point=0                        #current point in time system is at
            self.dt=0                           #size of timestep
            self.statedat=[[],[]]
            self.corrdat=[[],[]]                         #data list: [[time1,state1],[time2,state2],...]
            self.name='temp'
        
            self.mps=mps_block(0,0,0)           #blank mps block
            self.mpo=mpo_block(0,0,0)           #blank mpo block
       
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
    
    def set_hamiltonian(self,ham):
        self.checkdim(ham)
        self.ham2= -1j*(kron(ham,eye(self.dim)) - kron(eye(self.dim),ham.conj()))
    
    def add_dissipation(self,lind_list):
        for el in lind_list:
            self.checkdim(el[1])
            self.diss=self.diss+el[0]*(kron(el[1],el[1].conj())
                -0.5*(kron(dot(el[1].conj().T,el[1]),eye(self.dim))+kron(eye(self.dim),dot(el[1].conj().T,el[1]).T)))
    
    def add_bath(self,bath_list):
        for el in bath_list:
            self.checkdim(el[0])
            comm=kron(el[0].diagonal(),ones(self.dim))-kron(ones(self.dim),el[0].diagonal())
            acomm=kron(el[0].diagonal(),ones(self.dim))+kron(ones(self.dim),el[0].diagonal())
            if self.dt>0: 
                self.intparam.append([[comm,acomm],el[1],self.getcoeffs(el[1])])
            else: self.intparam.append([[comm,acomm],el[1],[]])
            self.nbaths=self.nbaths+1
    
    def find_degeneracy(self):
        clis=zeros((2*self.nbaths,self.dim**2))
        ii=0
        for el in self.intparam:
            clis[ii]=el[0][0]
            clis[ii+self.nbaths]=el[0][1]
            ii=ii+1
            
        b = ascontiguousarray(clis[:self.nbaths].T).view(np.dtype((np.void, (clis[:self.nbaths].T).dtype.itemsize * (clis[:self.nbaths].T).shape[1])))
        uh=unique(b,return_index=True,return_inverse=True)
        print('W/E degen: '+str(len(el[0][0]))+' to '+str(len(uh[0])))
        b = ascontiguousarray(clis.T).view(np.dtype((np.void, (clis.T).dtype.itemsize * (clis.T).shape[1])))
        uv=unique(b,return_index=True,return_inverse=True)
        print('N/S degen: '+str(len(el[0][0]))+' to '+str(len(uv[0])))
        self.deg=[[len(uh[0]),uh[1],uh[2]],[len(uv[0]),uv[1],uv[2]]]
            
    def add_operator(self,op_list):
        print('adding operator')
        for el in op_list:
            self.checkdim(el[1])
            self.ops.append([el[0],kron(el[1],eye(self.dim)).T])
            
    def add_operator2(self,op_list):
        print('adding operator')
        for el in op_list:
            self.checkdim(el[1][0])
            self.checkdim(el[1][1])
            self.ops.append([el[0],kron(el[1][0],el[1][1].T).T])
                 
    def convergence_params(self,dt_float,dkmax_int,truncprec_int):
        self.dkmax=dkmax_int
        self.prec=truncprec_int
        self.dt=dt_float
        for el in self.intparam: el[2]=self.getcoeffs(el[1])

    def getcoeffs(self,eta_function):
        #tb is list of values of eta evaulated at integer timestep
        tb=list(map(eta_function,array(range(self.dkmax+2))*self.dt))
        #etab takes finite differences on tb to create coeffs
        etab=[tb[1]]
        for jj in range(1,self.dkmax+1):
            etab.append(tb[jj+1]-2*tb[jj]+tb[jj-1])
            
        #etab=[tb[1]]+list(map(lambda jj: tb[jj+1]-2*tb[jj]+tb[jj-1],range(1,self.dkmax+1)))
        #if 
        #if self.mod>self.dkmax:
        #    tb=tb+list(map(eta_function,array(range(self.dkmax+2,self.mod+3))*self.dt))             
        #    etab=list(etab)+list(map(lambda jj: tb[jj+1]-tb[jj]-tb[self.dkmax]+tb[self.dkmax-1],range(self.dkmax+1,self.mod+2)))
        return etab
        
    def sysprop(self,j): 
        return expm((self.ham(j*self.dt)+self.diss)*self.dt/2).T

    def sysprop2(self,j): 
        return expm((self.ham2+self.diss)*self.dt/2).T            
                  
    def itab(self,dk):
        #function to store the influence function I_dk as a table
        #loop through all baths multiplyin in the corresponding matrix
        vec1=zeros(self.dim**2)
        vec2=zeros(self.dim**2)
        for el in self.intparam:
            #picking out the correct eta coeff
            vec1=vec1+el[0][0]
            eta_dk=el[2][dk+(self.point-dk)*int((self.mod and self.point)>self.dkmax and dk>1)]
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

        if dk==1:
            iffac=iffac*self.itab(0)
            if len(self.ops)>0 and self.ops[0][0]==self.point:
                print('using op')
                iffac=iffac*dot(self.sysprop2(self.point-1),dot(self.ops[0][1],self.sysprop2(self.point)))
                del self.ops[0]
            else:
                iffac=iffac*dot(self.sysprop2(self.point-1),self.sysprop2(self.point))
        
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
        self.state=dot(self.mps.readout3(),self.sysprop2(self.point-1))
        if len(self.ops)>0 and self.ops[0][0]==self.point: 
            self.state=dot(self.state,self.ops[0][1])
                         
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
                dot(self.state,self.sysprop2(0))*self.itab(0)
                                    ,-1),-1))
        self.getstate()
        
        #append first site to mpo to give a length=1 block
        self.mpo.append_mposite(self.temposite(1))
    
    def prop(self,kpoints=1,savemps=False):
        #propagates the system for ksteps - system must be prepped first
        for k in range(kpoints):
            
            t0=time()
            #find current time physical state and store this in self.dat
            #self.getstate()
            #print(self.statedat)
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
                #print(self.mpo.data[0].m.shape)
                self.mpo.append_mposite(self.temposite(self.point))
                #print(self.mpo.data[1].m.shape)
            elif (self.mod>self.dkmax) or (self.point==self.dkmax+1):
                #beyond the growth stage we either use new quapi and carry on updating end site
                #or update it just once at point=kmax+1 if using old quapi
                self.mpo.data[-1]=self.temposite(self.dkmax)
                self.mps.contract_end()       
            else:
                #if not using new quapi and point>kmax just contract away end site of mpo
                self.mps.contract_end()
            #avgt=(avgt*(self.point-2)+time()-t0)/(self.point-1)
            print("point:" +str(k+1)+' time:'+str(time()-t0)+' dkm:'+str(self.dkmax)+' pp:'+str(self.prec))
            print('abs point: '+str(self.point))
            self.mpsdims[0].append(time()-t0)
            self.mpsdims[1].append(self.mps.bonddims())
            print(self.mps.bonddims())
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
            dump(self.mpsdims,open(self.name+"_mpsdims_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
            
    def convergence_scan(self,dkm_list,prec_list,ntot):
        self.convdat=[]
        for pp in prec_list:
            self.convdat.append([])
            for dk in dkm_list:
                self.convergence_params(self.dt,dk,pp)
                self.prep()
                self.prop(ntot)
                self.convdat[-1].append(self.statedat)
       
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
        subplot(211)
        for ppdat in self.convdat[0]:
            opdat=[]
            for rhvec in ppdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(ppdat[0],opdat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(212)
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
                         
    def savemps(self):
        dump(self.mps,open(self.name+"_mps_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
        return 0

    def savesys(self):
        dump(self,open(self.name+"_sys_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
        
    def getopdat(self,op):
        od=[]
        for da in self.statedat[1]:
            od.append(dot(op,da.reshape((self.dim,self.dim))).trace().real)
        return [self.statedat[0],od]
    
    def observe(self,op,rhovec):
        return dot(op,rhovec.reshape((self.dim,self.dim))).trace()
    
    def plotopdat(self,op_lis):
        print('plotopdat')
        for el in op_lis:
            od=[]
            for da in self.statedat[1]:
                od.append(dot(el,da.reshape((self.dim,self.dim))).trace().real)
            #print(od)
            plot(self.statedat[0],od)
        show()
        return [self.statedat[0],od]
    
    def getcorrdat(self):
        return self.corrdat
    
    def bondplot(self):
        for el in self.mpsdims[1]:
            while len(el)<len(self.mpsdims[1][-1]):
                el.append(0)
        
        figure()
        imshow(self.mpsdims[1],extent=[0,self.point*self.dt,0,self.point*self.dt])#,vmin=60,vmax=90)
        colorbar()
        show()
    
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
        
    def ddcorr(self,a_array,adag_array,j1_int,j2_int):
            self.corrdat=[[],[],[self.dkmax,self.prec]]
            self.add_operator([[j1_int,a_array]])
            self.checkdim(a_array)
            self.prep()
            self.prop(j2_int)
            for jj in range(j1_int,j2_int+1):
                self.corrdat[0].append(self.statedat[0][jj]-j1_int*self.dt)
                self.corrdat[1].append(self.observe(a_array,self.statedat[1][jj]) ) 
            self.statedat[0]=self.statedat[0][:j1_int]
            self.statedat[1]=self.statedat[1][:j1_int]
            
            self.corrdat[1]=self.corrdat[1]/self.observe(dot(adag_array,a_array),self.statedat[1][-1])
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
            dump(self.corrdat,open(self.name+"_dddat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
            
            
            return 0
        
    def g1corr(self,a_array,adag_array,j1_int,j2_int):
            self.corrdat=[[],[],[self.dkmax,self.prec]]
            self.checkdim(a_array)
            self.checkdim(adag_array)
            self.add_operator2([[j1_int,[a_array,eye(self.dim)]]])
            self.prep()
            self.prop(j2_int)
            for jj in range(j1_int,j2_int+1):
                self.corrdat[0].append(self.statedat[0][jj]-j1_int*self.dt)
                self.corrdat[1].append(self.observe(adag_array,self.statedat[1][jj]) ) 
            self.statedat[0]=self.statedat[0][:j1_int]
            self.statedat[1]=self.statedat[1][:j1_int]
            
            self.corrdat[1]=self.corrdat[1]/self.observe(dot(adag_array,a_array),self.statedat[1][-1])
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
            dump(self.corrdat,open(self.name+"_g1dat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
            return 0
    
    def g2corr(self,a_array,adag_array,j1_int,j2_int):
            self.corrdat=[[],[],[self.dkmax,self.prec]]
            self.checkdim(a_array)
            self.checkdim(adag_array)
            self.add_operator2([[j1_int,[a_array,adag_array]]])
            self.prep()
            self.prop(j2_int)
            for jj in range(j1_int,j2_int+1):
                self.corrdat[0].append(self.statedat[0][jj]-j1_int*self.dt)
                self.corrdat[1].append(self.observe(dot(adag_array,a_array),self.statedat[1][jj]) ) 
            self.statedat[0]=self.statedat[0][:j1_int]
            self.statedat[1]=self.statedat[1][:j1_int]
            
            self.corrdat[1]=self.corrdat[1]/(self.observe(dot(adag_array,a_array),self.statedat[1][-1]))**2
            dump(self.statedat,open(self.name+"_statedat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
            dump(self.corrdat,open(self.name+"_g2dat_dkm"+str(self.dkmax)+"prec"+str(self.prec)+".pickle",'wb'))
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
    
    def g1_convergence_check(self,a_array,adag_array,j1_int,dkm_list,prec_list,ntot):
        self.convergence_params(self.dt,dkm_list[-1],prec_list[-1])
        self.g1corr(a_array,adag_array,j1_int,ntot)
        self.convdat=[[self.statedat],[self.statedat]]
        self.corr_convdat=[[self.corrdat],[self.corrdat]]
        for pp in prec_list[:-1]:
            self.convergence_params(self.dt,dkm_list[-1],pp)
            self.g1corr(a_array,adag_array,j1_int,ntot)
            self.convdat[0].append(self.statedat)
            self.corr_convdat[0].append(self.corrdat)
        for kk in dkm_list[:-1]:
            self.convergence_params(self.dt,kk,prec_list[-1])
            self.g1corr(a_array,adag_array,j1_int,ntot)
            self.convdat[1].append(self.statedat)
            self.corr_convdat[1].append(self.corrdat)
            
    def g2_convergence_check(self,a_array,adag_array,j1_int,dkm_list,prec_list,ntot):
        self.convergence_params(self.dt,dkm_list[-1],prec_list[-1])
        self.g2corr(a_array,adag_array,j1_int,ntot)
        self.convdat=[[self.statedat],[self.statedat]]
        self.corr_convdat=[[self.corrdat],[self.corrdat]]
        for pp in prec_list[:-1]:
            self.convergence_params(self.dt,dkm_list[-1],pp)
            self.g2corr(a_array,adag_array,j1_int,ntot)
            self.convdat[0].append(self.statedat)
            self.corr_convdat[0].append(self.corrdat)
        for kk in dkm_list[:-1]:
            self.convergence_params(self.dt,kk,prec_list[-1])
            self.g2corr(a_array,adag_array,j1_int,ntot)
            self.convdat[1].append(self.statedat)
            self.corr_convdat[1].append(self.corrdat)
            
    def g1_convergence_plot(self,op):
        subplot(221)
        for ppdat in self.convdat[0]:
            opdat=[]
            for rhvec in ppdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(ppdat[0],opdat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(223)
        for kkdat in self.convdat[1]:
            opdat=[]
            for rhvec in kkdat[1]:
                opdat.append(self.observe(op,rhvec))
            plot(kkdat[0],opdat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
        legend()
        
        subplot(222)
        for ppdat in self.corr_convdat[0]:
            redat=[]
            imdat=[]
            for datum in ppdat[1]:
                redat.append(datum.real)
                imdat.append(datum.imag)
            plot(ppdat[0],redat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
            #plot(ppdat[0],imdat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(224)
        for kkdat in self.corr_convdat[1]:
            redat=[]
            imdat=[]
            for datum in kkdat[1]:
                redat.append(datum.real)
                imdat.append(datum.imag)
            plot(kkdat[0],redat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
            #plot(kkdat[0],imdat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
        legend()
        
        show()
    
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
            #plot(ppdat[0],imdat,label='dkm'+str(ppdat[2][0])+'pp'+str(ppdat[2][1]))
        legend()
        
        subplot(235)
        for kkdat in self.corr_convdat[1]:
            redat=[]
            imdat=[]
            for datum in kkdat[1]:
                redat.append(datum.real)
                imdat.append(datum.imag)
            plot(kkdat[0],redat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
            #plot(kkdat[0],imdat,label='dkm'+str(kkdat[2][0])+'pp'+str(kkdat[2][1]))
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
def datload(filename):
    f=open(filename, "rb")
    dlist=load(f,encoding='bytes')
    '''
    t=0
    while t<1:
        try:
            dlist.append(load(f))
            t=1
        except (EOFError):
            break
        except (AttributeError):
            break
    '''
    f.close()
    return dlist

'''
for ssd in range(2,10):
    for pp in [40]:
        for cc in [10]:
            dat=datload('s'+str(ssd)+'_w50_c'+str(cc)+'0_t102_statedat_dkm'+str(pp)+'prec40.pickle')
            #bonddat=datload('s'+str(ssd)+'_w50_c'+str(cc)+'0_t34_bonddims_dkm200prec'+str(pp)+'.pickle')
            print(len(array(dat)[1]))
            dlis=[]
            cc=ssd
            sz=[ssd/2]
            while len(sz)<int(cc)+1: sz.append(sz[-1]-1)
            sz=kron(ones(int(cc)+1),sz)
            #sz=zeros((ssd+1)**2)
            #sz[1]=1
            for el in dat[1]:
                obv=0
                for jj in range(int(cc)+1):
                    obv=obv+sz[jj]*el[jj*cc]
                dlis.append(el[0].real)
            plot(dat[0],dlis)
show()
'''

'''
dlis=[[],[]]
for cc in [2,3,4,5,6,7,8,9]:
    dat=datload('s'+str(cc)+'_w50_c100_t102_statedat_dkm40prec40.pickle')
    dlis[0].append(cc)
    tot=0
    #for jj in dat:
    #    for kk in jj:
    #        
    #    tot=tot+j
    dlis[1].append(max(dat[-1]))
plot(dlis[0],dlis[1])
show()
'''
'''
dlis=[[],[]]
for cc in [25,50,75,100,125,150,175,200]:
    dat=datload('s1_w50_c50_t34_mpsdims_dkm'+str(cc)+'prec110.pickle')
    dlis[0].append(cc)
    tot=0
    #for jj in dat:
    #    for kk in jj:
    #        
    #    tot=tot+j
    dlis[1].append(max(dat[-1]))
plot(dlis[0],dlis[1])
dlis=[[],[]]
for cc in [25,50,75,100,125,150,175,200]:
    dat=datload('s1_w50_c100_t34_mpsdims_dkm'+str(cc)+'prec110.pickle')
    dlis[0].append(cc)
    tot=0
    #for jj in dat:
    #    for kk in jj:
    #        
    #    tot=tot+j
    dlis[1].append(max(dat[-1]))
    
plot(dlis[0],dlis[1])
show()
'''
'''
for el in self.mpsdims:
    while len(el)<len(self.mpsdims[-1]):
        el.append(0)
        figure()
        imshow(self.mpsdims,extent=[0,self.point*self.dt,0,self.point*self.dt])#,vmin=60,vmax=90)
        colorbar()
        show()
 '''   
#syss=datload('s1_w50_c10_t150_sys_dkm20prec50.pickle')
'''
#print('plotting')
#syss.plotopdat(array([[1,0],[0,-1]]))
#print(dat)
#vec2=array([4,3,2,1,0,-1,-2,-3,-4])
vec=array([1,0,-1])
ddifs=[]
for el in vec:
    ddifs=append(ddifs,el-vec)
vec2=array([2,0,-2])
on=ones(len(vec))
dif=kron(on,vec)-kron(vec,on)
dif2=kron(on,vec2)-kron(vec2,on)
summ=kron(on,vec)+kron(vec,on)
for el in unique(dif): print(el)
print('\n')
for el in dif: print(el)
print('\n')
for el in unique(dif):
    print(where(dif==el))

print(all(unique(dif,return_inverse=True)[1]==unique(dif2,return_inverse=True)[1]))
print(unique(dif2,return_inverse=True))
print(ddifs+dif)
     
vec1=[1,2,3,4]
vec2=[1,1,5,6]
vec3=[1,5,6]

mat1=zeros((4,4))
for jj in range(4):
    mat1[jj][jj]=vec1[jj]

#mat1=mat1[:]
mat2=array([vec1,vec1,vec1]).T
#for jj in range(2):
#    mat1[jj+2][jj]=vec1[jj]

mat1[0][:]=mat1[0][:]+mat1[1][:]
mat1=delete(mat1,1,0).T
print(mat1)
print('\n\n')
print(mat2)
'''
'''
v1=array([1,0,0])
v2=array([0,1,0])
v1=v2

cv1=kron(ones(len(v1)),v1)-kron(v1,ones(len(v1)))
cv2=kron(ones(len(v2)),v2)+kron(v2,ones(len(v2)))
print(cv1)
print(cv2)
'''
'''
cv2=kron(ones(4),v2)-kron(v2,ones(4))
av1=kron(ones(len(v1)),v1)+kron(v1,ones(len(v1)))
t1=array([v1])
ucv1=unique(cv1,return_index=True,return_inverse=True)
print(ucv1[2])
print(cv1[ucv1[1]])
uav1=unique(av1,return_index=True,return_inverse=True)
for el in cv1: print(el)
print('\n')
for el in av1: print(el)
print('\n')
b = ascontiguousarray(t1.T).view(np.dtype((np.void, (t1.T).dtype.itemsize * (t1.T).shape[1])))
ub=unique(b,return_index=True,return_inverse=True)
print(ub)
v1=[v1[i] for i in ub[1]]
print(v1)
'''
'''
ten=array([[1,2],[3,4,5]])
vec=array([6,7,8])
dot(ten[1],vec)
'''
