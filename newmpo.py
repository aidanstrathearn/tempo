import newquaPyVec as qp
from newquaPyVec import itab
import MpsMpo_site_level_operations as slo
import MpsMpo_block_level_operations as blo
from MpsMpo_block_level_operations import mps_block, mpo_block
from MpsMpo_site_level_operations import mps_site, mpo_site
import lineshapes as ln
import numpy as np
import pickle
import time
from numpy import linalg, zeros, identity, kron, array
import matplotlib.pyplot as plt
from math import fmod
from tensor_algebra import *
import os.path
from scipy.linalg import expm

def datload(filename):
    #function to unpickle data files and output them as a list
    f=open(filename, "rb")
    dlst=pickle.load(f,encoding='bytes')

    while 1:
        try:
            dlst.append(pickle.load(f))
        except (EOFError):
            break
    
    f.close()
    return dlst

def freeprop(ham,dt,op=[]):
    dim=len(ham)
    ham=array(ham)
    liou=-1j*(kron(ham.T,identity(dim)) - kron(identity(dim),ham.conj().T))
    '''
    for li in lind:
        liou=liou+li[0]*(kron(li[1].Tli[1].conj().T)
        -0.5*(kron(li[1].Tli[1].conj().T)))
    '''
    prop=expm(liou*dt)
    #function to create the freepropagator factor of the lambda tensor
    #for a given hamiltonian ham and timestep dt
    #define unitary time evol operator
    u=expm(ham*dt*(-1j))
    #take the kronecker product of u and its conjugate transpose
    #vectorized density matrix is made by stacking ROWS 
    #and we are propagating initial state from the RIGHT
    
    kprop=kron(u.T,u.conj().T)
    if len(op)==dim**2:
        print('using op')
        kprop=(np.dot(kprop,op))
         
    return kprop.T

def freeprop2(ham,dt,op=[]):
    dim=len(ham)
    ham=array(ham)
    liou=-1j*(kron(ham.T,identity(dim)) - kron(identity(dim),ham.conj().T))
    '''
    for li in lind:
        liou=liou+li[0]*(kron(li[1].Tli[1].conj().T)
        -0.5*(kron(li[1].Tli[1].conj().T)))
    '''

    #vectorized density matrix is made by stacking ROWS 
    #and we are propagating initial state from the RIGHT

    if len(op)==0:
        kprop=expm(liou*dt)
        
    else:
        print('using op')
        op=array(op)
        op=kron(op.T,identity(dim))
        kprop=np.dot(np.dot(expm(0.5*liou*dt),op),expm(0.5*liou*dt))
        
    return kprop.T


def sitetensor(eigl,dk,dkm,k,n,ham,dt,op=[]):
    #constructs the rank-4 tensors sites that make up the network
    
    #initialise rank-4 tensor of zeros
    l=len(eigl)
    tab=zeros((l**2,l**2,l**2,l**2),dtype=complex)
    #construct the bare rank-2 influence functional factor
    if dk==1:
        iffactor=itab(eigl,1,k,n,dkm)*qp.itab(eigl,0,k,n,dkm)*freeprop2(ham,dt,op)
    else:
        iffactor=qp.itab(eigl,dk,k,n,dkm)
    
    #loop through setting the non-zero elements of the rank-4 output tensor
    for i1 in range(l**2):
        for a1 in range(l**2):
            tab[i1][i1][a1][a1]=iffactor[i1][a1]    
    
    #if this is to be a site at the end of an mpo chain we  first sum over the east index
    #and replace with a dummy east index with dim=1
    if dk==dkm:
        return np.expand_dims(np.einsum('ijkl->ijk',tab),-1)
    else:        
        return tab

def tempo_mpoblock(eigl,ham,dt,dkm,k,n):
    #returns a TEMPO block of length dkm
    
    #initialise blank MPO block
    blk=mpo_block(0,0,0)
    #append dkm sites to the blank block
    for ii in range(1,dkm+1):
            blk.append_site(sitetensor(eigl,ii,dkm,k,n,ham,dt))
    return blk

def resumefunc(mod,eigl,ham,dt,ntot,dkm,datf,oplis):
    mpsfile=open('mps_'+datf,"rb")
    mps=pickle.load(mpsfile)           
    jj0=pickle.load(mpsfile)
    mpsl=mps.N_sites
    #build correct sized mpos to resume propagation
    if jj0<=dkm:
        propmpo=tempo_mpoblock(eigl,ham,dt,jj0,jj0,ntot)
    elif mod==1:
        propmpo=tempo_mpoblock(eigl,ham,dt,mpsl,jj0,ntot)
    else:
        propmpo=tempo_mpoblock(eigl,ham,dt,mpsl,mpsl+1,ntot)
        
    #import previous data
    datlis=datload(datf)
    #check we arent asking for data we already have
    if len(datlis)>=ntot:
        return print("data already collected up to ntot")
    #trim data back since there might already exist data for points beyond
    #where the last mps was saved
    datlis=datload(datf)[:jj0]
    for ell in oplis:
        if ell[0]<jj0:
            del ell
    #reopen previous data file and overwrite with trimmed data ready to append new data to
    datfile=open(datf,"wb")
    pickle.dump(datlis,datfile)
    print("resuming propagation")
    return mps, jj0, propmpo, datfile, datlis

def tempo(eigl,eta,irho,ham,dt,ntot,dkm,p,c=1,mod=0,oplis=[],datf=None,savemps=None):
    edge=np.expand_dims(np.eye(len(eigl)**2),1)
    t0=time.time()
    svds=['fraction','accuracy','chi']
    l=len(eigl)
    precision=10**(-0.1*p)
    qp.ctab=qp.mcoeffs(1,mod,eta,dkm,dt,ntot)
    rho=np.array(irho).reshape(l**2)
    datlis=[[0,rho]]
    if len(oplis)>0 and oplis[0][0]==0:
        oper=kron(oplis[0][1].T,identity(l))
        rho=np.dot(rho,oper)
        del oplis[0]    
    rho=np.dot(rho,freeprop2(ham,0.5*dt))*qp.itab(eigl,0,1,ntot,ntot)[0][:]
    rho=np.expand_dims(np.expand_dims(rho,-1),-1)
    mps=mps_block(0,0,0)
    mps.insert_site(0,rho)
    propmpo=tempo_mpoblock(eigl,ham,dt,1,1,ntot)
    jj0=1
    
    if type(datf)==str and type(savemps)==int:
        smps=savemps
    else:
        smps=ntot+1    
    if type(datf)==str:
        #check if data files and mps already exists
        if os.path.isfile(datf) and os.path.isfile('mps_'+datf):
            mpsfile=open('mps_'+datf,"rb")
            mps=pickle.load(mpsfile)           
            jj0=pickle.load(mpsfile)
            mpsl=mps.N_sites
            #build correct sized mpos to resume propagation
            if jj0<=dkm:
                propmpo=tempo_mpoblock(eigl,ham,dt,jj0,jj0,ntot)
            elif mod==1:
                propmpo=tempo_mpoblock(eigl,ham,dt,mpsl,jj0,ntot)
            else:
                propmpo=tempo_mpoblock(eigl,ham,dt,mpsl,mpsl+1,ntot)
                    
            #import previous data
            datlis=datload(datf)
            #check we arent asking for data we already have
            if len(datlis)>=ntot:
                return print("data already collected up to ntot")
            #trim data back since there might already exist data for points beyond
            #where the last mps was saved
            datlis=datload(datf)[:jj0]
            for ell in oplis:
                if ell[0]<jj0:
                    del ell
                    #reopen previous data file and overwrite with trimmed data ready to append new data to
            datfile=open(datf,"wb")
            pickle.dump(datlis,datfile)
            print("resuming propagation")
        else:
            #if either data or mps files dont exist then start fresh data file
            datfile=open(datf,"wb")
            pickle.dump(datlis,datfile)
          
    #iteratively apply MPO's to the MPS and readout/store data
    for jj in range(jj0,ntot+1):
        print("\npoint: "+str(jj)+" of "+str(ntot))
        ttt=time.time()
        
        rhoN=mps.readout2()
        datlis.append([jj*dt,np.dot(rhoN,freeprop(ham,0.5*dt))])
        
        if type(datf)==str:
            if fmod(jj,smps)==0:
                mpsfile=open('mps_'+datf,"wb")
                pickle.dump(mps,mpsfile)
                pickle.dump(jj,mpsfile)
                mpsfile.close()  
                
            pickle.dump([jj*dt,np.dot(rhoN,freeprop(ham,0.5*dt))],datfile)
            datfile.flush()
          
        if len(oplis)>0 and jj==oplis[0][0]:
            propmpo.data[0].update_site(tens_in=sitetensor(eigl,1,jj,jj,ntot+2,ham,dt,oplis[0][1]))
            del oplis[0]
        else:
            propmpo.data[0].update_site(tens_in=sitetensor(eigl,1,jj,jj,ntot+2,ham,dt))
        
        #contract with propagation mpo and insert the new end site, growing the MPS by one site
        mps.contract_with_mpo(propmpo,prec=precision,trunc_mode=svds[c])
        mps.insert_site(0,edge)
        
        if jj<dkm:
            #this is the growth stage: termination and propagation mpos each have their end sites
            #updated with new makri coefficients and then a new site appended
            propmpo.data[-1].update_site(tens_in=sitetensor(eigl,jj,jj+1,jj+1,ntot+1,ham,dt))
            propmpo.append_site(sitetensor(eigl,jj+1,jj+1,jj+1,ntot+1,ham,dt))
            #propmpo=tempo_mpoblock(eigl,ham,dt,jj,jj,ntot)
        elif jj==dkm:
            #if not using newquapi coefficients then only need to update end sites once at the dkm'th timestep
            propmpo.data[-1].update_site(tens_in=sitetensor(eigl,dkm,dkm,dkm+1,ntot+1,ham,dt))
            mps.contract_end()         
        else:
            #for jj>dkm without newquapi the propagation and termination are identical at every step so 
            #we only need to contract the end mps site
            mps.contract_end()   
              
        print("bond dims: "+str(mps.bonddims())+" total size: "+str(mps.totsize()))
        print("time: "+str(time.time()-ttt)+" prec: "+str(precision)+" length: "+str(mps.N_sites))
        
    print('\ntotal time: ' + str(time.time()-t0))
    #if type(datf)==str: datfile.close()
    return datlis 

hamil=[[0,0],
       [0,0]] 

eigs=[1,-1]

irho=[[0.5,0.5],[0.5,0.5]]

kk=7
pp=50
cc=12

sigp=[[0,1],[0,0]]
sigm=[[0,0],[1,0]]
idd=[[1,0],[0,1]]
oper2=kron(sigp,idd)
oper=kron(idd,idd)
daa=[]
def eta1(t):
    return ln.eta_all(t,0.2,1.00001,7.5,0,0.5*0.01*cc)

def eta1c(tc,t):
    if t<tc:
        return eta1(t)
    return eta1(tc)+(t-tc)*(eta1(tc+delt)-eta1(tc))/delt
           
dkmax=kk
                
delt=0.5
nt=150
#nt=int(np.ceil(2/delt))
#print(nt)
#qp.ctab=qp.mcoeffs(0,eta1,kk,delt,nt)
name="check2coup"+str(cc)+"_dkm"+str(dkmax)+"_prec"+str(pp)+".pickle"
daa.append(tempo(eigs,eta1,irho,hamil,delt,nt,dkmax,pp,oplis=[[60,sigp]],datf=name,savemps=10))
#daa.append(tempo(eigs,eta1,irho,hamil,delt,nt,dkmax,pp))

def dep(s,ti):
    return 2.7182818284590452353602874713527**(-4*eta1(ti).real-1j*4*(eta1(ti+s)-eta1(ti)-eta1(s)).imag).real

def depc(tc,s,ti):
    return (2.7182818284590452353602874713527**(-4*eta1c(tc,ti).real-1j*4*(eta1c(tc,ti+s)-eta1c(tc,ti)-eta1c(tc,s)).imag)).real

def dep2(s,ti):
    
    return 2.7182818284590452353602874713527**(-4*eta1(ti).real)


def dep2c(s,ti):
    return 2.7182818284590452353602874713527**(-4*eta1c(s,ti).real)

#print(dep(4))     
t=[]
d=[]
dd=[]
ddd=[]
tt=[]
#d2=[]
print(len(daa))
for jj in range(0,len(daa[0])-1):
    
    if jj>59:
        t.append(daa[0][jj][0])
        mult=np.dot(daa[0][jj][1],oper2.T)
        d.append(2*(mult[0].real+mult[3].real))
        #mult=np.dot(daa[1][jj][1],oper2.T)
        #dd.append((mult[0].real-mult[3].real))
        #mult=np.dot(daa[2][jj][1],oper2.T)
        #ddd.append((mult[0].real-mult[3].real))
        tt.append((jj)*delt)
        dd.append(depc(kk*delt,(60)*delt,daa[0][jj][0]-60*delt))
        #mult=np.dot(mult,oper2)
        #print(2*(mult[1].real))
        
        #mult=np.dot(oper,daa[1][jj][1])
        #mult=np.dot(oper2,mult)
        #d2.append(2*(mult[0]+mult[3]).real)
    else:
        mult=np.dot(oper2,daa[0][jj][1])
        #d.append(2*(mult[1].real))
        #dd.append(2*(mult[0]+mult[3]).conj())
        #tt.append(-daa[0][jj][0]+opp*delt)
        #mult=np.dot(oper2,daa[1][jj][1])
        #d2.append(2*(mult[0]+mult[3]).real)  

plt.plot(t,d)
#plt.plot(t,dd)
#plt.plot(t,ddd)
plt.plot(tt,dd)
axes = plt.gca()
#axes.set_xlim([-10,10])
#axes.set_ylim([0,1])
#plt.plot(t,d2)
plt.show()

'''   
xlis=[]
tlist=[]
for jj in range(len(daa)):
    xlis.append(daa[jj][0])
    tlist.append(2*np.dot(daa[jj][1],oper2.T)[0]+2*np.dot(daa[jj][1],oper2.T)[3])

plt.plot(xlis,tlist)
plt.show()
'''


'''
def tempocorr(oplis,eigl,eta,irho,ham,dt,ntot,dkm,p,c=1,mod=0,datf=None,savemps=None):
    edge=np.expand_dims(np.eye(len(eigl)**2),1)
    
    t0=time.time()
    svds=['fraction','accuracy','chi']
    l=len(eigl)
    precision=10**(-0.1*p)
    qp.trot=1
    qp.ctab=qp.mcoeffs(1,mod,eta,dkm,dt,ntot)
    
    rho=np.array(irho).reshape(l**2)
    datlis=[[0,rho]]
    rho=np.einsum('ij,i',freeprop(ham,dt),rho)*qp.itab(eigl,0,1,ntot,ntot)[0][:]
    datlis.append([dt,rho])
    rho=np.einsum('ijkl,j',sitetensor(eigl,1,1,1,ntot+2,ham,dt),rho)
    datlis.append([2*dt,np.einsum('ikl->k',rho)])
    #initialse blank MPS and create a single site from the initial state rho and the first 
    #influence functional factor I0
    mps=mps_block(0,0,0)
    mps.insert_site(0,rho)
    mps.insert_site(0,edge)
    jj0=2
    #initialise single site propagator mpo and termination propagator mpo 
    propmpo=tempo_mpoblock(eigl,ham,dt,2,2,ntot+450)
    #print(propmpo.data[0].m)
    #iteratively apply MPO's to the MPS and readout/store data
    for jj in range(jj0,ntot+1):
        print("\npoint: "+str(jj)+" of "+str(ntot))
        ttt=time.time()

        if jj==oplis[0]:
            propmpo.data[0].update_site(tens_in=sitetensor(eigl,1,jj,jj,ntot+2,ham,dt,oplis[1]))
        elif jj==oplis[0]+1:
            propmpo.data[0].update_site(tens_in=sitetensor(eigl,1,jj,jj,ntot+2,ham,dt))
        
        #contract with propagation mpo and insert the new end site, growing the MPS by one site
        mps.contract_with_mpo(propmpo,prec=precision,trunc_mode=svds[c])
        mps.insert_site(0,edge)
        
        dat=mps.readout2()
        datlis.append([(jj+1)*dt,dat])

        if jj<dkm:
            #this is the growth stage: termination and propagation mpos each have their end sites
            #updated with new makri coefficients and then a new site appended
            propmpo.data[-1].update_site(tens_in=sitetensor(eigl,jj,jj+1,jj+1,ntot+1,ham,dt))
            propmpo.append_site(sitetensor(eigl,jj+1,jj+1,jj+1,ntot+1,ham,dt))
            #propmpo=tempo_mpoblock(eigl,ham,dt,jj,jj,ntot)
        elif jj==dkm:
            #if not using newquapi coefficients then only need to update end sites once at the dkm'th timestep
            propmpo.data[-1].update_site(tens_in=sitetensor(eigl,dkm,dkm,dkm+1,ntot+1,ham,dt))
            mps.contract_end()         
        else:
            #for jj>dkm without newquapi the propagation and termination are identical at every step so 
            #we only need to contract the end mps site
            mps.contract_end()   
              
        print("bond dims: "+str(mps.bonddims())+" total size: "+str(mps.totsize()))
        print("time: "+str(time.time()-ttt)+" prec: "+str(precision)+" length: "+str(mps.N_sites))
        
    print('\ntotal time: ' + str(time.time()-t0))
    #if type(datf)==str: datfile.close()
    return datlis

def tempo(eigl,eta,irho,ham,dt,ntot,dkm,p,c=1,mod=0,datf=None,savemps=None):
    #implements state propogation using TEMPO
    #datf and mpsf are the filenames for the data and mps to be stored to
    #c is the truncation method and labels elemlents of svds list below
    #p is the appropriate truncation paramater that goes with the method c

    #define rank-3 tensor by giving a delt a dummy west index. this is used as the new end site
    #of the mps after each contraction with an mpo
    edge=np.expand_dims(np.eye(len(eigl)**2),1)
           
    #set some parameters and create table of makri coeffs qp.ctab from eta
    t0=time.time()
    svds=['fraction','accuracy','chi']
    l=len(eigl)
    precision=10**(-0.1*p)
    qp.trot=0
    qp.ctab=qp.mcoeffs(0,mod,eta,dkm,dt,ntot)
    
    #reshape initial matrix irho into vector rho and create data list
    rho=np.array(irho).reshape(l**2)
    datlis=[[0,rho]]
    
    #initialse blank MPS and create a single site from the initial state rho and the first 
    #influence functional factor I0
    mps=mps_block(0,0,0)
    mps.insert_site(0,np.expand_dims(np.expand_dims(rho*qp.itab(eigl,0,0,ntot,ntot)[0][:],-1),-1))
    jj0=1
    #initialise single site propagator mpo and termination propagator mpo 
    propmpo,termmpo=tempo_mpoblock(eigl,ham,dt,1,1,ntot),tempo_mpoblock(eigl,ham,dt,1,1,1)
    
    #set how regularly the mps is saved - only if also writing data to file
    if type(datf)==str and type(savemps)==int:
        smps=savemps
    else:
        smps=ntot+1
    
    if type(datf)==str:
        #check if data files and mps already exists
        if os.path.isfile(datf) and os.path.isfile('mps_'+datf):
            #extraxt mps and the timestep it is at
            mpsfile=open('mps_'+datf,"rb")
            mps=pickle.load(mpsfile)           
            jj0=pickle.load(mpsfile)
            mpsl=mps.N_sites
            #build correct sized mpos to resume propagation
            if jj0<=dkm:
                propmpo,termmpo=tempo_mpoblock(eigl,ham,dt,jj0,jj0,ntot),tempo_mpoblock(eigl,ham,dt,jj0,jj0,jj0)
            elif mod==1:
                propmpo,termmpo=tempo_mpoblock(eigl,ham,dt,mpsl,jj0,ntot),tempo_mpoblock(eigl,ham,dt,mpsl,jj0,jj0)
            else:
                propmpo,termmpo=tempo_mpoblock(eigl,ham,dt,mpsl,mpsl+1,ntot),tempo_mpoblock(eigl,ham,dt,mpsl,mpsl+1,mpsl+1)
            
            #import previous data
            datlis=datload(datf)
            #check we arent asking for data we already have
            if len(datlis)>=ntot:
                return print("data already collected up to ntot")
            #trim data back since there might already exist data for points beyond
            #where the last mps was saved
            datlis=datload(datf)[:jj0]
            #reopen previous data file and overwrite with trimmed data ready to append new data to
            datfile=open(datf,"wb")
            pickle.dump(datlis,datfile)
            print("resuming propagation")
         
        else:
            #if either data or mps files dont exist then start fresh data file
            datfile=open(datf,"wb")
            pickle.dump(datlis,datfile)

    
    #iteratively apply MPO's to the MPS and readout/store data
    for jj in range(jj0,ntot+1):
        print("\npoint: "+str(jj)+" of "+str(ntot))
        ttt=time.time()
        
        #readout physical density matrix and append to data list/save to file
        dat=mps.readout(termmpo)
        datlis.append([jj*dt,dat])
        
        if type(datf)==str:
            if fmod(jj,smps)==0:
                mpsfile=open('mps_'+datf,"wb")
                pickle.dump(mps,mpsfile)
                pickle.dump(jj,mpsfile)
                mpsfile.close()  
                
            pickle.dump([jj*dt,dat],datfile)
            datfile.flush()
        
        #contract with propagation mpo and insert the new end site, growing the MPS by one site
        mps.contract_with_mpo(propmpo,prec=precision,trunc_mode=svds[c])
        mps.insert_site(0,edge)
        
        if jj<dkm:
            #this is the growth stage: termination and propagation mpos each have their end sites
            #updated with new makri coefficients and then a new site appended
            termmpo.data[jj-1].update_site(tens_in=sitetensor(eigl,jj,jj+1,jj+1,jj+1,ham,dt))
            termmpo.append_site(sitetensor(eigl,jj+1,jj+1,jj+1,jj+1,ham,dt))
            propmpo.data[jj-1].update_site(tens_in=sitetensor(eigl,jj,jj+1,jj+1,ntot,ham,dt))
            propmpo.append_site(sitetensor(eigl,jj+1,jj+1,jj+1,ntot,ham,dt))
        elif mod==1:
            #beyond the growth stage and if we are using newquapi coefficients we
            #update the end (dkm'th) site each timestep with modified coefficients
            #but dont append a new site. Finally contract away the now unused end site of the mps
            #(opposite side to the side the new tensor "edge" is attached to above)         
            termmpo.data[dkm-1].update_site(tens_in=sitetensor(eigl,dkm,dkm,jj+1,jj+1,ham,dt))
            propmpo.data[dkm-1].update_site(tens_in=sitetensor(eigl,dkm,dkm,jj+1,ntot,ham,dt))
            mps.contract_end()
        elif jj==dkm:
            #if not using newquapi coefficients then only need to update end sites once at the dkm'th timestep
            termmpo.data[dkm-1].update_site(tens_in=sitetensor(eigl,dkm,dkm,dkm+1,dkm+1,ham,dt))
            propmpo.data[dkm-1].update_site(tens_in=sitetensor(eigl,dkm,dkm,dkm+1,ntot,ham,dt))
            mps.contract_end()         
        else:
            #for jj>dkm without newquapi the propagation and termination are identical at every step so 
            #we only need to contract the end mps site
            mps.contract_end()   
        
        print("bond dims: "+str(mps.bonddims())+" total size: "+str(mps.totsize()))
        print("time: "+str(time.time()-ttt)+" prec: "+str(precision)+" length: "+str(mps.N_sites))
        
    print('\ntotal time: ' + str(time.time()-t0))
    if type(datf)==str: datfile.close()
    return datlis 
'''





