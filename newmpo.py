import newquaPyVec as qp
from newquaPyVec import itab, freeprop
import MpsMpo_site_level_operations as slo
import MpsMpo_block_level_operations as blo
from MpsMpo_block_level_operations import mps_block, mpo_block
from MpsMpo_site_level_operations import mps_site, mpo_site
import lineshapes as ln
import numpy as np
import pickle
import time
from numpy import linalg, zeros
import matplotlib.pyplot as plt
from math import fmod

def sitetensor(eigl,dk,dkm,k,n,ham,dt):
    #constructs the rank-4 tensors sites that make up the network
    
    #initialise rank-4 tensor of zeros
    l=len(eigl)
    tab=zeros((l**2,l**2,l**2,l**2),dtype=complex)
    
    #construct the bare rank-2 influence functional factor
    if dk==1:
        iffactor=itab(eigl,1,k,n,dkm)*qp.itab(eigl,0,k,n,dkm)*freeprop(ham,dt)
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

def tempo(eigl,eta,irho,ham,dt,ntot,dkm,p,c=1,mod=0,datf=None,mpsf=None):
    #implements state propogation using TEMPO
    #datf and mpsf are the filenames for the data and mps to be stored to
    #c is the truncation method and labels elemlents of svds list below
    #p is the appropriate truncation paramater that goes with the method c
    
    #set some parameters and create table of makri coeffs qp.ctab from eta
    t0=time.time()
    svds=['fraction','accuracy','chi']
    l=len(eigl)
    qp.trot=0
    qp.ctab=qp.mcoeffs(mod,eta,dkm,dt,ntot)
    
    #reshape initial matrix irho into vector rho and create data list
    rho=np.array(irho).reshape(l**2)
    datlis=[[0,rho]]
    
    #if saving the file then dump the initial data
    if type(datf)==str:
        datfile=open(datf,"wb")
        pickle.dump(datlis,datfile)
    
    #initialse blank MPS and create a single site from the initial state rho and the first 
    #influence functional factor I0
    mps=mps_block(0,0,0)
    mps.insert_site(0,np.expand_dims(np.expand_dims(rho*qp.itab(eigl,0,0,ntot,ntot)[0][:],-1),-1))
    
    #define rank-3 tensor by giving a delt a dummy west index. this is used as the new end site
    #of the mps after each contraction with an mpo
    edge=np.expand_dims(np.eye(len(eigl)**2),1)
    
    #initialise single site propagator mpo and termination propagator mpo 
    propmpo,termmpo=tempo_mpoblock(eigl,ham,dt,1,1,ntot),tempo_mpoblock(eigl,ham,dt,1,1,1)
    
    #iteratively apply MPO's to the MPS and readout/store data
    for jj in range(1,ntot+1):
        print("\npoint: "+str(jj))
        ttt=time.time()
        
        #readout physical density matrix and append to data list/save to file
        dat=mps.readout(termmpo)
        datlis.append([jj*dt,dat])
        if type(datf)==str: pickle.dump([jj*dt,dat],datfile)
        
        #contract with propagation mpo and insert the new end site, growing the MPS by one site
        mps.contract_with_mpo(propmpo,prec=p,trunc_mode=svds[c])
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
        
        #save mps to file every dkm+1 steps if valid filename mpsf is provided
        if fmod(jj,dkm+1)==0 and type(mpsf)==str:
            mpsfile=open(mpsf,"wb")
            pickle.dump(mps,mpsfile)
            mpsfile.close()        
        print("bond dims: "+str(mps.bonddims())+" total size: "+str(mps.totsize()))
        print("time: "+str(time.time()-ttt)+" prec: "+str(p)+" length: "+str(mps.N_sites))
    print(time.time()-t0)
    if type(datf)==str: datfile.close()
    return datlis 

hamil=[[0,1],[1,0]] 
eigs=[1,-1]
irho=[[1,0],[0,0]]

for cc in [45]:
    for mu in [50]:
        for kk in [20]:
            for pp in [30]:
                def eta1(t):
                    #return ln.sp3d_norm(0.000001,1,1)*ln.eta_sp_s3(t,1,1,0.01*mu,0.5*0.01*cc)/ln.sp3d_norm(0.01*mu,1,1)
                    #return ln.neta_sp_s3(t,1,1,0.01*mu,0.5*0.01*kk) #timestep 10/85
                    return ln.eta_sp_s1(t,0.2,7.5,0.001,0.5*0.01*10) 
                
                dkmax=kk
                nt=30
                delt=0.1
                daa=tempo(eigs,eta1,irho,hamil,delt,nt,dkmax,10**(-3))

#daa2=qp.quapi(modc,eigs,eta,6,hamil,0.1,irho,18,"_dk")

#######################################################################################################
#######################################################################################################
 #timestep=1/15 prec=70 for mccutcheon11
'''
qp.ctab=qp.mcoeffs(modc,eta,dkmax,delt,nt) 
block=tempo_mpoblock(eigs,hamil,delt,dkmax,dkmax+1,nt)
block.contract_with_mpo(tempo_mpoblock(eigs,hamil,delt,dkmax,dkmax+1,nt))
for j in range(dkmax):
    print(block.data[j].m.shape)
block.contract_with_mpo(tempo_mpoblock(eigs,hamil,delt,dkmax,dkmax+1,nt))
for j in range(dkmax):
    print(block.data[j].m.shape)
#tempo_mpoblock(eigs,hamil,delt,dkmax,dkmax+1,nt).contract_with_mpo(tempo_mpoblock(eigs,hamil,delt,dkmax,dkmax+1,nt))

def sitetensor(eigl,dk,dkm,k,n,ham,dt):
    l=len(eigl)
    tab=zeros((l**2,l**2,l**2,l**2),dtype=complex)
    if dk>dkm:
        for i1 in range(l**2):
            for j1 in range(l**2):
                for b1 in range(l**2):
                    for a1 in range(l**2):
                        if j1==i1 and a1==b1:
                            tab[i1][j1][b1][a1]=1
                            
    elif dk==1:
        tens=itab(eigl,1,k,n,dkm)*qp.itab(eigl,0,k,n,dkm)*freeprop(ham,dt)
        for i1 in range(l**2):
            for j1 in range(l**2):
                for b1 in range(l**2):
                    for a1 in range(l**2):
                        if j1==i1 and a1==b1:
                            tab[i1][j1][b1][a1]=tens[j1][a1]       
    
    else:
        for i1 in range(l**2):
            for j1 in range(l**2):
                for b1 in range(l**2):
                    for a1 in range(l**2):
                        if j1==i1 and a1==b1:
                            tab[i1][j1][b1][a1]=qp.itab(eigl,dk,k,n,dkm)[j1][a1]
        
    return tab
    
def tempo_mpoblock2(eigl,ham,dt,dkm,k,n):
    #dimension of hilbert space and initiate block - just a single site the rest are appended
    l=len(eigl)
    blk=mpo_block(l**2,l**2,1)
    
    blk.data[0]=mpo_site(tens_in=sitetensor(eigl,1,dkm,k,n,ham,dt))
    for ii in range(2,dkm):
            blk.append_site(sitetensor(eigl,ii,dkm,k,n,ham,dt))
    
    tens=sitetensor(eigl,dkm,dkm,k,n,ham,dt)
    blk.append_site(np.expand_dims(np.einsum('ijkl->ijk',tens),-1))
    return blk

def initalg(mod,eigl,eta,ham,dt,irho,ntot,c,p,dkm):
    t0=time.time()
    svds=['fraction','accuracy','chi']
    l=len(eigl)
    
    rho=np.array(irho).reshape(l**2)
    datlis=[[0,rho]]
    rho=rho*qp.itab(eigl,0,0,ntot,ntot)[0][:]
    
    rho1=np.einsum('i,jikl',rho,sitetensor(eigl,1,1,1,1,ham,dt))
    datlis.append([dt,np.einsum('ijk->j',rho1)])
    
    rho=np.einsum('i,jikl',rho,sitetensor(eigl,1,1,1,ntot,ham,dt))

    mps=mps_block(0,0,0)
    print(mps.N_sites)
    mps.insert_site(0,rho)  
    mps.insert_site(0,edgetens(eigl))

    for jj in range(2,dkm+1):
        ttt=time.time()
        datlis.append([jj*dt,mps.readout(tempo_mpoblock(eigl,ham,dt,jj,jj,jj))])
        
        size=0
        for ss in range(mps.N_sites):
           size=mps.data[ss].m.shape[0]*mps.data[ss].m.shape[1]*mps.data[ss].m.shape[2]+size
        bond=[]                
        for ss in range(mps.N_sites):
           bond.append(mps.data[ss].m.shape[2])        
        print("\n bond dims: "+str(bond))
        print("total size: "+str(size))
        print("prec: "+str(p))
                         
        mps.contract_with_mpo(tempo_mpoblock(eigl,ham,dt,jj,jj,ntot),prec=p,trunc_mode=svds[c])
        mps.insert_site(0,edgetens(eigl))

        print("point: "+str(jj)+" time: "+str(time.time()-ttt))
            
    print(time.time()-t0)
    return mps, datlis 

def tempoalg(mod,eigl,dkm,eta,ham,dt,irho,ntot,c,p,filename):
    
    svds=['fraction','accuracy','chi']
    
    mps,daa=initalg(mod,eigl,eta,ham,dt,irho,ntot,c,p,dkm)
    
    datfile=open(filename,"wb")
    pickle.dump(daa,datfile)
    
    
    prop_mpo, term_mpo = tempo_mpoblock(eigl,ham,dt,dkm,dkm+1,ntot), tempo_mpoblock(eigl,ham,dt,dkm,dkm+1,dkm+1)
    
    t0=time.time()
    
    for kk in range(dkm+1,ntot+1):
        ttt=time.time()
        if mod==1:
            prop_mpo, term_mpo = tempo_mpoblock(eigl,ham,dt,dkm,kk,ntot), tempo_mpoblock(eigl,ham,dt,dkm,kk,kk)
            print('mpo build time: ',time.time()-ttt)
        
        mps.contract_end()
        size=0
        for ss in range(mps.N_sites):
           size=mps.data[ss].m.shape[0]*mps.data[ss].m.shape[1]*mps.data[ss].m.shape[2]+size
        
        bond=[]
        for ss in range(mps.N_sites):
           bond.append(mps.data[ss].m.shape[2])
            
        print("\n bond dims: "+str(bond))
        print("total size: "+str(size))
        print("prec: "+str(p))

        daa.append([kk*dt,mps.readout(term_mpo)])
        pickle.dump([kk*dt,mps.readout(term_mpo)],datfile)
        
        #if fmod(kk,dkm+1)==0:
        #    mpsfile=open("mps_"+filename,"wb")
        #    pickle.dump(mps,mpsfile)
        #    mpsfile.close()
        mps.contract_with_mpo(prop_mpo,prec=p,trunc_mode=svds[c])
        mps.insert_site(0,edgetens(eigl))

        print("point: "+str(kk)+" time: "+str(time.time()-ttt))
        print("length: "+str(dkm))
        
    #datfile.close()
    del mps
    #print(prop_mpo.data[dkm-1].m)
    print(time.time()-t0)
    print("FINISHED")
    return daa
   

hamil=[[0,1],[1,0]] 
eigs=[1,-1]
irho=[[1,0],[0,0]]
modc=0

qp.trot=0

dlist=[]
for cc in [45]:
    for mu in [50]:
        for kk in [12]:
            for pp in [30]:
                def eta(t):
                    #return ln.sp3d_norm(0.000001,1,1)*ln.eta_sp_s3(t,1,1,0.01*mu,0.5*0.01*cc)/ln.sp3d_norm(0.01*mu,1,1)
                    #return ln.neta_sp_s3(t,1,1,0.01*mu,0.5*0.01*kk) #timestep 10/85
                    return ln.eta_sp_s1(t,0.0001,1,0.01*10,0.5*0.01*10) 
                dkmax=kk
                nt=24
                delt=0.1
                qp.ctab=qp.mcoeffs(modc,eta,dkmax,delt,nt)
                daa=tempoalg(modc,eigs,dkmax,eta,hamil,delt,irho,nt,1,10**(-3),'mu'+str(mu)+'_1dspatial_coup'+str(cc)+'_dkm'+str(kk)+'_prec'+str(pp)+'.pickle')
                dlist.append(daa)
'''









