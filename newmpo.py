import newquaPyVec as qp
from newquaPyVec import itab, freeprop
import MpsMpo_site_level_operations as slo
import MpsMpo_block_level_operations as blo
from MpsMpo_block_level_operations import mps_block, mpo_block
from MpsMpo_site_level_operations import mps_site, mpo_site
import lineshapes as ln
import numpy as np
import pickle
import copy
import time
from numpy import linalg, zeros
import scipy.sparse.linalg as ssl
from math import sqrt, ceil, floor, log, fmod
import matplotlib.pyplot as plt

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

def edgetens(eigl):
    return np.expand_dims(np.eye(len(eigl)**2),1)




def tempo_mpoblock(eigl,ham,dt,dkm,k,n):
    #dimension of hilbert space
    l=len(eigl)

    #initiate block - just a single site the rest are appended
    blk=mpo_block(l**2,l**2,1)
    blk.data[0]=mpo_site(tens_in=sitetensor(eigl,1,dkm,k,n,ham,dt))
    for ii in range(2,dkm):
            blk.append_site(sitetensor(eigl,ii,dkm,k,n,ham,dt))
    
    tens=sitetensor(eigl,dkm,dkm,k,n,ham,dt)
    blk.append_site(np.expand_dims(np.einsum('ijkl->ijk',tens),-1))

    return blk

def read(mps,mpoterm):
    l=len(mpoterm.data)
    out=np.einsum('ijk,limn',mps.data[l-1].m,mpoterm.data[l-1].m)
    out=np.einsum('ijklm->ijlm',out)
    out=np.einsum('ijkl->ik',out)
    for jj in range(l-1):
        nout=np.einsum('ijk,limn',mps.data[l-2-jj].m,mpoterm.data[l-2-jj].m)
        nout=np.einsum('ijklm->ijlm',nout)
        out=np.einsum('imkn,mn',nout,out)
    
    out=np.einsum('ij->j',out)
    
    return out


def initalg(mod,eigl,eta,ham,dt,irho,ntot,c,p,dkm):
    
    t0=time.time()
    svds=['fraction','accuracy','chi']
    l=len(eigl)
    
    rho=np.array(irho).reshape(l**2)
    datlis=[[0,rho]]
    rho=rho*qp.itab(eigl,0,0,ntot,ntot)[0][:]
    
    rho1=np.einsum('i,jikl',rho,sitetensor(eigl,1,1,1,1,ham,dt))
    datlis.append([dt,np.einsum('ijk->k',rho1)])
    
    rho=np.einsum('i,jikl',rho,sitetensor(eigl,1,1,1,ntot,ham,dt))
    rho=np.einsum('ijk->ij',rho)
    rho=np.expand_dims(rho,-1)

    mps=mps_block(0,0,0)
    mps.insert_site(0,rho)

    mps.insert_site(0,edgetens(eigl))
    #datlis.append([dt,mps.readout()])
    
    for jj in range(2,dkm+1):
        ttt=time.time()
        #mpsN=copy.deepcopy(mps)
        #mpsN.contract_with_mpo(tempo_mpoblock(eigl,ham,dt,jj,jj,jj),prec=p,trunc_mode=svds[c])
        #mpsN.insert_site(0,edgetens(eigl))
        datlis.append([jj*dt,read(mps,tempo_mpoblock(eigl,ham,dt,jj,jj,jj))])
        #del mpsN
        
        bond=[]
        
        size=0
        for ss in range(mps.N_sites):
           size=mps.data[ss].m.shape[0]*mps.data[ss].m.shape[1]*mps.data[ss].m.shape[2]+size
                        
        for ss in range(mps.N_sites):
           bond.append(mps.data[ss].m.shape[2])
         
        print("\n bond dims: "+str(bond))
        print("total size: "+str(size))
        print("prec: "+str(p))
                         
        mps.contract_with_mpo(tempo_mpoblock(eigl,ham,dt,jj,jj,ntot),prec=p,trunc_mode=svds[c])
        mps.insert_site(0,edgetens(eigl))
        #datlis.append([jj*dt,mps.readout()])
        #print([jj*dt,mps.readout()])
        print("point: "+str(jj)+" time: "+str(time.time()-ttt))
        print()
        
    
    
        
    print(time.time()-t0)
    return mps, datlis 

def tempoalg(mod,eigl,dkm,eta,ham,dt,irho,ntot,c,p,filename):
    
    svds=['fraction','accuracy','chi']
    
    mps,daa=initalg(mod,eigl,eta,ham,dt,irho,ntot,c,p,dkm)
    
    datfile=open(filename,"wb")
    pickle.dump(daa,datfile)
    
    prop_mpo, term_mpo = tempo_mpoblock(eigl,ham,dt,dkm,dkm+1,ntot), tempo_mpoblock(eigl,ham,dt,dkm,dkm+1,dkm+1)
    
    t0=time.time()
    #for jj in range(term_mpo.N_sites):
    #    term_mpo.data[jj].m=np.einsum('ijkl->jkl',term_mpo.data[jj].m)
    
    
    for kk in range(dkm+1,ntot+1):
        ttt=time.time()
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
        #mpsN=copy.deepcopy(mps)
        #mpsN.contract_with_mpo(term_mpo,prec=p,trunc_mode=svds[c])
        #mpsN.insert_site(0,edgetens(eigl))
        #daa.append([kk*dt,read(mps,term_mpo)])
        pickle.dump([kk*dt,read(mps,term_mpo)],datfile)
        
        if fmod(kk,dkm+1)==0:
            mpsfile=open("mps_"+filename,"wb")
            pickle.dump(mps,mpsfile)
            mpsfile.close()
        mps.contract_with_mpo(prop_mpo,prec=p,trunc_mode=svds[c])
        mps.insert_site(0,edgetens(eigl))
        #daa.append([kk*dt,mps.readout()])
        print("point: "+str(kk)+" time: "+str(time.time()-ttt))
        print("length: "+str(dkm))
        
    datfile.close()
    del mps

    print(time.time()-t0)
    print("FINISHED")
    return daa
'''
ep=0
hamil=[[ep,1],[1,-ep]]
eigs=[1,-1]
nsteps=15
hdim=len(eigs)
irho=[[1,0],[0,0]]
meth=1
vals=1
modc=0
#defining local and operator dimensions


qp.trot=0
nt=200

def eta(t):
    return ln.eta_0T_s1(t,10,0.5*1.2)


dlist=[]
for kk in range(11,12):
    for jj in range(6,7):
        dkmax=5*kk
        delt=2/(7*10)
        qp.ctab=qp.mcoeffs(modc,eta,dkmax,delt,nt) 
        dlist.append(tempoalg(modc,eigs,dkmax,eta,hamil,delt,irho,nt,meth,10**(-jj)))


#daa2=qp.quapi(modc,eigs,eta,dkmax,hamil,delt,irho,nt,"_dk")

#for jj in range(len(daa)):
 #   print(daa[jj][1]-daa2[jj][1])
    


x1=[]
x2=[]
x3=[]
x4=[]
#x5=[]
#x6=[]
y1=[]
y2=[]
y3=[]
y4=[]
for xi in range(nt):
    x1.append((dlist[0][xi][1][0]-dlist[0][xi][1][3]).real)
    x2.append((dlist[1][xi][1][0]-dlist[1][xi][1][3]).real)
    #x3.append((dlist[2][xi][1][0]-dlist[2][xi][1][3]).real)
    #x4.append(dlist[3][xi][1][0].real)
    #x5.append(dlist[4][xi][1][0].real)
    #x6.append(dlist[5][xi][1][0].real)
    y1.append(dlist[0][xi][0])
    y2.append(dlist[1][xi][0])
    #y3.append(dlist[2][xi][0])
    #y4.append(dlist[3][xi][0])
    

plt.plot(y1,x1)
plt.plot(y2,x2)
#plt.plot(y3,x3)
#plt.plot(y4,x4)
#plt.plot(y,x5)
#plt.plot(y,x6)
plt.ylim([0,1])
plt.show()
'''












