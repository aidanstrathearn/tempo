import newquaPyVec as qp
import definitions as df
import lineshapes as ln
import numpy as np
import block_multiplication as bm
import pickle

#defininf the lineshape used to find the makri coeffs
def eta(t):
    return ln.eta_0T(t,3,1,1)
    
#some test parameters
hamil=[[0,1],[1,0]]
eigs=[-1,1]
dkmax=3
delt=0.1
nsteps=20
hdim=len(eigs)

#defining local and operator dimensions
df.local_dim=hdim**2
df.opdim=hdim**4


#function to create a block of mpo sites using the class in definitions.py
def create_block(eigl,eta,ham,dt,dkm,k,n,ntot):
    #dimension of hilbert space
    l=len(eigl)
    qp.ctab=qp.mcoeffs(0,eta,dkm,dt,ntot)
    #initialising the block
    blk=df.mpo_block(l,dkm)
    #calculating makricoeffs and storing in ctab
    #loops each site setting equal to the tensors created with newquaPyVec.py
    if k>dkm-1:
        for ii in range(dkm):
            if ii==0:
                blk[ii].m=qp.mpostartsite(eigl,dkm,k,n,ham,dt)
            elif ii==dkm-1:
                blk[ii].m=qp.mpoendsite(eigl,dkm,dkm,k,n)
            else:
                blk[ii].m=qp.mpomidsite(eigl,ii+1,dkm,k,n)
    elif k==1:
        for ii in range(dkm):
            if ii==0:
                blk[ii].m=qp.gr_mpostartsite(eigl,dkm,k,n,ham,dt)
            elif ii==dkm-1:
                blk[ii].m=qp.gr_mpodummyedge(eigl)
            else:
                blk[ii].m=qp.gr_mpodummymid(eigl)
    else:
        for ii in range(dkm):
            if ii==0:
                blk[ii].m=qp.mpostartsite(eigl,dkm,k,n,ham,dt)
            elif ii==k-1:
                blk[ii].m=qp.gr_mpoendsite(eigl,k,dkm,k,n)
            elif ii==dkm-1:
                blk[ii].m=qp.gr_mpodummyedge(eigl)
            elif ii<k-1:
                blk[ii].m=qp.mpomidsite(eigl,ii+1,dkm,k,n)
            else:
                blk[ii].m=qp.gr_mpodummymid(eigl)
                
        
    return blk

#set up the dummy sites for the initial state mps
def init_mps(eigl,rho,dkm):
    l=len(eigl)
    mpsblk=df.mps_block(l**2,dkm)
    for ii in range(dkm):
        if ii==0:
            mpsblk[ii].m=qp.mpsrho(eigl,rho)
        elif ii==dkm-1:
            mpsblk[ii].m=qp.mpsdummyend(eigl)
        else:
            mpsblk[ii].m=qp.mpsdummymid(eigl)
    return mpsblk

#function to go through contracting the sites in a single block together using einsum
#to produce the full lambda matrix
def block_contract(block):
    dkk=len(block)
    init=np.einsum('ijkl,mnlo',block[0].m,block[1].m)
    for jj in range(2,dkk):
        init=np.einsum('...i,jkil',init,block[jj].m)
    init=np.sum(np.sum(init,-1),2)
    return init

#contracts all sites in a block to give the full augmented density tensor
def mps_contract(block):
    dkk=len(block)
    init=np.einsum('ijk,lkm',block[0].m,block[1].m)
    for jj in range(2,dkk):
        init=np.einsum('...i,jik',init,block[jj].m)
    #init=np.sum(np.sum(init,-1),2)
    init=np.sum(np.sum(init,-1),1)
    return init
 

#creating 5 mpos to propagate the system 5 steps forward with kmax=3
#block_contract contracts all the west/east indices to form one big tensor 
tblk1=create_block(eigs,eta,hamil,delt,3,1,5,5)
tblk1=block_contract(tblk1)
tblk2=create_block(eigs,eta,hamil,delt,3,2,5,5)
tblk2=block_contract(tblk2)
tblk3=create_block(eigs,eta,hamil,delt,3,3,5,5)
tblk3=block_contract(tblk3)
tblk4=create_block(eigs,eta,hamil,delt,3,4,5,5)
tblk4=block_contract(tblk4)
tblk5=create_block(eigs,eta,hamil,delt,3,5,5,5)
tblk5=block_contract(tblk5)

#set up initial mps and contract east west indices
mp=mps_contract(init_mps([-1,1],[1,0,0,0],3))
print mp.shape
#propagate the whole block one step at a time
mp=np.einsum('ijk,iljmkn',mp,tblk1)
mp=np.einsum('ijk,iljmkn',mp,tblk2)
mp=np.einsum('ijk,iljmkn',mp,tblk3)
mp=np.einsum('ijk,iljmkn',mp,tblk4)
mp=np.einsum('ijk,iljmkn',mp,tblk5)
#contract indices for readout
mp=np.einsum('ijk->i',mp)

#display density matrix
print mp

#same propagation with same parameters but using QUAPi, agrees perfect
#it outputs all the data put the data point at time 0.5 is the relevent one
qp.quapi(0,eigs,eta,3,hamil,0.1,[1,0,0,0],5,"tempocheck")
f=open("tempocheck3.pickle")
myf=pickle.load(f)
print myf

'''
mstart=qp.gr_mpostartsite2(eigs,1,1,2,hamil,delt)
print mstart.shape
mstart=np.sum(np.sum(mstart,-1),-1)
print tblk1-qp.initprop(eigs,1,2,hamil,delt)
print mstart.shape
'''

'''tblk=np.einsum('ijklmn->ijlm',block_contract(tblk))
mp[0].m=np.einsum('ijk->ik',mp[0].m)
mp[1].m=np.einsum('ijk->ij',mp[1].m)
mp2=np.einsum('ij,kj',mp[0].m,mp[1].m)
mp2=np.einsum('ijkl,ik',tblk,mp2)
mp2=np.einsum('ij->i',mp2)
print mp2'''
#lam_mat should now be the full rank-2dk propagator - (actually rank-(2dk+2).. the further 2 ranks are 'null'
#west/east legs of the start/end sites respectively)
'''lam_mat=block_contract(tblk)

lam_mat=sum(sum(lam_mat,2),-1)
lam_mat=np.swapaxes(lam_mat,0,-1)
lam_mat=np.reshape(lam_mat,(16,16))
#print np.linalg.eig(np.reshape(lam_mat,(16,16)))[0]
print lam_mat.shape


lam_mat=np.swapaxes(np.swapaxes(np.swapaxes(np.swapaxes(sum(sum(lam_mat,2),-1),0,1),1,3),2,5),4,5)
lam_mat=np.swapaxes(np.swapaxes(lam_mat,0,2),3,5)
lam_mat=np.reshape(lam_mat,(4**3,4**3))


eigenvals= np.linalg.eig(lam_mat)
vec=eigenvals[1].T[0]
tol = 1e-16
vec.real[abs(vec.real) < tol] = 0.0
vec.imag[abs(vec.imag) < tol] = 0.0
print vec
#nor=np.dot(vec,vec.conj())
#print vec/nor

#inner two sums below contract over the null indices while the others contract over the additional dk-1 legs that 
#the lambda propagator as defined in standard quapi doesnt have - this is in order to compare it to the lambda tensor
#that is built in newquaPyVec.py... finally arrange the remaining indices into the same order as the lamtens from quapy
#and comparing the resulting tensor with one built the standard quapi way.. perfect match!
#note that the sums and index swapping done here is specific to kmax=3

#lam_mat=np.swapaxes(np.swapaxes(np.swapaxes(sum(sum(sum(sum(lam_mat,2),-1),2),0),2,3),0,3),1,2)
#print lam_mat-qp.lamtens(eigs,3,4,5,hamil,delt)'''


