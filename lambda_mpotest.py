import newquaPyVec as qp
import definitions as df
import lineshapes as ln
import numpy as np
import block_multiplication as bm
import pickle
import copy
#defininf the lineshape used to find the makri coeffs
def eta(t):
    return ln.eta_0T(t,3,1,0.5)
    
#some test parameters
hamil=[[0,1],[1,0]]
eigs=[-1,1]
delt=0.1
nsteps=20
hdim=len(eigs)
irho=[1,0,0,0]

#defining local and operator dimensions
df.local_dim=hdim**2

def create_block(eigl,eta,ham,dt,dkm,k,n,ntot):
    #dimension of hilbert space
    l=len(eigl)
    qp.ctab=qp.mcoeffs(0,eta,dkm,dt,ntot)
    #initialising the block
    
    #calculating makricoeffs and storing in ctab
    #loops each site setting equal to the tensors created with newquaPyVec.py
    if k>dkm-1:
        blk=df.mpo_block(l**4,dkm)
        for ii in range(dkm):
            if ii==0:
                blk[ii].m=qp.mpostartsite(eigl,dkm,k,n,ham,dt)
            elif ii==dkm-1:
                blk[ii].m=qp.mpoendsite(eigl,dkm,dkm,k,n)
            else:
                blk[ii].m=qp.mpomidsite(eigl,ii+1,dkm,k,n)
    elif k==1:
        blk=df.mpo_block(l**4,k)
        blk[0].m=qp.gr_mpostartsite(eigl,dkm,k,n,ham,dt)
        blk[0].Sdim=l**2
        blk[0].Ndim=l**2
        blk[0].Wdim=1
        blk[0].Edim=l**2

    else:
        blk=df.mpo_block(l**4,k)
        for ii in range(k):
            if ii==0:
                blk[ii].m=qp.mpostartsite(eigl,dkm,k,n,ham,dt)
            elif ii==k-1:
                blk[ii].m=qp.gr_mpoendsite(eigl,k,dkm,k,n)
                blk[ii].Sdim=l**2
                blk[ii].Ndim=l**2
                blk[ii].Wdim=l**4
                blk[ii].Edim=l**2
            elif ii<k-1:
                blk[ii].m=qp.mpomidsite(eigl,ii+1,dkm,k,n)
    return blk
    
def mps_readout(mps):
    ns=len(mps)
    mps[0].m=np.einsum('ijk->ik',mps[0].m)

    for jj in range(1,ns-1):
        mps[jj].m=np.einsum('ijk->jk',mps[jj].m)
    mps[ns-1].m=np.einsum('ijk->j',mps[ns-1].m)
    
    rh=np.einsum('ij,jk',mps[0].m,mps[1].m)
    for jj in range(2,ns-1):
        rh=np.einsum('ij,jk',rh,mps[jj].m)
    rh=np.einsum('ij,j',rh,mps[ns-1].m)
    
    return rh
    
    
        
def growth(rho,eigl,eta,ham,dt,dkm,n,ntot,svals,pr):
    qp.ctab=qp.mcoeffs(0,eta,dkm,dt,ntot)
    l=len(eigl)
    svds=['fraction','accuracy','chi']
    gmps=df.mps_block(l**2,2)
    mpos=qp.gr_mpostartsite(eigl,dkm,1,n,ham,dt)
    gmps[0].m=np.einsum('i,jikl',rho,mpos)
    gmps[1].m=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigl))
    for jj in range(0,dkm-2):
        bm.multiply_block(gmps,create_block(eigl,eta,ham,dt,dkm,jj+2,n,ntot),svds[svals],pr)
        tempmps_blk=df.mps_block(l**2,jj+3)
        for kk in range(jj+2):
            tempmps_blk[kk].m=gmps[kk].m
            tempmps_blk[kk].SNdim=gmps[kk].SNdim
            tempmps_blk[kk].Wdim=gmps[kk].Wdim
            tempmps_blk[kk].Edim=gmps[kk].Edim
        tempmps_blk[jj+2].m=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigl))
        tempmps_blk[jj+2].SNdim=gmps[jj+1].Edim
        tempmps_blk[jj+2].Wdim=gmps[jj+1].Edim
        tempmps_blk[jj+2].Edim=1
        gmps=tempmps_blk
    bm.multiply_block(gmps,create_block(eigl,eta,ham,dt,dkm,dkm,n,ntot),svds[svals],pr)
    return gmps
    
def growth2(rho,eigl,eta,ham,dt,dkm,n,ntot,svals,pr):
    qp.ctab=qp.mcoeffs(0,eta,dkm,dt,ntot)
    l=len(eigl)
    svds=['fraction','accuracy','chi']
    gmps=df.mps_block(l**2,2)
    mpos=qp.gr_mpostartsite(eigl,dkm,1,n,ham,dt)
    gmps[0].m=np.einsum('i,jikl',rho,mpos)
    gmps[1].m=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigl))
    for jj in range(0,dkm-2):
        bm.multiply_block(gmps,create_block(eigl,eta,ham,dt,dkm,jj+2,n,ntot),svds[svals],pr)
        gmps.append(df.mps_site(input_tensor=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigl))))
    bm.multiply_block(gmps,create_block(eigl,eta,ham,dt,dkm,dkm,n,ntot),svds[svals],pr)
    return gmps
    
def alg(rho,eigl,eta,ham,dt,dkm,ntot,c,p):
    data=[]
    svds=['fraction','accuracy','chi']
    mps=growth(rho,eigl,eta,ham,dt,dkm,dkm+1,ntot,c,p)
    mpo=create_block(eigl,eta,ham,dt,dkm,dkm+1,dkm+2,ntot)
    mpoterm=create_block(eigl,eta,ham,dt,dkm,dkm+1,dkm+1,ntot)
    for jj in range(ntot):
        mpsN=copy.deepcopy(mps)
        bm.multiply_block(mpsN,mpoterm,svds[c],p)
        data.append([(dkm+jj+1)*dt,mps_readout(mpsN)])
        bm.multiply_block(mps,mpo)
    
    return data
    
dkmax=4

qp.quapi(0,eigs,eta,dkmax,hamil,delt,irho,20,"tempocheck")
f=open("tempocheck"+str(dkmax)+".pickle")
myf=pickle.load(f)
print alg(irho,eigs,eta,hamil,delt,dkmax,20,2,16)
print myf[16]

'''clis=['fraction','accuracy','chi']
p=65
c=2
mps_s2=growth(irho,eigs,eta,hamil,delt,5,7,10,c,p)
bm.multiply_block(mps_s2,create_block(eigs,eta,hamil,delt,5,6,7,10),clis[c],p)
bm.multiply_block(mps_s2,create_block(eigs,eta,hamil,delt,5,7,7,10),clis[c],p)
print mps_s2[0].m.shape
print mps_s2[1].m.shape
print mps_s2[2].m.shape
print mps_s2[3].m.shape
print mps_s2[4].m.shape

print mps_readout(mps_s2)'''
#print np.einsum('ijkl->i',np.einsum('ijk,lkm',np.einsum('ijkl->ikl',np.einsum('ijk,lkn',mps_s2[0].m,mps_s2[1].m)),mps_s2[2].m))

#print alg(irho,eigs,eta,hamil,delt,3,5)




'''   
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
    
def mps_append(block):
    le=len(block)
    blk=df.mps_block(2,le+1)
    for ii in range(le):
        blk[ii].m=block[ii].m
    blk[le].m=qp.app([-1,1])
    
    return blk
    '''
    
'''
#lam_mat should now be the full rank-2dk propagator - (actually rank-(2dk+2).. the further 2 ranks are 'null'
#west/east legs of the start/end sites respectively)
lam_mat=block_contract(tblk)

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


