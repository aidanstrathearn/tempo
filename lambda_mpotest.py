import newquaPyVec as qp
import definitions as df
import lineshapes as ln
import numpy as np
import block_multiplication
import pickle
import copy

def create_block(eigl,ham,dt,dkm,k,n):
    #function that creates an mpo block for:
    #system eigenvales-eigl,
    #hamiltonian-ham
    #timestep-dt
    #deltakmax-dkm
    #current position -k 
    #readout point -n

    #dimension of hilbert space
    l=len(eigl)

    #initiate block - just a single site the rest are appended
    blk=df.mpo_block(l**4,l**2,1)
    blk.data[0]=df.mpo_site(input_tensor=qp.mpostartsite(eigl,dkm,k,n,ham,dt))
    
    #if k>dkm-1 then we are in bulk evolution no growth so standard quapi mpo sites are appended in order
    #until there there deltak_max sites
    if k>dkm-1:
        for ii in range(1,dkm-1):
            blk.append_site(qp.mpomidsite(eigl,ii+1,dkm,k,n))
        blk.append_site(qp.mpoendsite(eigl,dkm,dkm,k,n))
    #if k<dkm-1 there are only k sites with the last one having east index not =1
    else:
        for ii in range(1,k-1):
            blk.append_site(qp.mpomidsite(eigl,ii+1,dkm,k,n))
        blk.append_site(qp.gr_mpoendsite(eigl,k,dkm,k,n))
        
    return blk

def growth(rho,eigl,eta,ham,dt,dkm,n,ntot,svals,pr):
    #this function grows the mps to the size ready for bulk propagation
    #svals indexs the choice of svd options given in list svds
    #pr is the precision variable
    svds=['fraction','accuracy','chi']
    #eta is lineshape used to calculate makri coeffs
    qp.ctab=qp.mcoeffs(0,eta,dkm,dt,ntot)
    l=len(eigl)
    datlis=[[0,rho]]
    for j in range(dkm):
        datlis.append([[dt*j,'no data']])
    
    #initialize mps block with 1 site
    gmps=df.mps_block(l**2,l**2,1)
    #since we cant do the svd on a single site block this first step is just done with einsum  
    gmps.data[0]=df.mps_site(input_tensor=np.einsum('i,jikl',rho,qp.gr_mpostartsite(eigl,dkm,1,n,ham,dt)))
    #grow the block by 1 site with the delta function edge tensor
    gmps.append_site(qp.gr_mpoedge(eigl))
    #loop through multipling mpo into mps then growing the mps by one site
    for jj in range(0,dkm-2):
        gmps.multiply_block(create_block(eigl,ham,dt,dkm,jj+2,n),svds[svals],pr)
        gmps.append_site(qp.gr_mpoedge(eigl))
    
    #multiplies in first deltak_max site mpo giving augmented density tensor mps at the deltak_max step
    gmps.multiply_block(create_block(eigl,ham,dt,dkm,dkm,n),svds[svals],pr)
    return gmps,datlis
    
def growth_alg(rho,eigl,eta,ham,dt,dkm,n,ntot,svals,pr):
    #the same as growth but records data for each timestep during growth
    #acheived by at each step making a copy of the mps (mpsN) and multiplying 
    #it with the termination mpo, reading out data from this,
    #then growing the original mps (gmps) one more site and repeating
    datlis=[[0,rho]]
    qp.ctab=qp.mcoeffs(0,eta,dkm,dt,ntot)
    l=len(eigl)
    svds=['fraction','accuracy','chi']
    gmps=df.mps_block(l**2,l**2,2)
    
    mpos=qp.gr_mpostartsite(eigl,dkm,1,1,ham,dt)
    gmps.data[0].m=np.einsum('i,jikl',rho,mpos)
    gmps.data[1].m=qp.gr_mpoedge(eigl)
    datlis.append([dt,gmps.readout()])
    
    mpos=qp.gr_mpostartsite(eigl,dkm,1,n,ham,dt)
    gmps.data[0].m=np.einsum('i,jikl',rho,mpos)
    gmps.data[1].m=qp.gr_mpoedge(eigl)
    for jj in range(0,dkm-2):
        mpsN=copy.deepcopy(gmps)
        mpsN.multiply_block(create_block(eigl,ham,dt,dkm,jj+2,jj+2),svds[svals],pr)
        datlis.append([(jj+2)*dt,mpsN.readout()])
        del mpsN
        gmps.multiply_block(create_block(eigl,ham,dt,dkm,jj+2,n),svds[svals],pr)
        gmps.append_site(qp.gr_mpoedge(eigl))
    
    mpsN=copy.deepcopy(gmps)
    mpsN.multiply_block(create_block(eigl,ham,dt,dkm,dkm,dkm),svds[svals],pr)
    datlis.append([(dkm)*dt,mpsN.readout()])
    del mpsN
    gmps.multiply_block(create_block(eigl,ham,dt,dkm,dkm,n),svds[svals],pr)
    return gmps,datlis
    
def tempo(eigl,eta,dkm,ham,dt,rho,ntot,c,p):
    #algorithm to run tempo
    #c lables svd option in svds list and p is the precision option
    svds=['fraction','accuracy','chi']
    #first calls growth_alg to obtain initial data and mps - and create makri coeffs
    mps,datlis=growth_alg(rho,eigl,eta,ham,dt,dkm,dkm+1,ntot,c,p)
    #creates bulk propagation mpo
    mpo=create_block(eigl,ham,dt,dkm,dkm+1,dkm+2)
    #creates bulk propagtion temination mpo
    mpoterm=create_block(eigl,ham,dt,dkm,dkm+1,dkm+1)
    #loops through copying mps, terminating, reading out, then multiplying mps
    for jj in range(ntot-dkm):
        mpsN=copy.deepcopy(mps)
        mpsN.multiply_block(mpoterm,svds[c],p)
        datlis.append([(dkm+jj+1)*dt,mpsN.readout()])
        del mpsN
        mps.multiply_block(mpo,svds[c],p)
    #returns list of data
    return datlis

#defininf the lineshape used to find the makri coeffs
def eta(t):
    return ln.eta_0T(t,3,1,0.5)
    
#some test parameters
hamil=[[0,1],[1,0]]
eigs=[-1,1]
delt=0.1
nsteps=8
hdim=len(eigs)
irho=[1,0,0,0]

#defining local and operator dimensions
df.local_dim=hdim**2
dkmax=5



#get tempo data
datf=tempo(eigs,eta,dkmax,hamil,delt,irho,nsteps,0,1)

#get quapi dat
qp.quapi(0,eigs,eta,dkmax,hamil,delt,irho,nsteps,"tempocheck")
f=open("tempocheck"+str(dkmax)+".pickle")
myf=pickle.load(f)

#comparing - keeping all sv's reproduces quapi results
print datf[8][1]
print myf[8][1]
print datf[8][1][0]+datf[8][1][3]


'''
print df.mps_site(input_tensor=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigs)),t=1).m.shape
print df.mps_site(input_tensor=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigs)),t=1).SNdim
print df.mps_site(input_tensor=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigs)),t=1).Wdim
print df.mps_site(input_tensor=np.einsum('ijkl->ikl',qp.gr_mpoedge(eigs)),t=1).Edim'''

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


