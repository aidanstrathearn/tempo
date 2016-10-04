import newquaPyVec as qp
import definitions as df
import lineshapes as ln
import numpy as np

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
    #initialising the block
    blk=df.mpo_block(l,dkm)
    #calculating makricoeffs and storing in ctab
    qp.ctab=qp.mcoeffs(0,eta,dkm,dt,ntot)
    #loops each site setting equal to the tensors created with newquaPyVec.py
    for ii in range(dkmax):
        if ii==0:
            blk[ii].m=qp.mpostartsite(eigl,dkm,k,n,ham,dt)
        elif ii==dkm-1:
            blk[ii].m=qp.mpoendsite(eigl,dkm,dkm,k,n)
        else:
            blk[ii].m=qp.mpomidsite(eigl,ii+1,dkm,k,n)
    return blk
     
#creating a test block     
tblk=create_block(eigs,eta,hamil,delt,dkmax,dkmax+1,dkmax+2,nsteps)

#function to go through contracting the blocks together using einsum
def block_contract(block):
    dkk=len(block)
    init=np.einsum('ijkl,mnlo',block[0].m,block[1].m)
    for jj in range(2,dkk):
        init=np.einsum('...i,jkil',init,block[jj].m)
    return init

#lam_mat should now be the full rank-2dk propagator - (actually rank-(2dk+2).. the further 2 ranks are 'null'
#west/east legs of the start/end sites respectively)
lam_mat=block_contract(tblk)

#inner two sums below contract over the null indices while the others contract over the additional dk-1 legs that 
#the lambda propagator as defined in standard quapi doesnt have - this is in order to compare it to the lambda tensor
#that is built in newquaPyVec.py... finally arrange the remaining indices into the same order as the lamtens from quapy
#and comparing the resulting tensor with one built the standard quapi way.. perfect match!
#note that the sums and index swapping done here is specific to kmax=3
lam_mat=np.swapaxes(np.swapaxes(np.swapaxes(sum(sum(sum(sum(lam_mat,2),-1),2),0),2,3),0,3),1,2)
print lam_mat-qp.lamtens(eigs,3,4,5,hamil,delt)


