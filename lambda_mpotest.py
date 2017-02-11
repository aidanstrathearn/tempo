<<<<<<< Updated upstream
import newquaPyVec as qp
import definitions as df
import lineshapes as ln
import numpy as np
import pickle
import copy
import time
from numpy import linalg as la
import scipy.sparse.linalg as ssl

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
    
def growth_alg(mod,rho,eigl,eta,ham,dt,dkm,n,ntot,svals,pr):
    #the same as growth but records data for each timestep during growth
    #acheived by at each step making a copy of the mps (mpsN) and multiplying 
    #it with the termination mpo, reading out data from this,
    #then growing the original mps (gmps) one more site and repeating
    datlis=[[0,rho]]
    qp.ctab=qp.mcoeffs(mod,eta,dkm,dt,ntot)
    print qp.ctab
    l=len(eigl)
    svds=['fraction','accuracy','chi']
    gmps=df.mps_block(l**2,l**2,2)
    
    gmps.data[0].m=np.einsum('i,jikl',rho,qp.gr_mpostartsite(eigl,dkm,1,1,ham,dt))
    gmps.data[1].m=qp.gr_mpoedge(eigl)
    datlis.append([dt,gmps.readout()])
    
    gmps.data[0].m=np.einsum('i,jikl',rho,qp.gr_mpostartsite(eigl,dkm,1,n,ham,dt))
    gmps.data[1].m=qp.gr_mpoedge(eigl)
    ss,uu,vv=np.linalg.svd(np.einsum('ijk->ik',gmps.data[0].m), full_matrices=True)
    print uu
    print gmps.data[0].m.shape
    for jj in range(0,dkm-2):
        mpsN=copy.deepcopy(gmps)
        mpsN.multiply_block(create_block(eigl,ham,dt,dkm,jj+2,jj+2),svds[svals],pr)
        datlis.append([(jj+2)*dt,mpsN.readout()])
        del mpsN
        gmps.multiply_block(create_block(eigl,ham,dt,dkm,jj+2,n),svds[svals],pr)
        gmps.append_site(qp.gr_mpoedge(eigl))
        print "timestep: " +str(jj)
    
    mpsN=copy.deepcopy(gmps)
    mpsN.multiply_block(create_block(eigl,ham,dt,dkm,dkm,dkm),svds[svals],pr)
    datlis.append([(dkm)*dt,mpsN.readout()])
    del mpsN
    gmps.multiply_block(create_block(eigl,ham,dt,dkm,dkm,n),svds[svals],pr)
    return gmps,datlis
    
def tempo(mod,eigl,eta,dkm,ham,dt,rho,ntot,filename,c,p):
    t0=time.time()
    #algorithm to run tempo
    #c lables svd option in svds list and p is the precision option
    svds=['fraction','accuracy','chi']
    #first calls growth_alg to obtain initial data and mps - and create makri coeffs
    mps,datlis=growth_alg(mod,rho,eigl,eta,ham,dt,dkm,dkm+1,ntot,c,p)
    
    if mod==0:
        
        #creates bulk propagation mpo
        mpo=create_block(eigl,ham,dt,dkm,dkm+1,dkm+2)
        #creates bulk propagtion temination mpo
        mpoterm=create_block(eigl,ham,dt,dkm,dkm+1,dkm+1)
        #loops through copying mps, terminating, reading out, then multiplying mps
        for jj in range(ntot-dkm):
            print "timestep: " +str(jj+dkm+1)
            mpsN=copy.deepcopy(mps)
            mpsN.multiply_block(mpoterm,svds[c],p)
            datlis.append([(dkm+jj+1)*dt,mpsN.readout()])
            del mpsN
            mps.multiply_block(mpo,svds[c],p)
    
    else:
        for jj in range(ntot-dkm):
            print "timestep: " +str(jj+dkm+1)
            mpsN=copy.deepcopy(mps)
            mpsN.multiply_block(create_block(eigl,ham,dt,dkm,mod*jj+dkm+1,mod*jj+dkm+1),svds[c],p)
            datlis.append([(dkm+jj+1)*dt,mpsN.readout()])
            del mpsN
            mps.multiply_block(create_block(eigl,ham,dt,dkm,mod*jj+dkm+1,mod*jj+dkm+2),svds[c],p)
    
    
    #returns list of data
    print "Time for tempo algorithm: "+str(time.time()-t0)
    print "method: " +str(c)+" precision: " +str(p)
    #pickles data for later use but also returns it if you want to use itimmediately
    datfilep=open(filename+str(dkm)+".pickle","w")
    pickle.dump(datlis,datfilep)
    datfilep.close()

    return datlis
    

#defininf the lineshape used to find the makri coeffs
def eta(t):
    return ln.eta_0T(t,3,1,0.5*2)
    
def eta2(t):
    return 10*t**2
    
#some test parameters
hamil=[[0,1],[1,0]]
eigs=[-1,1]
delt=0.1
nsteps=10
hdim=len(eigs)
irho=[1,0,0,0]
meth=0
vals=1
modc=0
#defining local and operator dimensions
df.local_dim=hdim**2
dkmax=2
qp.trot=0

qp.ctab=qp.mcoeffs(modc,eta,dkmax,delt,10)

location="C:\\Users\\admin\\Desktop\\phd\\tempodata\\"


tar=np.zeros((1,2,3,4,5,6,7,8))

tar=np.rollaxis(tar,2,1)
print tar.shape
tar=np.rollaxis(tar,4,2)
print tar.shape
tar=np.rollaxis(tar,6,3)
print tar.shape

def block_contract(block):
    dkk=block.N_sites
    init=np.einsum('ijkl,mnlo',block.data[0].m,block.data[1].m)
    for jj in range(2,dkk):
        init=np.einsum('...i,jkil',init,block.data[jj].m)
    init=np.sum(np.sum(init,-1),2)
    return init

def lam_matrix(mod,eigl,eta,dkm,k,n,ham,dt,ntot):
    l=len(eigl)
    qp.ctab=qp.mcoeffs(mod,eta,dkm,dt,ntot)
    print qp.ctab
    lmat=block_contract(create_block(eigl,ham,dt,dkm,k,n))
    #print lmat
    print 'break2'
    for jj in range(dkm):
        lmat=np.rollaxis(lmat,2*jj,jj)
    #print lmat
    lmat=np.reshape(lmat,(l**(2*dkm),l**(2*dkm)))
    print 'breakkkk'   
    return lmat.T


#print lam_matrix(0,[-1,1],eta,2,4,5,hamil,delt,20)


eis= la.eig(lam_matrix(0,[-1,1],eta,2,4,5,hamil,delt,20))
vv=eis[0].real
print vv
#print ei.shape
#print ev.T.shape
#print ei
#print ev[:,0]

#print qp.arnoldi_ss(modc,eigs,eta,dkmax,hamil,delt)
'''
#get tempo data
tempo(modc,eigs,eta,dkmax,hamil,delt,irho,nsteps,location+"test",meth,vals)
#get quapi dat
qp.quapi(modc,eigs,eta,dkmax,hamil,delt,irho,nsteps,location+"tempocheck")

tdat=open(location+"test"+str(dkmax)+".pickle")
mytdat=pickle.load(tdat)
tdat.close()
dd=open(location+"test"+str(dkmax)+".dat","w")
for k in range(0,len(mytdat)):
    dd.write(str(mytdat[k][0])+" "+str(2*(mytdat[k][1][1]).real)+" "+str(2*(mytdat[k][1][1]).imag)+" "+str((mytdat[k][1][0]-mytdat[k][1][3]).real)+"\n")
dd.close()

qdat=open(location+"tempocheck"+str(dkmax)+".pickle")
myqdat=pickle.load(qdat)
qdat.close()
dd=open(location+"tempocheck"+str(dkmax)+".dat","w")
for k in range(0,len(myqdat)):
    dd.write(str(myqdat[k][0])+" "+str(2*(myqdat[k][1][1]).real)+" "+str(2*(myqdat[k][1][1]).imag)+" "+str((myqdat[k][1][0]-myqdat[k][1][3]).real)+"\n")
dd.close()

#comparing a datpoint- keeping all sv's reproduces quapi results
print 'TEMPO data:'
print mytdat[8][1]
print 'QUAPI data:'
print myqdat[8][1]
print 'Trace of tempo data (bond truncuation seems to affect trace preservation and hermiticity):'
print myqdat[8][1][0]+mytdat[8][1][3]

#print np.einsum('ijkl->i',np.einsum('ijk,lkm',np.einsum('ijkl->ikl',np.einsum('ijk,lkn',mps_s2[0].m,mps_s2[1].m)),mps_s2[2].m))
'''

'''   
#function to go through contracting the sites in a single block together using einsum
#to produce the full lambda matrix


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


=======
import newquaPyVec as qp
import definitions as df
import lineshapes as ln
import numpy as np
import pickle
import copy
import time
from numpy import linalg as la
import scipy.sparse.linalg as ssl

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
    
def growth_alg(mod,rho,eigl,eta,ham,dt,dkm,n,ntot,svals,pr):
    #the same as growth but records data for each timestep during growth
    #acheived by at each step making a copy of the mps (mpsN) and multiplying 
    #it with the termination mpo, reading out data from this,
    #then growing the original mps (gmps) one more site and repeating
    datlis=[[0,rho]]
    qp.ctab=qp.mcoeffs(mod,eta,dkm,dt,ntot)
    print qp.ctab
    l=len(eigl)
    svds=['fraction','accuracy','chi']
    gmps=df.mps_block(l**2,l**2,2)
    
    gmps.data[0].m=np.einsum('i,jikl',rho,qp.gr_mpostartsite(eigl,dkm,1,1,ham,dt))
    gmps.data[1].m=qp.gr_mpoedge(eigl)
    datlis.append([dt,gmps.readout()])
    
    gmps.data[0].m=np.einsum('i,jikl',rho,qp.gr_mpostartsite(eigl,dkm,1,n,ham,dt))
    gmps.data[1].m=qp.gr_mpoedge(eigl)
    ss,uu,vv=np.linalg.svd(np.einsum('ijk->ik',gmps.data[0].m), full_matrices=True)
    print uu
    print gmps.data[0].m.shape
    for jj in range(0,dkm-2):
        mpsN=copy.deepcopy(gmps)
        mpsN.multiply_block(create_block(eigl,ham,dt,dkm,jj+2,jj+2),svds[svals],pr)
        datlis.append([(jj+2)*dt,mpsN.readout()])
        del mpsN
        gmps.multiply_block(create_block(eigl,ham,dt,dkm,jj+2,n),svds[svals],pr)
        gmps.append_site(qp.gr_mpoedge(eigl))
        print "timestep: " +str(jj)
    
    mpsN=copy.deepcopy(gmps)
    mpsN.multiply_block(create_block(eigl,ham,dt,dkm,dkm,dkm),svds[svals],pr)
    datlis.append([(dkm)*dt,mpsN.readout()])
    del mpsN
    gmps.multiply_block(create_block(eigl,ham,dt,dkm,dkm,n),svds[svals],pr)
    return gmps,datlis
    
def tempo(mod,eigl,eta,dkm,ham,dt,rho,ntot,filename,c,p):
    t0=time.time()
    #algorithm to run tempo
    #c lables svd option in svds list and p is the precision option
    svds=['fraction','accuracy','chi']
    #first calls growth_alg to obtain initial data and mps - and create makri coeffs
    mps,datlis=growth_alg(mod,rho,eigl,eta,ham,dt,dkm,dkm+1,ntot,c,p)
    
    if mod==0:
        
        #creates bulk propagation mpo
        mpo=create_block(eigl,ham,dt,dkm,dkm+1,dkm+2)
        #creates bulk propagtion temination mpo
        mpoterm=create_block(eigl,ham,dt,dkm,dkm+1,dkm+1)
        #loops through copying mps, terminating, reading out, then multiplying mps
        for jj in range(ntot-dkm):
            print "timestep: " +str(jj+dkm+1)
            mpsN=copy.deepcopy(mps)
            mpsN.multiply_block(mpoterm,svds[c],p)
            datlis.append([(dkm+jj+1)*dt,mpsN.readout()])
            del mpsN
            mps.multiply_block(mpo,svds[c],p)
    
    else:
        for jj in range(ntot-dkm):
            print "timestep: " +str(jj+dkm+1)
            mpsN=copy.deepcopy(mps)
            mpsN.multiply_block(create_block(eigl,ham,dt,dkm,mod*jj+dkm+1,mod*jj+dkm+1),svds[c],p)
            datlis.append([(dkm+jj+1)*dt,mpsN.readout()])
            del mpsN
            mps.multiply_block(create_block(eigl,ham,dt,dkm,mod*jj+dkm+1,mod*jj+dkm+2),svds[c],p)
    
    
    #returns list of data
    print "Time for tempo algorithm: "+str(time.time()-t0)
    print "method: " +str(c)+" precision: " +str(p)
    #pickles data for later use but also returns it if you want to use itimmediately
    datfilep=open(filename+str(dkm)+".pickle","w")
    pickle.dump(datlis,datfilep)
    datfilep.close()

    return datlis
    

#defininf the lineshape used to find the makri coeffs
def eta(t):
    return ln.eta_0T(t,3,1,0.5*2)
    
def eta2(t):
    return 10*t**2
    
#some test parameters
hamil=[[0,1],[1,0]]
eigs=[-1,1]
delt=0.1
nsteps=10
hdim=len(eigs)
irho=[1,0,0,0]
meth=0
vals=1
modc=0
#defining local and operator dimensions
df.local_dim=hdim**2
dkmax=2
qp.trot=0

qp.ctab=qp.mcoeffs(modc,eta,dkmax,delt,10)

location="C:\\Users\\admin\\Desktop\\phd\\tempodata\\"


tar=np.zeros((1,2,3,4,5,6,7,8))

tar=np.rollaxis(tar,2,1)
print tar.shape
tar=np.rollaxis(tar,4,2)
print tar.shape
tar=np.rollaxis(tar,6,3)
print tar.shape

def block_contract(block):
    dkk=block.N_sites
    init=np.einsum('ijkl,mnlo',block.data[0].m,block.data[1].m)
    for jj in range(2,dkk):
        init=np.einsum('...i,jkil',init,block.data[jj].m)
    init=np.sum(np.sum(init,-1),2)
    return init

def lam_matrix(mod,eigl,eta,dkm,k,n,ham,dt,ntot):
    l=len(eigl)
    qp.ctab=qp.mcoeffs(mod,eta,dkm,dt,ntot)
    print qp.ctab
    lmat=block_contract(create_block(eigl,ham,dt,dkm,k,n))
    #print lmat
    print 'break2'
    for jj in range(dkm):
        lmat=np.rollaxis(lmat,2*jj,jj)
    #print lmat
    lmat=np.reshape(lmat,(l**(2*dkm),l**(2*dkm)))
    print 'breakkkk'   
    return lmat.T


#print lam_matrix(0,[-1,1],eta,2,4,5,hamil,delt,20)


eis= la.eig(lam_matrix(0,[-1,1],eta,2,4,5,hamil,delt,20))
vv=eis[0].real
print vv
#print ei.shape
#print ev.T.shape
#print ei
#print ev[:,0]

#print qp.arnoldi_ss(modc,eigs,eta,dkmax,hamil,delt)
'''
#get tempo data
tempo(modc,eigs,eta,dkmax,hamil,delt,irho,nsteps,location+"test",meth,vals)
#get quapi dat
qp.quapi(modc,eigs,eta,dkmax,hamil,delt,irho,nsteps,location+"tempocheck")

tdat=open(location+"test"+str(dkmax)+".pickle")
mytdat=pickle.load(tdat)
tdat.close()
dd=open(location+"test"+str(dkmax)+".dat","w")
for k in range(0,len(mytdat)):
    dd.write(str(mytdat[k][0])+" "+str(2*(mytdat[k][1][1]).real)+" "+str(2*(mytdat[k][1][1]).imag)+" "+str((mytdat[k][1][0]-mytdat[k][1][3]).real)+"\n")
dd.close()

qdat=open(location+"tempocheck"+str(dkmax)+".pickle")
myqdat=pickle.load(qdat)
qdat.close()
dd=open(location+"tempocheck"+str(dkmax)+".dat","w")
for k in range(0,len(myqdat)):
    dd.write(str(myqdat[k][0])+" "+str(2*(myqdat[k][1][1]).real)+" "+str(2*(myqdat[k][1][1]).imag)+" "+str((myqdat[k][1][0]-myqdat[k][1][3]).real)+"\n")
dd.close()

#comparing a datpoint- keeping all sv's reproduces quapi results
print 'TEMPO data:'
print mytdat[8][1]
print 'QUAPI data:'
print myqdat[8][1]
print 'Trace of tempo data (bond truncuation seems to affect trace preservation and hermiticity):'
print myqdat[8][1][0]+mytdat[8][1][3]

#print np.einsum('ijkl->i',np.einsum('ijk,lkm',np.einsum('ijkl->ikl',np.einsum('ijk,lkn',mps_s2[0].m,mps_s2[1].m)),mps_s2[2].m))
'''

'''   
#function to go through contracting the sites in a single block together using einsum
#to produce the full lambda matrix


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


>>>>>>> Stashed changes
