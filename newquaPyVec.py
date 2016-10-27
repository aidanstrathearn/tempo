from cmath import exp
from numpy import array,zeros,kron,reshape,swapaxes,expand_dims,repeat,einsum,eye,sum
import time
from scipy.linalg import expm
import pickle
import lineshapes as ln

def mcoeffs(mod,et,dk,dt,ntot):
    #function to calculate coeffs for a given lineshape et, delta_k_max dk, timestep dt, and number of
    #propagation points ntot
    #mod=1/0 gives modified/standard coeffs
    if mod==0:
        etab=array([zeros(2*(dk+1)+1,dtype=complex),zeros(2*(dk+1)+1,dtype=complex),zeros((dk+1),dtype=complex)])
        tb=zeros(2*(dk+2)+1,dtype=complex)
        
        for j in range(1,2*(dk+2)+1):
            tb[j]=et(j*0.5*dt)
        etab[0][0]=tb[2]
        etab[1][0]=tb[1]
        etab[2][0]=tb[1]
        
        for j in range(1,dk+1):
            #calculating the coefficients by taking finite differences on the lineshape
            #as detailed in my report
            etab[0][j]=tb[(2*j+2)]-tb[2*j]-tb[2*j]+tb[(2*j-2)]
            etab[1][j]=tb[2*j+1]-tb[2*j-1]-tb[2*j]+tb[2*(j-1)]
            etab[2][j]=tb[(2*j)]-tb[2*j-1]-tb[2*j-1]+tb[2*(j-1)]
            
        for j in range(1,dk+3):
            etab[0][dk+j]=etab[0][dk]
            etab[1][dk+j]=etab[1][dk]
    else:
        etab=array([zeros((dk+1+ntot+dk),dtype=complex),zeros((dk+1+ntot+dk),dtype=complex),zeros((dk+1),dtype=complex)])
        #etab will be the table of makri coeffecients that the influencece functional uses
        # and has the format:
        #[[eta_kk, eta_dk1, eta_dk2...],           (mid evolution coeffs)
        #[eta_00, eta_01, eta_02...]               (end/start evolution coeffs)
        #[eta_00, eta_0N1, eta_0N2...]]            (coeffs connecting start point to end point)
        tb=zeros(2*(ntot+dk)+2,dtype=complex)

        for j in range(1,2*(ntot+dk)+2):
            tb[j]=et(j*0.5*dt)
        
        etab[0][0]=tb[2]
        etab[1][0]=tb[1]
        etab[2][0]=tb[1]
        for j in range(1,dk+1):
            #calculating the coefficients by taking finite differences on the lineshape
            #as detailed in my report
            etab[0][j]=tb[(2*j+2)]-tb[2*j]-tb[2*j]+tb[(2*j-2)]
            etab[1][j]=tb[2*j+1]-tb[2*j-1]-tb[2*j]+tb[2*(j-1)]
            etab[2][j]=tb[(2*j)]-tb[2*j-1]-tb[2*j-1]+tb[2*(j-1)]

        for j in range(1,ntot+dk+1):
            etab[0][dk+j]=tb[2*j+1]-tb[2*j-1]-tb[2*dk]+tb[2*(dk-1)]
            etab[1][dk+j]=tb[2*j]-tb[2*j-1]-tb[2*dk-1]+tb[2*(dk-1)]
    
        
    return etab


def freeprop(ham,dt):
    #function to create the freepropagator factor of the lambda tensor
    #for a given hamiltonian ham and timestep dt
    #define unitary time evol operator
    u=expm(array(ham)*dt*(-1j))
    #take the kronecker product of u and its conjugate transpose
    kprop=kron(u.conj(),u).conj()
    return kprop


def icomp(sp,sm,sdp,sdm,dk,k,n,dkm):
    #gives a single component of discrete influence functional for a given
    #current point k, end point n, and memory span dk
    #bl is used to determine which row of ctab the makri coeffecients are taken from
    bl=int(k==dk or k==n)+int(k==dk==n)
    #bl2 is only important when modified coeffs are used and picks out the correct modified
    #delta k max coeff for the given time point k*dt
    bl2=k*int(dk==dkm and k>dk)
    #phi is the influence phase and the exponential of this is returned
    phi=-(sp-sm)*(ctab[bl][dk+bl2]*sdp-ctab[bl][dk+bl2].conj()*sdm)
    return exp(phi)
    
def itab(eigl,dk,k,n,dkm):
    #explicitly constructs the influence functional factors Idk(sk,sdk)
    #eigl is the list of eigenvalues of the coupled system operator 
    #(these are the values that sk and sdk can take)
    #l is the hilbert space dimension
    #ec is the list of all possible pairs of eigenvalues which are simultaneously
    #inserted into sp and sm, treating them together as a single index with l^2 values
    #rather than 2 separate indices with l values each
    l=len(eigl)
    ec=zeros((l,l,2))
    for j in range(l):
        for kk in range(l):
            ec[j][kk][0]=eigl[j]
            ec[j][kk][1]=eigl[kk]
    ec=ec.reshape((l**2,2))
    #initializing the tensor
    tab=zeros((l**2,l**2),dtype=complex)
    #the bool here is used when dk=0 and serves to remove dependence on 
    #the sdk coordinate since dk=0 piece of the influence functional
    #depends on sk only
    bl=int(dk==0)
    #loops through each element of tab and assigns it the correct influence
    #functional component with the bl part removing sdp/sdm where necessary
    for sd in range(l**2):
        for s in range(l**2):
            tab[sd][s]=icomp(ec[s][0],ec[s][1],ec[sd][0]-bl*(ec[sd][0]-ec[s][0]),ec[sd][1]-bl*(ec[sd][1]-ec[s][1]),dk,k,n,dkm)
    return tab
    

    


def itpad(eigl,dk,k,n,dkm):
    #takes the rank-2 itab tensors and gives them additional dummy indices
    #to make them rank-(dk+1)
    #in numpy language the indices are called axes
    l=len(eigl)
    #first store the rank-2 tensor
    itp=itab(eigl,dk,k,n,dkm)
    #for dk=0,1 the tensor is already in the correct form so return it
    if dk==0 or dk==1:
        return itp
    else:
        #expand_dims increases the depth of the array by one and repeat
        #copies the existing element of this new array level to give
        #the correct dimensions
        for k in range(0,dk-1):
            itp=repeat(expand_dims(itp,1),l**2,1)
        return itp

    
def lamtens(eigl,dkm,k,n,ham,dt):
    #constructs the lambda propagator tensor by multiplying
    #together the padded influence functional factors and finally
    #multiplying in the free propagator 
    #multiplying here means element-by-element multiplication
    #numpys array broadcasting allows tensors of different rank
    #to be multiplied together but is set so that the tensor of lower rank is multiplied into
    #the most righthand indices of the higher rank tensor
    tens=itpad(eigl,dkm,k,n,dkm)
    for j in range(0,dkm):
        tens=tens*itpad(eigl,dkm-j-1,k,n,dkm)
    tens=tens*freeprop(ham,dt)
    return tens
    
    
def initprop(eigl,dkm,n,ham,dt):
    #this function constructs the initializing propagator
    #which is used to take the reduced density matrix at point 0
    #to the augmented density tensor at point delta_k_max
    #ready for propagation
    #essentially the same as the tensor growth part of tempo
    #hilbert space dimension
    l=len(eigl)
    #initialise tensor - the k=0 dk=0 itpad has its indices switched with einsum
    #and multiplied into the kmax=1 lambda tensor
    prop=lamtens(eigl,1,1,n,ham,dt)*itpad(eigl,0,0,0,0).T
    #successively multiplies in larger kmax lambda tensors to give the exact
    #influence functional between k=0 and k=dkm
    for j in range(0,dkm-1):
        #indices are wrong way round to rely on array broadcasting here so 
        #the tensor must be made into the same rank as the lambda tensor that is to
        #be multiplied in using expand_dims and repeat
        prop=repeat(expand_dims(prop,-1),l**2,-1)
        prop=prop*lamtens(eigl,j+2,j+2,n,ham,dt)
    return prop

def exact(eigl,dkm,ham,dt,initrho):
    #propagates the first dkm steps where no memory cutoff is used
    #initialize data
    #currently setup to record the population difference of a TLS
    data=[[0,initrho]]
    for j in range(dkm):
        #multiplying in the initial reduced density matrix with the exact propagator with einsum
        aug=einsum('i...,i...->i...',initrho,initprop(eigl,j+1,j+1,ham,dt))
        for k in range(j+1):
            #contracting 1 index at a time until what is left is the reduced
            #density matrix at point k=j
            aug=einsum('ij...->j...',aug)
        #appending the population difference of the red dens matrix and the time to data
        data.append([(j+1)*dt,aug])
    #returns the final data list, ready to have the next set of data appended
    return data

def quapi(mod,eigl,eta,dkm,ham,dt,initrho,ntot,filename):
    l=len(eigl)
    t0=time.time()
    rhovec=array(initrho).reshape(l**2)
    #full algorithm to find data at dkm+ntot points
    #mod turns on/off the modified coeffs, eigl is the list of eigenvalues of the system
    #operator, eta is the lineshape, dkm is delta_k_max, ham is the hamiltonian, dt is the timestep
    #initrho is the initial state density matrix, ntot is the number of points to propagator after the
    #initial dkm exact points, filename is the name of the file data is saved to
    #eventually coming out as "filename"+str(dkm)+".pickle"
    print "deltakmax: "+str(dkm)
    print " # of points: "+str(ntot)
    #globally defining ctab since it is called in a previous function but never defined previously
    global ctab
    ctab=mcoeffs(mod,eta,dkm,dt,ntot)
    print "Time for coeffs: "+ str(time.time()-t0)
    #first do the dkm points that are exact and initialize data to this
    t1=time.time()
    data=exact(eigl,dkm,ham,dt,rhovec)
    #initialize augmented density tensor with the exact propagator to point k=dkm
    #the ellipses tells einsum to use array broadcasting to do the multiplication
    aug=einsum('i...,i...->i...',rhovec,initprop(eigl,dkm,dkm+1,ham,dt))
    #contracts first two indices to get rank-(dk-1) aug density tensor
    aug=einsum('ij...->j...',aug)
    if mod==0:
        #standard quapi uses the same propagator at each step so these are stored before
        #propagation begins
        #define the propagor that takes aug from one point to the next
        prop=lamtens(eigl,dkm,dkm+1,dkm+2,ham,dt)
        #define the termination tensor
        term=lamtens(eigl,dkm,dkm+1,dkm+1,ham,dt)
    
        for j in range(ntot-dkm):
            #first pad aug to the same size as term
            augN=repeat(expand_dims(aug,-1),l**2,-1)
            #multiply in the termination tensor
            augN=augN*term
            #successively contract indices until you are left with reduced density matrix
            for k in range(dkm):
                augN=einsum('ij...->j...',augN)
            #print data to check its coming out
            #extract pop difference and append to data
            data.append([(j+1+dkm)*dt,augN])
            print [(j+1+dkm)*dt,augN]
            #pad aug ready for propagation one step forward
            aug=repeat(expand_dims(aug,-1),l**2,-1)
            #multiply in propagator and
            #contract last 2 indices to give new augmnented density tensor ready to be 
            #terminated in the next iteration of the for loop
            aug=einsum('ij...->j...',aug*prop)

    else:
        #with the modified coeffs a different propagator is required at each step so these are constructed
        #with every iteration of the loop
        for j in range(ntot-dkm):
            #first pad aug to the same size as term
            augN=repeat(expand_dims(aug,-1),l**2,-1)
            #multiply in the termination tensor
            augN=augN*lamtens(eigl,dkm,dkm+1+j,dkm+1+j,ham,dt)
            #successively contract indices until you are left with reduced density matrix
            for k in range(dkm):
                augN=einsum('ij...->j...',augN)
            #print data to check its coming out
            #extract pop difference and append to data
            data.append([(j+1+dkm)*dt,augN])
            print [(j+1+dkm)*dt,augN]
            #pad aug ready for propagation one step forward
            aug=repeat(expand_dims(aug,-1),l**2,-1)
            #multiply in propagator
            #contract last 2 indices to give new augmnented density tensor ready to be 
            #terminated in the next iteration of the for loop
            aug=einsum('ij...->j...',aug*lamtens(eigl,dkm,dkm+1+j,dkm+2+j,ham,dt))
        
    print "Time for algorithm: "+str(time.time()-t1)
    #pickles data for later use but also returns it if you want to use itimmediately
    datfilep=open(filename+str(dkm)+".pickle","w")
    pickle.dump(data,datfilep)
    datfilep.close()
    #deletes global variable ctab
    print "Total running time: "+str(time.time()-t0)+"\n"
    del ctab
    return data

#functions below to construct individual mpo sites
#I have used int(a==b) to represent kronecker_delta(a,b)




def mpostartsite(eigl,dkm,k,n,ham,dt):
    l=len(eigl)
    #ec is created as a list of all possible pairs of values of the pairs of west/east legs
    #which are then inserted simultaneously into the expression for the component of the tensor
    #so that we end up with just a single west leg and a single east leg
    ec=zeros((l**2,l**2,2))
    for j in range(l**2):
        for kk in range(l**2):
            ec[j][kk][0]=j
            ec[j][kk][1]=kk
    ec=ec.reshape((l**4,2))
    #initialize the block as tab in the form required for dainius' mpo definitions (south, north,east west)
    #here the dimension of the east leg is 1 since this is the start site
    tab=zeros((l**2,l**2,1,l**4),dtype=complex)
    #the combination of I0 I1 and K whose components make up the start site tensor
    tens=itab(eigl,1,k,n,dkm)*itab(eigl,0,k,n,dkm)*freeprop(ham,dt)
    #looping through each index and assigning values
    for i1 in range(l**2):
        for j1 in range(l**2):
            for a1 in range(l**4):
                if (ec[a1][0]==i1 and ec[a1][1]==j1):
                    tab[j1][i1][0][a1]=tens[j1][i1]
    return tab
    
def mpomidsite(eigl,dk,dkm,k,n):
    l=len(eigl)
    ec=zeros((l**2,l**2,2),dtype=int)
    for j in range(l**2):
        for kk in range(l**2):
            ec[j][kk][0]=j
            ec[j][kk][1]=kk
    ec=ec.reshape((l**4,2))
    tab=zeros((l**2,l**2,l**4,l**4),dtype=complex)
    for i1 in range(l**2):
        for j1 in range(l**2):
            for a1 in range(l**4):
                for b1 in range(l**4):
                    if (ec[a1][0]==ec[b1][0] and ec[b1][1]==i1 and ec[a1][1]==j1):
                        tab[j1][i1][b1][a1]=itab(eigl,dk,k,n,dkm)[j1][ec[b1][0]]
    return tab
    
def mpoendsite(eigl,dk,dkm,k,n):
    l=len(eigl)
    ec=zeros((l**2,l**2,2),dtype=int)
    for j in range(l**2):
        for kk in range(l**2):
            ec[j][kk][0]=j
            ec[j][kk][1]=kk
    ec=ec.reshape((l**4,2))
    tab=zeros((l**2,l**2,l**4,1),dtype=complex)
    for i1 in range(l**2):
        for j1 in range(l**2):
            for b1 in range(l**4):
                if i1==ec[b1][1]:
                    tab[j1][i1][b1][0]=itab(eigl,dk,k,n,dkm)[j1][ec[b1][0]]
    return tab

def gr_mpostartsite(eigl,dkm,k,n,ham,dt):
    l=len(eigl)
    #ec is created as a list of all possible pairs of values of the pairs of west/east legs
    #which are then inserted simultaneously into the expression for the component of the tensor
    #so that we end up with just a single west leg and a single east leg
    #initialize the block as tab in the form required for dainius' mpo definitions (south, north,east west)
    #here the dimension of the east leg is 1 since this is the start site
    tab=zeros((l**2,l**2,1,l**2),dtype=complex)
    #the combination of I0 I1 and K whose components make up the start site tensor
    tens=itab(eigl,1,k,n,dkm)*itab(eigl,0,k,n,dkm)*freeprop(ham,dt)*itab(eigl,0,0,n,dkm).T
    #looping through each index and assigning values
    for i1 in range(l**2):
        for j1 in range(l**2):
            for a1 in range(l**2):
                if a1==j1:
                    tab[j1][i1][0][a1]=tens[j1][i1]
    return tab
    
    
def gr_mpoendsite(eigl,dk,dkm,k,n):
    l=len(eigl)
    ec=zeros((l**2,l**2,2),dtype=int)
    for j in range(l**2):
        for kk in range(l**2):
            ec[j][kk][0]=j
            ec[j][kk][1]=kk
    ec=ec.reshape((l**4,2))
    tab=zeros((l**2,l**2,l**4,l**2),dtype=complex)
    for i1 in range(l**2):
        for j1 in range(l**2):
            for b1 in range(l**4):
                for a1 in range(l**2):
                    if j1==a1 and i1==ec[b1][1]:
                        tab[j1][i1][b1][a1]=itab(eigl,dk,k,n,dkm)[j1][ec[b1][0]]
    return tab

def gr_mpodummymid(eigl):
    l=len(eigl)
    tab=zeros((l**2,l**2,l**2,l**2),dtype=complex)
    for i1 in range(l**2):
        for j1 in range(l**2):
            for b1 in range(l**2):
                for a1 in range(l**2):
                    if (j1==a1 and i1==a1 and j1==b1 and i1==b1):
                        tab[j1][i1][b1][a1]=1
    return tab
    
def gr_mpodummyedge(eigl):
    l=len(eigl)
    tab=zeros((l**2,l**2,l**2,1),dtype=complex)
    for i1 in range(l**2):
        for j1 in range(l**2):
            for b1 in range(l**2):
                if (j1==b1 and i1==b1):
                    tab[j1][i1][b1][0]=1
    return tab

def mpsdummymid(eigl):
    l=len(eigl)
    tab=zeros((l**2,l**2,l**2),dtype=complex)
    for i1 in range(l**2):
        for b1 in range(l**2):
            for a1 in range(l**2):
                if (i1==b1 and i1==a1):
                    tab[i1][b1][a1]=1
    return tab

def mpsdummyend(eigl):
    l=len(eigl)
    tab=zeros((l**2,l**2,1),dtype=complex)
    for i1 in range(l**2):
        for b1 in range(l**2):
            if i1==b1:
                tab[i1][b1][0]=1
    return tab

def mpsrho(eigl,rho):
    l=len(eigl)
    tab=zeros((l**2,1,l**2),dtype=complex)
    for i1 in range(l**2):
        for b1 in range(l**2):
            if i1==b1:
                tab[i1][0][b1]=rho[i1]
    return tab
#Test stuff


#print einsum('ijkl->i',mpomidsite2([-1,1],3,4,5,6)-mpomidsite([-1,1],3,4,5,6))


#print einsum('ijkl->l',gr_mpoendsite([-1,1],3,3,5,6))



'''
con=einsum('ijkl,mnlo',mpostartsite([-1,1],2,3,4,[[0,1],[1,0]],1),mpoendsite([-1,1],2,2,3,4))
con=swapaxes(swapaxes(sum(sum(sum(con,2),-1),0),1,2),0,2)
con2=einsum('ijkl,mnlo',mpostartsite([-1,1],3,4,5,[[0,1],[1,0]],1),mpomidsite([-1,1],2,3,4,5))
con2=einsum('...i,jkil',con2,mpoendsite([-1,1],3,3,4,5))
con2=swapaxes(swapaxes(swapaxes(sum(sum(sum(sum(con2,2),-1),0),1),2,3),0,3),1,2)
#print mpostartsite([-1,1],[[0,1],[1,0]],1).shape
print con.shape
print lamtens([-1,1],2,3,4,[[0,1],[1,0]],1)- con
print con2.shape
print lamtens([-1,1],3,4,5,[[0,1],[1,0]],1)- con2'''
#print lamtens([-1,1],1,3,4,[[0,1],[1,0]],1)- sum(sum(mpostartsite([-1,1],[[0,1],[1,0]],1),-1),-1)
'''con=einsum('...i,jkil->...jkl',con,mpomidsite([-1,1],2))
print con.shape'''



'''N_sites=4; defs.local_dim=4; defs.bond_dim=1; defs.opdim=16
test=defs.mpo_block(defs.opdim,N_sites)
print test[3].m.shape'''



#Timings per data point on my laptop
#OLD QUAPI: 
#kmax 1-7 all < 0.01 s
#kmax 8 ~ 0.02 s
#kmax 9 ~ 0.1 s
#kmax 10 ~ 0.3 s
#NEW QUAPI
#kmax 1-6 all < 0.01 s
#kmax 7 ~ 0.02 s
#kmax 8 ~ 0.08 s
#kmax 9 ~ 0.3 s
#kmax 10 ~ 1.3 s






