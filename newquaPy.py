from cmath import exp
from numpy import array,zeros,kron,reshape,swapaxes,expand_dims,repeat,einsum
import time
from scipy.linalg import expm
import pickle

def mcoeffs(mod,et,dk,dt,ntot):
    #function to calculate coeffs for a given lineshape et, delta_k_max dk, timestep dt, and number of
    #propagation points ntot
    #mod=1/0 gives modified/standard coeffs
    etab=array([zeros((dk+1+ntot+dk),dtype=complex),zeros((dk+1+ntot+dk),dtype=complex),zeros((dk+1),dtype=complex)])
    #etab will be the table of makri coeffecients that the influencece functional uses
    # and has the format:
    #[[eta_kk, eta_dk1, eta_dk2...],           (mid evolution coeffs)
    #[eta_00, eta_01, eta_02...]               (end/start evolution coeffs)
    #[eta_00, eta_0N1, eta_0N2...]]            (coeffs connecting start point to end point)
    etab[0][0]=et(dt)
    etab[1][0]=et(dt*0.5)
    etab[2][0]=et(dt*0.5)
    for j in range(dk):
        #calculating the coefficients by taking finite differences on the lineshape
        #as detailed in my report
        etab[0][j+1]=et((j+2)*dt)-et((j+1)*dt)-et((j+1)*dt)+et(j*dt)
        etab[1][j+1]=et((j+1.5)*dt)-et((j+0.5)*dt)-et((j+1)*dt)+et(j*dt)
        etab[2][j+1]=et((j+1)*dt)-et((j+0.5)*dt)-et((j+0.5)*dt)+et(j*dt)
    if mod==1:
    #these are the modified coeffs 
        for j in range(ntot+dk):
            etab[0][dk+1+j]=et((j+1.5)*dt)-et((j+0.5)*dt)-et(dk*dt)+et((dk-1)*dt)
            etab[1][dk+1+j]=et((j+1)*dt)-et((j+0.5)*dt)-et((dk-0.5)*dt)+et((dk-1)*dt)
    else:
    #otherwise they are just the same as the the standard delta k max coeffs
        for j in range(ntot+dk):
            etab[0][dk+1+j]=etab[0][dk]
            etab[1][dk+1+j]=etab[1][dk]
    return etab
    
    
def freeprop(ham,dt):
    #function to create the freepropagator factor of the lambda tensor
    #for a given hamiltonian ham and timestep dt
    #l is dimension of hilbert space
    l=len(ham)
    #define unitary time evol operator
    u=expm(array(ham)*dt*(-1j))
    #take the kronecker product of u and its conjugate transpose
    kprop=kron(u,u.T.conj())
    #reshapes into a 4 index tensor
    kprop=reshape(kprop,(l,l,l,l))
    #swaps indices to correct order
    kprop=swapaxes(kprop,0,2)
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
    l=len(eigl)
    #initializing the tensor
    tab=zeros((l,l,l,l),dtype=complex)
    #the bool here is used when dk=0 and serves to remove dependence on 
    #the sdk coordinate since dk=0 piece of the influence functional
    #depends on sk only
    bl=int(dk==0)
    #loops through each element of tab and assigns it the correct influence
    #functional component with the bl part removing sdp/sdm where necessary
    for sp in range(l):
        for sm in range(l):
            for sdp in range(l):
                for sdm in range(l):
                    tab[sdp][sdm][sp][sm]=icomp(eigl[sp],eigl[sm],eigl[sdp-bl*(sdp-sp)],eigl[sdm-bl*(sdm-sm)],dk,k,n,dkm)      
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
            itp=expand_dims(expand_dims(itp,2),2)
            itp=repeat(repeat(itp,l,2),l,3)
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
    prop=lamtens(eigl,1,1,n,ham,dt)*einsum('ijkl->klij',itpad(eigl,0,0,0,0))
    #successively multiplies in larger kmax lambda tensors to give the exact
    #influence functional between k=0 and k=dkm
    for j in range(0,dkm-1):
        #indices are wrong way round to rely on array broadcasting here so 
        #the tensor must be made into the same rank as the lambda tensor that is to
        #be multiplied in using expand_dims and repeat
        prop=expand_dims(expand_dims(prop,-1),-1)
        prop=repeat(repeat(prop,l,-1),l,-2)
        prop=prop*lamtens(eigl,j+2,j+2,n,ham,dt)
    return prop

def exact(eigl,dkm,ham,dt,initrho):
    #propagates the first dkm steps where no memory cutoff is used
    #initialize data
    #currently setup to record the population difference of a TLS
    data=[[0,initrho[0][0]-initrho[1][1]]]
    for j in range(dkm):
        #multiplying in the initial reduced density matrix with the exact propagator with einsum
        aug=einsum('ij...,ij...->ij...',initrho,initprop(eigl,j+1,j+1,ham,dt))
        for k in range(j+1):
            #contracting 2 indices at a time until what is left is the reduced
            #density matrix at point k=j
            aug=einsum('ijk...->k...',aug)
        #appending the population difference of the red dens matrix and the time to data
        data.append([(j+1)*dt,(aug[0][0]-aug[1][1]).real])
    #returns the final data list, ready to have the next set of data appended
    return data

def quapi(mod,eigl,eta,dkm,ham,dt,initrho,ntot,filename):
    #full algorithm to find data at dkm+ntot points
    #mod turns on/off the modified coeffs, eigl is the list of eigenvalues of the system
    #operator, eta is the lineshape, dkm is delta_k_max, ham is the hamiltonian, dt is the timestep
    #initrho is the initial state density matrix, ntot is the number of points to propagator after the
    #initial dkm exact points, filename is the name of the file data is saved to
    #eventually coming out as "filename"+str(dkm)+".pickle"

    #globally defining ctab since it is called in a previous function but never defined previously
    global ctab
    ctab=mcoeffs(mod,eta,dkm,dt,ntot)    
    #first do the dkm points that are exact and initialize data to this
    data=exact(eigl,dkm,ham,dt,initrho)
    #initialize augmented density tensor with the exact propagator to point k=dkm
    #the ellipses tells einsum to use array broadcasting to do the multiplication
    aug=einsum('ij...,ij...->ij...',initrho,initprop(eigl,dkm,dkm+1,ham,dt))
    #contracts first two indices to get rank-(dk-1) aug density tensor
    aug=einsum('ijk...->k...',aug)
    if mod==0:
        #standard quapi uses the same propagator at each step so these are stored before
        #propagation begins
        #define the propagor that takes aug from one point to the next
        prop=lamtens(eigl,dkm,dkm+1,dkm+2,ham,dt)
        #define the termination tensor
        term=lamtens(eigl,dkm,dkm+1,dkm+1,ham,dt)
    
        for j in range(ntot):
            t0=time.time()
            #first pad aug to the same size as term
            augN=expand_dims(expand_dims(aug,-1),-1)
            augN=repeat(repeat(augN,2,-1),2,-2)
            #multiply in the termination tensor
            augN=augN*term
            #successively contract indices until you are left with reduced density matrix
            for k in range(dkm):
                augN=einsum('ijk...->k...',augN)
            #print data to check its coming out
            #extract pop difference and append to data
            data.append([(j+1+dkm)*dt,(augN[0][0]-augN[1][1]).real])
            #pad aug ready for propagation one step forward
            aug=expand_dims(expand_dims(aug,-1),-1)
            aug=repeat(repeat(aug,2,-1),2,-2)
            #multiply in propagator and
            #contract last 2 indices to give new augmnented density tensor ready to be 
            #terminated in the next iteration of the for loop
            aug=einsum('ijk...->k...',aug*prop)
            print time.time()-t0
    else:
        #with the modified coeffs a different propagator is required at each step so these are constructed
        #with every iteration of the loop
        for j in range(ntot):
            t0=time.time()
            #first pad aug to the same size as term
            augN=expand_dims(expand_dims(aug,-1),-1)
            augN=repeat(repeat(augN,2,-1),2,-2)
            #multiply in the termination tensor
            augN=augN*lamtens(eigl,dkm,dkm+1+j,dkm+1+j,ham,dt)
            #successively contract indices until you are left with reduced density matrix
            for k in range(dkm):
                augN=einsum('ijk...->k...',augN)
            #print data to check its coming out
            #extract pop difference and append to data
            data.append([(j+1+dkm)*dt,(augN[0][0]-augN[1][1]).real])
            print [(j+1+dkm)*dt,(augN[0][0]-augN[1][1]).real]
            #pad aug ready for propagation one step forward
            aug=expand_dims(expand_dims(aug,-1),-1)
            aug=repeat(repeat(aug,2,-1),2,-2)
            #multiply in propagator
            #contract last 2 indices to give new augmnented density tensor ready to be 
            #terminated in the next iteration of the for loop
            aug=einsum('ijk...->k...',aug*lamtens(eigl,dkm,dkm+1+j,dkm+2+j,ham,dt))
            print time.time()-t0
        
            
    #pickles data for later use but also returns it if you want to use itimmediately
    datfilep=open(filename+str(dkm)+".pickle","w")
    pickle.dump(data,datfilep)
    datfilep.close()
    #deletes global variable ctab
    del ctab
    return data

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







