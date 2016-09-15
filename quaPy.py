import cmath
import numpy as np
import scipy as scp


#Also, ctab is global and is the one thing that has to be
#predefined for the code to work
#currently no python code to actually calculate what these coefficients are
#but can import from mathematica or something in the meantime


#the lineshape (double integral over 2-time correlation) for ohmic SD at 0K 
#with cutoff frequency wc
def eta(t,wc):
    return cmath.log(1+1j*wc*t)


def makricoeffs(a,wc,dk,dt):
    #a is the coupling strength, the prefactor of the spectral density
    #initialize coefficients table
    etab=np.zeros((3,dk+1),dtype=complex)
    #etab will be the table of makri coeffecients that the influencece functional uses
    # and has the format:
    #[[eta_kk, eta_dk1, eta_dk2...],           (mid evolution coeffs)
    #[eta_00, eta_01, eta_02...]               (end/start evolution coeffs)
    #[eta_00, eta_0N1, eta_0N2...]]            (coeffs connecting start point to end point)
    etab[0][0]=a*eta(dt,wc)
    etab[1][0]=a*eta(dt*0.5,wc)
    etab[2][0]=a*eta(dt*0.5,wc)
    for j in range(dk):
        #calculating the coefficients by taking finite differences on the lineshape
        #as detailed in my report
        etab[0][j+1]=a*(eta((j+2)*dt,wc)-eta((j+1)*dt,wc)-eta((j+1)*dt,wc)+eta((j*dt),wc))
        etab[1][j+1]=a*(eta((j+1.5)*dt,wc)-eta((j+0.5)*dt, wc)-eta((j+1)*dt,wc)+eta(j*dt,wc))
        etab[2][j+1]=a*(eta((j+1)*dt,wc)-eta((j+0.5)*dt,wc)-eta((j+0.5)*dt,wc)+eta(j*dt,wc))
    return etab
    

def freeprop(ham,dt):
    #function to create the freepropagator factor of the lambda tensor
    #for a given hamiltonian ham and timestep dt
    #l is dimension of hilbert space
    l=len(ham)
    #initialize tensor
    kprop=np.zeros((l,l,l,l),dtype=complex)
    #define unitary time evol operator
    u=scp.linalg.expm(np.array(ham)*dt*(-1j))
    #loops through assigning the elements of the tensor from u and its conjugate transpose
    for sp in range(l):
        for sm in range(l):
            for sdp in range(l):
                for sdm in range(l):
                   kprop[sdp][sdm][sp][sm]=u[sp][sdp]*u.conjugate().transpose()[sdm][sm]
    return kprop

def icomp(sp,sm,sdp,sdm,dk,k,n):
    #gives a single component of discrete influence functional for a given
    #current point k, end point n, and memory span dk
    #bl is used to determine which row of ctab the makri coeffecients are taken from
    bl=int(k==dk or k==n)+int(k==dk==n)
    #phi is the influence phase and the exponential of this is returned
    phi=-(sp-sm)*(ctab[bl][dk]*sdp-ctab[bl][dk].conjugate()*sdm)
    return cmath.exp(phi)
    
def itab(eigl,dk,k,n):
    #explicitly constructs the influence functional factors Idk(sk,sdk)
    #eigl is the list of eigenvalues of the coupled system operator 
    #(these are the values that sk and sdk can take)
    #initializing the tensor
    #l is the hilbert space dimension
    l=len(eigl)
    tab=np.zeros((l,l,l,l),dtype=complex)
    #the bool here is used when dk=0 and serves to remove dependence on 
    #the sdk coordinate since dk=0 piece of the influence functional
    #depends on sk
    bl=int(dk==0)
    #loops through each element of tab and assigns it the correct influence
    #functional component with the bl part removing sdp/sdm where necessary
    for sp in range(l):
        for sm in range(l):
            for sdp in range(l):
                for sdm in range(l):
                    tab[sdp][sdm][sp][sm]=icomp(eigl[sp],eigl[sm],eigl[sdp-bl*(sdp-sp)],eigl[sdm-bl*(sdm-sm)],dk,k,n)      
    return tab    

def itpad(eigl,dk,k,n):
    #takes the rank-2 itab tensors and gives them additional dummy indices
    #to make them rank-(dk+1) then puts indices in correct order
    #in numpy language the indices are called axes
    #first store the rank-2 tensor
    l=len(eigl)
    itp=itab(eigl,dk,k,n)
    #for dk=0,1 the tensor is already in the correct form so return it
    if dk==0 or dk==1:
        return itp
    else:
        #expand_dims increases the depth of the array by one and repeat
        #copies the existing element of this new array level to give
        #the correct dimensions
        for k in range(0,dk-1):
            itp=np.expand_dims(np.expand_dims(itp,0),0)
            itp=np.repeat(np.repeat(itp,l,0),l,1)
        #swapping the second last pair of
        #indices with the front pair before returning the tensor
        itp=np.swapaxes(itp,2*(dk+1)-3,1) 
        itp=np.swapaxes(itp,2*(dk+1)-4,0)
        return itp
        
    
def lamtens(eigl,dkm,k,n,ham,dt):
    #constructs the lambda propagator tensor by multiplying
    #together the padded influence functional factors and finally
    #multiplying in the free propagator 
    #multiplying here means element-by-element multiplication
    tens=itpad(eigl,dkm,k,n)
    for j in range(0,dkm):
        tens=np.multiply(tens,itpad(eigl,dkm-j-1,k,n))
    tens=np.multiply(tens,freeprop(ham,dt))
    return tens
    
def initprop(eigl,dkm,n,ham,dt):
    #this function constructs the initializing propagator
    #which is used to take the reduced density matrix at point 0
    #to the augmented density tensor at point delta_k_max
    #ready for propagation
    #essentially the same as the tensor growth part of tempo
    #hilbert space dimension
    l=len(eigl)
    #initialise tensor
    prop=lamtens(eigl,1,1,n,ham,dt)
    #i0 is the k=0 self interaction influence functional factor
    i0=np.swapaxes(np.swapaxes(itpad(eigl,0,0,n),0,2),1,3)
    prop=np.multiply(prop,i0)
    #successively multiplies in lambda tensor to give the exact
    #influence functional between k=0 and k=dkm
    for j in range(0,dkm-1):
        prop=np.expand_dims(np.expand_dims(prop,-1),-1)
        prop=np.repeat(np.repeat(prop,l,-1),l,-2)
        prop=np.multiply(prop,lamtens(eigl,j+2,j+2,n,ham,dt))
    return prop

def exact(eigl,dkm,ham,dt,initrho):
    #propagates the first dkm steps where no memory cutoff is used
    #initialize data
    data=[[0,1]]
    for j in range(dkm):
        #multiplying in the initial reduced density matrix with einsum
        aug=np.einsum('ij...,ij...->ij...',initrho,initprop(eigl,j+1,j+1,ham,dt))
        for k in range(j+1):
            #contracting 2 indices at a time until what is left is the reduced
            #density matrix at point k=j
            aug=np.einsum('ijk...->k...',aug)
        #print data as check
        print [(j+1)*dt,(aug[0][0]-aug[1][1]).real]
        #appending the population difference of the red dens matrix and the time to data
        data.append([(j+1)*dt,(aug[0][0]-aug[1][1]).real])
    #returns the final data list, ready to have the next set of data appended
    return data

def quapi(eigl,dkm,ham,dt,initrho,ntot):
    #full algorithm to find data at dkm+ntot points
    #first do the dkm points that are exact and initialize data to this
    data=exact(eigl,dkm,ham,dt,initrho)
    #initialize augmented density tensor with the exact propagator to point k=dkm
    aug=np.einsum('ij...,ij...->ij...',initrho,initprop(eigl,dkm,dkm+1,ham,dt))
    #define the propagor that takes aug from one point to the next
    prop=lamtens(eigl,dkm,dkm+1,dkm+2,ham,dt)
    #define the termination tensor
    term=lamtens(eigl,dkm,dkm+1,dkm+1,ham,dt)
    for j in range(ntot):
        #first pad aug to the same size as term
        augN=np.expand_dims(np.expand_dims(aug,-1),-1)
        augN=np.repeat(np.repeat(augN,2,-1),2,-2)
        #multiply in the termination tensor
        augN=np.multiply(augN,term)
        #successively contract indices until you are left with reduced density matrix
        for k in range(dkm+1):
            augN=np.einsum('ijk...->k...',augN)
        #print data to check its coming out
        print [(j+1+dkm)*dt,(augN[0][0]-augN[1][1]).real]
        #extract pop difference and append to data
        data.append([(j+1+dkm)*dt,(augN[0][0]-augN[1][1]).real])
        #pad aug ready for propagation one step forward
        aug=np.expand_dims(np.expand_dims(aug,-1),-1)
        aug=np.repeat(np.repeat(aug,2,-1),2,-2)
        #multiply in propagator
        aug=np.multiply(aug,prop)
        #contract last 2 indices to give new augmnented density tensor ready to be 
        #terminated in the next iteration of the for loop
        aug=np.einsum('ijk...->k...',aug)
    #returns data for plotting or whatever
    return data
    
#test hamiltonian
hamil=[[0,1],[1,0]]
#test timestep delta t
delt=0.1
#initial state
rho0=np.array([[1,0],[0,0]])
#cutoff frequency
wcut=4
#coupling strength
A=1
#delta_k_max
dkmax=7
#creating ctab - for ohmic spectral density at 0 temperature with cutoff frequency
#and coupling strength A
ctab=makricoeffs(A,wcut,dkmax,delt)
#number of steps to propagate
nsteps=15
#eigenvalues of coupled system operator
eigs=[-1,1]

#testing the algorithm- data finally stored as dat - matches perfectly with the mathematica code
dat=quapi(eigs,dkmax,hamil,delt,rho0,nsteps)

