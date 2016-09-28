import cmath
import numpy as np
import scipy as scp

#ctab is the table of makri coeffecients that the influencece functional uses
# and has the format:
#[[eta_kk, eta_dk1, eta_dk2...],           (mid evolution coeffs)
#[eta_00, eta_01, eta_02...]               (end/start evolution coeffs)
#[eta_00, eta_0N1, eta_0N2...]]            (coeffs connecting start point to end point)
#Also, ctab is global and is the one thing that has to be
#predefined for the code to work
#currently no python code to actually calculate what these coefficients are
#but can import from mathematica or something in the meantime
ctab=[[1+2j,2+1j,4,3,3,3,3],[1+2j,2+1j,4,3,3,3,3],[1+2j,2+1j,4,3,3,3,3]]
#test hamiltonian
hamil=[[0,1],[1,0]]
#test timestep delta t
delt=1


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
    for sp in range(0,l):
        for sm in range(0,l):
            for sdp in range(0,l):
                for sdm in range(0,l):
                   kprop[sdp][sdm][sp][sm]=u[sp][sdp]*u.conjugate().transpose()[sdm][sm]
    return kprop

def icomp(sp,sm,sdp,sdm,dk,k,n):
    #gives a single component of discrete influence functional for a given
    #current point k, end point n, and memory span dk
    #bl is used to determine which row of ctab the makri coeffecients are taken from
    bl=int(bool(k==dk or k==n))+int(bool(k==dk==n))
    #phi is the influence phase and the exponential of this is returned
    phi=-(sp-sm)*(ctab[bl][dk]*sdp-ctab[bl][dk].conjugate()*sdm)
    return cmath.exp(phi)
    
    
def itab(eigl,dk,k,n):
    #explicitly constructs the influence functional factors Idk(sk,sdk)
    #eigl is the list of eigenvalues of the coupled system operator 
    #(these are the values that sk and sdk can take)
    #initializing the tensor
    tab=[[[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]]]]
    #the bool here is used when dk=0 and serves to remove dependence on 
    #the sdk coordinate since dk=0 piece of the influence functional
    #depends on sk
    bl=int(bool(dk==0))
    #l is the hilbert space dimension
    l=len(eigl)
    #loops through each element of tab and assigns it the correct influence
    #functional component with the bl part removing sdp/sdm where necessary
    for sp in range(0,l):
        for sm in range(0,l):
            for sdp in range(0,l):
                for sdm in range(0,l):
                    tab[sdp][sdm][sp][sm]=icomp(eigl[sp],eigl[sm],eigl[sdp-bl*(sdp-sp)],eigl[sdm-bl*(sdm-sm)],dk,k,n)      
    return tab

def itpad(eigl,dk,k,n):
    #takes the rank-2 itab tensors and gives them additional dummy indices
    #to make them rank-(dk+1) then puts indices in correct order
    #in numpy language the indices are called axes
    #first store the rank-2 tensor
    itp=itab(eigl,dk,k,n)
    #for dk=0,1 the tensor is already in the correct form so return it
    if dk==0 or dk==1:
        return np.array(itp)
    else:
        #the for loop nests the tensor in a list to give it an extra pair of 
        #indices at the 'front' (the left hand side)
        for k in range(0,dk-1):
            itp=[[itp,itp],[itp,itp]]
        #converting to a numpy array and then swapping the second last pair of
        #indices with the front pair before returning the tensor
        itp=np.array(itp)
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
    
def initprop(eigl,dkm,n):
    #not finished and not working
    #this function will eventually construct the initializing propagator
    #which is used to take the reduced density matrix at point 0
    #to the augmented density tensor at point delta_k_max
    #ready for propagation
    #essentially the same as the tensor growth part of tempo
    prop=lamtens(eigl,1,1,n)
    i0=np.swapaxes(itpad(0,1,itab(eigl,0,0,n)),0,2)
    i0=np.swapaxes(i0,1,3)
    prop=np.multiply(prop,i0)
    for j in range(0,dkm-1):
        prop=np.multiply(np.array([[prop,prop],[prop,prop]]),lamtens(eigl,j+2,j+2,n))
    return prop


#testing the lamda tensor function - this matches up exactly with 
#what i get using the same makri coeffs, hamiltonian and timestep
#in the mathematica code
#the test list of eigenvalues used is [-1,1], corresponding to the eigenvalues 
#of a sigma_z coupling to the environment
print lamtens([-1,1],3,4,5,hamil,delt)





