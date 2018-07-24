from temposys_class import temposys
from numpy import array, exp, diag, zeros, sqrt, dot, kron, eye, insert
import matplotlib.pyplot as plt
from time import time
from mpmath import besselj,sinc, cos

#==============================================================================
###########################################################################################
######### Example 2 - Cavity-Dot-Phonons: Damped Rabi Oscillations
###########################################################################################
######### takes ~90secs to run on EliteBook with i7 Core and 16GB RAM
###########################################################################################
#A more complicated example - a 2-level system coupled to both a single oscillator and a bath
#Could model a quantum dot with exciton-phonon interactions placed in a cavity
#We use TEMPO to model the phonon bath but treat the cavity mode as part of the reduced system

#This example is to demonstrate how TEMPO can easily deal with a relatively large reduced 
#system (8 states) without needing a memory cutoff - impossible using standard QUAPI!!

#set maximum number of cavity excitations
Ncav=3

#define creation and number operators of cavity - note tensor product with identity matrix
#since hilbert space is product: (cavity x 2-level system)
def cr(Nmodes):
    cre=zeros((Nmodes+1,Nmodes+1))
    for jj in range(Nmodes):
        cre[jj][jj+1]=sqrt(jj+1)
    return kron(cre.T,eye(2))

def num(Nmodes):
    return dot(cr(Nmodes),cr(Nmodes).T)

#define 2-level system operators - again tensor producting with identity
signum=kron(eye(Ncav+1),array([[1,0],[0,0]]))
sigp=kron(eye(Ncav+1),array([[0,1],[0,0]]))

#the dimension of the (cavity x 2-level system) hilbert space
hil_dim=(Ncav+1)*2

#set hamiltonian parameters - cavity frequency w_cav, coupling g, 2-level system splitting ep
w_cav=0.2
g=1
ep=0

#set up the hamiltonian - sticking with a Jaynes-Cummings number conserving interaction
hami=ep*signum + g*(dot(cr(Ncav).T,sigp)+dot(cr(Ncav),sigp.T)) + w_cav*num(Ncav)

#set phonon bath paramaters
wc=1
T=0.0862*4
a=0.5

#set superohmic spectral density with gaussian decay - standard QD-phonon spectral density
def jay(w):
    return a*w**3*exp(-(w/(2*wc))**2)

#start cavity in ground state
rho_cav=diag(insert(zeros(Ncav),0,1))
#start QD in excited state
rho_dot=array([[1,0],[0,0]])
#take product for overall initial state
rho=kron(rho_cav,rho_dot)
#NOTE: we have number conserving hamiltonian and initial state has only 1 excitation
#so really we only need Ncav=1 but lets stick with Ncav=3 for safety

#set up the tempo system with dimension, hamiltonian and initial state
cdp=temposys(hil_dim)
cdp.set_hamiltonian(hami)
cdp.set_state(rho)

#attach the bath to the 2-level system and set timestep and precision
cdp.add_bath(signum,jay,T)
cdp.convergence_params(dt=0.125,prec=50)

#prepare and propagate system for 100 steps
t0=time()
cdp.prep()
cdp.prop(100)
print('total time: '+str(time()-t0))

#get data for QD population, cavity population and their sum
#their sum should be 1 for all times because of number conservation
datz=cdp.getopdat(signum)
datnum=cdp.getopdat(num(Ncav))
datTnum=cdp.getopdat(num(Ncav)+signum)

plt.plot(datz[0],datz[1])
plt.plot(datnum[0],datnum[1])
plt.plot(datTnum[0],datTnum[1])
plt.show()
#==============================================================================


