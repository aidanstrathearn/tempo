from numpy import array, exp
import matplotlib.pyplot as plt
from time import time
from SBM import spin_boson

###########################################################################################
######### Fig.2 in Makri & Makarov J. Chem. Phys 102, 4600, (1995) 
###########################################################################################
######### takes ~25secs to run on HP EliteBook with i7 Core and 16GB RAM
###########################################################################################

#This example is to highlight the main difference between TEMPO and QUAPI which
#is the introduction of a new convergence parmater: the singular value cutoff which in turn
#we control through varying the precision parameter.

#This reproduces Fig.2 of the Makri paper, dynamics which are seen to be 
#converged with dkmax=7, dt=0.25 We use the same timestep but 
#no memory cutoff and increase the precision to #show how convergence is acheived,
# similar to increasing dkmax to get convergence in QUAPI. 

#First we set up the system for a spin-1/2  (s=1) and set values taking account of
#factors of 0.5 between pauli and spin operators
s=1
Om=2
rho=array([[1,0],[0,0]])

#set the kondo parameter a, the cutoff frequency wc,the temperature T=1/beta
#and define the ohmic spectral density Jw
a=0.1
wc=7.5
T=0.2
def Jw(w):
    return 2*a*w*exp(-w/wc)

#set up the spin boson model and get spin operators
sbm,sz,sx=spin_boson(s,Om,rho,T,Jw)
sbm.convergence_params(dt=0.25)
#propagate for 100 steps using three different values of svd truncation precision
#and plot operator expectations to check for convergence
#can see convergence with pp=30 i.e. lambda_c=0.001*lambda_max
t0=time()
for pp in [10,20,30,40]:
    sbm.convergence_params(prec=pp)
    sbm.prep()
    sbm.prop(100)
    datz=sbm.getopdat(2*sz)
    plt.plot(datz[0],datz[1])
    #datx=sbm.getopdat(sx)
    #can also plot the Sx observable
    #plt.plot(datx[0],datx[1])
print('total time: '+str(time()-t0))
plt.show()

