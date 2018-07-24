from SBM import spin_boson
from temposys_class import temposys
from numpy import array, exp, diag, zeros, sqrt, dot, kron, eye, insert
import matplotlib.pyplot as plt
from time import time
from mpmath import besselj,sinc, cos

#==============================================================================
###########################################################################################
#########  Fig.2(a) in TEMPO paper
###########################################################################################
######### takes ~250secs to run on EliteBook with i7 Core and 16GB RAM
###########################################################################################

#this is to show that, although we had to ensure the data in Fig.3(a) of the paper was very 
#well converged to do scaling analysis near the critical point, we can obtain 
#qualitavely (and pretty much quantitively) correct dynamics extremely easily at a lower level of convergence
#

#Set up the spin size and initial up state
s=1
Om=1
rho=array([[1,0],[0,0]])

#set cutoff frequency and temperature
wc=5
T=0

#now set timestep and kmax - the timestep is larger and the cutoff tau_c smaller than
#used in paper so less converged -- better convergence needed for scaling analysis
Del=0.06
K=50
pp=60

#propagate system for 120 steps at lowest three values of coupling and plot Sz data
t0=time()
for a in [0.1, 0.3, 0.7, 1, 1.2, 1.5]:
    def j1(w):
        return 2*a*w*exp(-w/wc)
    sbm,sz,sx=spin_boson(s,Om,rho,T,j1)
    sbm.convergence_params(Del,K,pp)
    sbm.prep()
    sbm.prop(300)
    datz=sbm.getopdat(sz)
    plt.plot(datz[0],datz[1])
print('total time: '+str(time()-t0))
plt.show()
#==============================================================================


