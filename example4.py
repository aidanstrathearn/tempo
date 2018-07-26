from SBM import spin_boson
from temposys_class import temposys
from numpy import array, exp
import matplotlib.pyplot as plt
from time import time
from mpmath import besselj,sinc, cos


#==============================================================================
###########################################################################################
#########  Fig.4(b) in TEMPO paper for R=40
###########################################################################################
######### takes ~30mins to run on EliteBook with i7 Core and 16GB RAM
###########################################################################################

#Set up the spin size and initial up state
s=1
Om=1
rho=array([[1,0],[0,0]])

#set cutoff frequency and temperature
wc=0.5
T=0.5
dd=1
al=2
#now set timestep and kmax 

Del=0.35
pp=40


#propagate system for 175 steps at lowest three values of coupling and plot Sz data
t0=time()
for R in [40]:
    def j1(w):
        jay=al*(w**dd)/(wc**(dd-1))*exp(-(w/wc))
        jay=jay*(1-[0,cos(R*w),besselj(0,R*w),sinc(R*w)][dd])
        return jay
    sbm,sz,sx=spin_boson(s,Om,rho,T,j1)
    sbm.convergence_params(dt=Del,prec=pp)
    sbm.prep()
    sbm.prop(175)
    datz=sbm.getopdat(sz)
    plt.plot(datz[0],datz[1])
print('total time: '+str(time()-t0))
plt.show()
#==============================================================================
