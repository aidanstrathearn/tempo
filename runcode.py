import newquaPy as qp
import numpy as np
import pickle
import lineshapes as ln

mod=0
#test hamiltonian
hamil=[[0,1],[1,0]]
#test timestep delta t
delt=0.1
#initial state
rho0=np.array([[1,0],[0,0]])
#cutoff frequency
wcut=4
#coupling strength
A=1.2
#delta_k_max
dkmax=5
#number of steps to propagate
nsteps=30
#creating ctab - for ohmic spectral density at 0 temperature with cutoff frequency
#and coupling strength A
#eigenvalues of coupled system operator
eigs=[-1,1]
T=3
mu=2

#defining the lineshape to be used in quapi from lineshapes.py
def eta(t):
        return ln.eta_sp_s3(t,T,wcut,mu,A)

print qp.quapi(mod,eigs,eta,dkmax,hamil,delt,rho0,nsteps,"filename")




print "\n pickle data: \n"
f=open("filename"+str(dkmax)+".pickle")
myf=pickle.load(f)
f.close()
print myf
