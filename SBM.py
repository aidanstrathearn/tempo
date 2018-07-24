from numpy import array, exp, diag, zeros, sqrt, dot, kron, eye, insert
from temposys_class import temposys

####################################################
############ The Spin Boson Model ##################
####################################################
#This function sets up a tempo system for the spin-(S/2) SBM, Eq.(4),
#for a bath at temperature T and with spectral density Jw as defined in Eq.(5)
#and initial spin state rho
#it also returns spin operators since we might want to find their expectations
def spin_boson(S,Om,rho,T,Jw):   
    #first define spin operators  Sz and Sx
    Sz=[S/2]
    while len(Sz)<S+1: Sz.append(Sz[-1]-1)
    Sz=diag(Sz)

    Sx=zeros((S+1,S+1))
    for jj in range(len(Sx)-1):
        Sx[jj][jj+1]=0.5*sqrt(0.5*S*(0.5*S+1)-(0.5*S-jj)*(0.5*S-jj-1))
    Sx=Sx+Sx.T

    #now set up the system
    #Hilbert space dimension=S+1
    system=temposys(S+1)
    #set the system name which labels output files
    system.set_filename('spinboson')
    #the symmetric hamiltoniain
    system.set_hamiltonian(Om*Sx)
    system.set_state(rho)
    #attach the bath
    #  "Let the Spin, see the Boson!" - Paddy McGuinness, hopefully
    system.add_bath(Sz,Jw,T)
    return system, Sz, Sx
