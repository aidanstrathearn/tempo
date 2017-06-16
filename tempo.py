import newquaPyVec as qp
import lineshapes as ln
import newmpo as nmp
import numpy as np
import pickle
import sys, getopt

# Store input and output file names
in_filename=''
out_filename=''
 
# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"i:o:")
 
###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-i':
        in_filename=a
    elif o == '-o':
        out_filename=a
    else:
        print("Correct usage: %s -i input -o output" % sys.argv[0])
 
# Display input and output file name passed as the args
print ("Input file : %s and output file: %s" % (in_filename, out_filename) )


# Load data from in_file
delt, nsteps, meth, vals, coup, dkmax = np.loadtxt(in_filename, unpack=True) 

#convert input to integers 
nsteps = int(nsteps)
meth = int(meth)
dkmax = int(dkmax)
vals= int(vals)

#Check if params were loaded correctly
print("Params loaded from in_file: ", delt, nsteps, meth, vals, coup, dkmax)

#to go to larger spin
#hamil=[[0,1,0],[1,0,1],[0,1,0]]
#eigs=[-1,0,1]
#irho=[[1,0,0],[0,0,0],[0,0,0]]
#

hamil=[[0,1],[1,0]]
eigs=[-1,1]
hdim=len(eigs)
irho=[[1,0],[0,0]]
qp.trot=0
modc=0
def eta(t):
    return ln.eta_0T_s1(t,10,coup)
qp.ctab=qp.mcoeffs(modc,eta,dkmax,delt,nsteps)

#get tempo data
nmp.tempoalg(modc,eigs,dkmax,eta,hamil,delt,irho,nsteps,meth,10**(-0.1*vals),out_filename)

'''
tdat=open(location+"test"+str(dkmax)+".pickle","rb")
mytdat=pickle.load(tdat,encoding='bytes')
tdat.close()
#dd=open(location+"test"+str(dkmax)+".dat","w")
out_file=open(out_filename,"a")
for k in range(0,len(mytdat)):
    out_file.write(str(mytdat[k][0])+" "+str(2*(mytdat[k][1][1]).real)+" "+str(2*(mytdat[k][1][1]).imag)+" "+str((mytdat[k][1][0]-mytdat[k][1][3]).real)+"\n")
out_file.close()
'''
