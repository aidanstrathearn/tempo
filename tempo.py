import newquaPyVec as qp
import lineshapes as ln
import lambda_mpotest as lmpo
import numpy as np
import pickle
import copy
import time
import sys, getopt


#defininf the lineshape used to find the makri coeffs
def eta(t):
    return ln.eta_0T(t,3,1,0.5*2)
    
def eta2(t):
    return 10*t**2

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
delt, nsteps, meth, vals, modc, dkmax = np.loadtxt(in_filename, unpack=True) 

#convert input to integers 
nsteps = int(nsteps)
meth = int(meth)
vals = int(vals)
modc = int(modc)
dkmax = int(dkmax)

#Check if params were loaded correctly
print("Params loaded from in_file: ", delt, nsteps, meth, vals, modc, dkmax)


#some test parameters
hamil=[[0,1],[1,0]]
eigs=[-1,1]
hdim=len(eigs)
irho=[[1,0],[0,0]]

qp.trot=0

qp.ctab=qp.mcoeffs(modc,eta,dkmax,delt,6)

location="TEMPO"

#get tempo data
lmpo.tempo(modc,eigs,eta,dkmax,hamil,delt,irho,nsteps,location+"test",meth,vals)
#get quapi dat
qp.quapi(modc,eigs,eta,dkmax,hamil,delt,irho,nsteps,location+"tempocheck")

#test output file
out_file=open(out_filename,"w")
out_file.write("Completed OK")
out_file.close()

tdat=open(location+"test"+str(dkmax)+".pickle","rb")
mytdat=pickle.load(tdat,encoding='bytes')
tdat.close()
#dd=open(location+"test"+str(dkmax)+".dat","w")
out_file=open(out_filename,"a")
for k in range(0,len(mytdat)):
    out_file.write(str(mytdat[k][0])+" "+str(2*(mytdat[k][1][1]).real)+" "+str(2*(mytdat[k][1][1]).imag)+" "+str((mytdat[k][1][0]-mytdat[k][1][3]).real)+"\n")
out_file.close()

qdat=open(location+"tempocheck"+str(dkmax)+".pickle","rb")
myqdat=pickle.load(qdat,encoding='bytes')
qdat.close()
#dd=open(location+"tempocheck"+str(dkmax)+".dat","w")
out_file=open(out_filename,"a")
for k in range(0,len(myqdat)):
    out_file.write(str(myqdat[k][0])+" "+str(2*(myqdat[k][1][1]).real)+" "+str(2*(myqdat[k][1][1]).imag)+" "+str((myqdat[k][1][0]-myqdat[k][1][3]).real)+"\n")
out_file.close()

#comparing a datpoint- keeping all sv's reproduces quapi results
print( 'TEMPO data:')
print( mytdat[13][1])
print( 'QUAPI data:')
print( myqdat[13][1])
print( 'Trace of tempo data (bond truncuation seems to affect trace preservation and hermiticity):')
print( myqdat[13][1][0]+mytdat[13][1][3])
