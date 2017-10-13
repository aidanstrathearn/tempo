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

#eigs,coup,rho,ham,dt,ntot,dkmax,prec = np.loadtxt(in_filename, unpack=True)

f=open(in_filename,"rb")
din=pickle.load(f)

#Check if params were loaded correctly
print("Params loaded from in_file: ",din)

def eta(t):
    return ln.eta_all(t,din[7][3],din[7][1],din[7][2],din[7][4],din[7][0])

#get tempo data
nmp.tempo(din[2],eta,din[1],din[0],din[3],din[4],din[5],din[6],datf=out_filename)
