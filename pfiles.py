import pickle

hamil=[[0,1],[1,0]] 
eigs=[1,-1]
irho=[[1,0],[0,0]]
#timestep
delt=0.12/7
#total number of steps to propagate to
ntot=150
#precision = 10**(-0.1*pp)
pp=[85]
#overall coupling=0.5*0.01*cvals = 0.5*raw_coupling, where raw_coupling = 0.01*cvals
coup=0.5*0.01
s=1
wc=10
T=0
mu=0
#typical range (90 - 120) corresponding to coupling (0.9 -1.2) = 0.01*cvals = raw_coupling
cvals=[100]
dkmax=[100]
nc=1

for kk in pp:
    for ll in dkmax:
        for jj in cvals:
            indata=[hamil,irho,eigs,delt,ntot,ll,kk,[coup*jj,s,wc,T,mu]]
            name='coup'+str(jj)+'_dkm'+str(ll)+'_prec'+str(kk)
            paramfile=open('in_'+name+".pickle","wb")
            pickle.dump(indata,paramfile)
            paramfile.close()
             
            shfile=open('sh_'+name+".sh","w")
            shfile.write('#! /bin/bash'+'\n')
            shfile.write('#$ -S /bin/bash'+'\n')
            shfile.write('#$ -cwd'+'\n')
            shfile.write('#$ -pe smp '+str(nc)+'\n')
            shfile.write('#$ -j y'+'\n')
            shfile.write('export LD_LIBRARY_PATH=/share/apps/sage/local/lib'+'\n')
            shfile.write('~/anaconda3/bin/python ../tempo.py -i '+'in_'+name +".pickle"+ ' -o ' +'out_'+name +".pickle")
            shfile.close()
            
            #rfile=open('run_temp.sh',"w")
            #rfile.write('#!/bin/bash'+'\n')
            #rfile.write('declare -a kk_lis=(')
            #rfile.close()

'''
            name='coup'+str(jj)+'_dkm'+str(ll)+'_prec'+str(kk)
            paramfile=open('in_'+name+".txt","w")
            paramfile.write(str(eigs)+ ' #eigenvalues \n')
            paramfile.write(str(coup*jj)+ ' #overall coupling \n')
            paramfile.write(str(irho)+ ' #initial state\n')
            paramfile.write(str(len(hamil))+ ' #hilbert space dim\n')
            paramfile.write(str(delt)+ ' #timestep \n')
            paramfile.write(str(ntot)+ ' #number of data points \n')
            paramfile.write(str(ll)+ ' #dkmax \n')
            paramfile.write(str(kk)+ ' #precision \n')
            paramfile.flush()
            paramfile.close()
'''