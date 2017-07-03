#timestep
delt=0.2/7
#total number of steps to propagate to
ntot=60
#method - always fixed  precision meth=1
meth=1

#precision = 10**(-0.1*pp)
pp=[85]

#overall coupling=0.5*0.01*cvals = 0.5*raw_coupling, where raw_coupling = 0.01*cvals
coup=0.5*0.01
#typical range (90 - 120) corresponding to coupling (0.9 -1.2) = 0.01*cvals = raw_coupling
cvals=[99]

dkmax=[40]

for kk in pp:
    for ll in dkmax:
        for jj in cvals:
            name='coup'+str(jj)+'_dkm'+str(ll)+'_prec'+str(kk)
            paramfile=open('in_'+name+".txt","w")
            paramfile.write(str(delt)+ ' #delt \n')
            paramfile.write(str(ntot)+ ' #ntot \n')
            paramfile.write(str(meth)+ ' #method \n')
            paramfile.write(str(kk)+ ' #prec \n')
            paramfile.write(str(coup*jj)+ ' #coupling \n')
            paramfile.write(str(ll)+ ' #dkmax \n')
            paramfile.flush()
            paramfile.close()
    
            shfile=open('sh_'+name+".sh","w")
            shfile.write('#! /bin/bash'+'\n')
            shfile.write('#$ -S /bin/bash'+'\n')
            shfile.write('#$ -cwd'+'\n')
            shfile.write('#$ -pe smp 1'+'\n')
            shfile.write('#$ -j y'+'\n')
            shfile.write('export LD_LIBRARY_PATH=/share/apps/sage/local/lib'+'\n')
            shfile.write('~/anaconda3/bin/python ../tempo.py -i '+'in_'+name +".txt"+ ' -o ' +'out_'+name +".pickle")
            shfile.close()

