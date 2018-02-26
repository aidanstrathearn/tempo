import os
import sys
import subprocess

#klist = [5,10,15]
#clist = [5,10,15,20,25,30,35,40,45,50]

klist = [5,10]
clist = [5,10]
pp = 80
cpu = 6
omp = int(cpu/2)

pps = str(pp)
cpus = str(cpu)
omps = str(omp)

for cc in clist:
    ccs = str(cc)
    dirname = 'coup'+ccs
    try:
        os.chdir(dirname)
    except FileNotFoundError:
        os.mkdir(dirname)
        os.chdir(dirname)
    
    
    for kk in klist:
        kks = str(kk)
        fname = 'a'+ccs+'k'+kks+'p'+pps+'.pbs'
        with open(fname, 'w') as script:
            script.write('#!/bin/bash --login \n')
            script.write('#PBS -l select=1:ncpus='+cpus+'\n')
            script.write('#PBS -l walltime=96:00:00 \n')
            script.write('#PBS -A d422 \n')
            
            script.write('\n')
            script.write('cd $PBS_O_WORKDIR \n')
            script.write('module load anaconda/python3 \n')
            script.write('export OMP_NUM_THREADS='+omps +'\n')
            script.write('\n')
            script.write('python ../../spin1_newmpo.py '+kks+' '+ccs+' '+pps)
            
        subprocess.call(['qsub', fname])
    os.chdir('..')
