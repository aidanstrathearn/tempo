import newquaPyVec as qp
import numpy as np
import pickle
import lineshapes as ln

hamil=[[0,1],[1,0]]
rho0=np.array([[1,0],[0,0]])
wcut=1
A=0.5*wcut**(-2)*0.1
eigs=[-1,1]
T=1
dks=8
dk=8
ls=10
lf=10
location="C:\\Users\\admin\\Desktop\\phd\\paperplots\\newdat"
oper=[[0,0],[1,0]]

qp.trot=0
mod=1


for ll in range(ls,lf+1):
    for j in range(dks,dk+1):
        def eta(t):
            return ln.eta_g(t,0.2,1.2,5,0.01)
        qp.quapi(mod,eigs,eta,j,hamil,1,rho0,10*j,location+"\\mod"+str(mod)+"_a"+str(ll)+"_dk")


#def eta(t):
#            return ln.eta_g(t,1,3,1,0.5*0.5)

for ll in range(ls,lf+1):
    for j in range(dks,dk+1):
        f=open(location+"\\mod"+str(mod)+"_a"+str(ll)+"_dk"+str(j)+".pickle")
        myf=pickle.load(f)
        mathf=open(location+"\\mod"+str(mod)+"_a"+str(ll)+"_dk"+str(j)+".dat","w")
        for k in range(0,len(myf)):
            mathf.write(str(myf[k][0])+" "+str(2*(myf[k][1][1]).real)+" "+str(2*(myf[k][1][1]).imag)+" "+str((myf[k][1][0]-myf[k][1][3]).real)+"\n")
        mathf.close()
        f.close()

'''
for ll in range(ls,lf+1):
    
    for j in range(dks,dk+1):
        f=open(location+"\\mod"+str(mod)+"_a"+str(ll)+"_dk"+str(j)+".pickle")
        myf=pickle.load(f)
        mathf=open(location+"\\mod"+str(mod)+"_a"+str(ll)+"_dk"+str(j)+".dat","w")
        for k in range(0,len(myf)):
            mathf.write(str(myf[k][0])+" "+str((myf[k][1]).real)+"\n")
        mathf.close()
        f.close()
'''
