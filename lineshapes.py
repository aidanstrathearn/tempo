from scipy.special import gamma
from cmath import sin, atan, log
from mpmath import zeta,polygamma,harmonic,euler
from numpy import array,zeros

#lineshape for spectral density: A*w^s*e^(-w/wc) at temperature T
# zetas have poles at s=1,2 so need separate function for these cases
def eta_g(t,T,s,wc,A):
    eta=2j*wc**(s-1)*(1+t**2*wc**2)**(0.5-0.5*s)*sin((s-1)*atan(t*wc))
    eta=eta+T**(s-1)*(2*zeta(s-1,1+T*wc**(-1))+2*zeta(s-1,T*wc**(-1))-zeta(s-1,(1-1j*t*wc)*T*wc**(-1))-zeta(s-1,(1+T**(-1)*wc-1j*t*wc)*T*wc**(-1))-zeta(s-1,(1+1j*t*wc)*T*wc**(-1))-zeta(s-1,(1+T**(-1)*wc+1j*t*wc)*T*wc**(-1)))
    eta=0.5*A*gamma(s-1)*eta
    return eta

#same as above but only for s=1,2 - doesnt actually work yet  
def eta_12(t,T,s,wc,A):
    return -A*(-T)**s*T**(-1)*(polygamma(s-2,1+T*wc**(-1))+polygamma(s-2,T*wc**(-1))-polygamma(s-2,(1+T**(-1)*wc-1j*t*wc)*T*wc**(-1))-polygamma(s-2,(1+1j*t*wc)*T*wc**(-1)))

#the T=0 limit of eta_g.. should work for all s apart from s=1
def eta_0T(t,s,wc,A):
    return -1j*A*gamma(s)*wc**s*((1+1j*t*wc)**(-s)*(t*wc+1j*(-1+(1+1j*t*wc)**s))*(s-1)**(-1)*wc**(-1))

#T=0 and s=1
def eta_0T_s1(t,wc,A):
    return A*log(1+1j*t*wc)

#lineshape for super ohmic with spatial correlations: J=A*w^3*e^(-w/wc)*(1-sinc(w/mu))
def eta_sp_s3(t,T,wc,mu,A):
    return 0.5*A*T**2*(-2*wc**3*(-2j*t*mu**4+mu**2*wc+wc**3)*T**(-2)*(mu**2+wc**2)**(-2)-1j*mu*T**(-1)*(euler-harmonic(T*(-1j*mu**(-1)*(1+t*mu)+wc**(-1)))+polygamma(0,1+T*(-1j*t+1j*mu**(-1)+wc**(-1)))+2*polygamma(0,(mu-1j*wc)*T*(mu*wc)**(-1))-2*polygamma(0,(mu+1j*wc)*T*(mu*wc)**(-1))-polygamma(0,T*(mu*wc)**(-1)*(mu+1j*(-1+t*mu)*wc))+polygamma(0,T*(mu*wc)**(-1)*(mu+1j*(1+t*mu)*wc)))+4*polygamma(1,T*wc**(-1))-2*polygamma(1,T*wc**(-1)*(1+wc*T**(-1)-1j*t*wc))-2*polygamma(1,T*wc**(-1)*(1+1j*t*wc)))

#combines all of above for the general lineshape - still buggy though for some params
def eta_all(t,T,s,wc,mu,A):
    if T==0:
        if s==1:
            return eta_0T_s1(t,wc,A)
        else:
            return eta_0T(t,s,wc,A)
    else:
        if mu==0:
            if s==1 or s==2:
                return eta_12(t,T,s,wc,A)
            else:
                return eta_g(t,T,s,wc,A)
        else:
            return eta_sp_s3(t,T,wc,mu,A)

#function to calculate makri coeffs from lineshape - same as the one appearing in newquaPy            
def coeffs(mod,eta,dk,dt,ntot):
   
    etab=array([zeros((dk+1+ntot+dk),dtype=complex),zeros((dk+1+ntot+dk),dtype=complex),zeros((dk+1),dtype=complex)])
    #etab will be the table of makri coeffecients that the influencece functional uses
    # and has the format:
    #[[eta_kk, eta_dk1, eta_dk2...],           (mid evolution coeffs)
    #[eta_00, eta_01, eta_02...]               (end/start evolution coeffs)
    #[eta_00, eta_0N1, eta_0N2...]]            (coeffs connecting start point to end point)
    etab[0][0]=eta(dt)
    etab[1][0]=eta(dt*0.5)
    etab[2][0]=eta(dt*0.5)
    for j in range(dk):
        #calculating the coefficients by taking finite differences on the lineshape
        #as detailed in my report
        etab[0][j+1]=(eta((j+2)*dt)-eta((j+1)*dt)-eta((j+1)*dt)+eta((j*dt)))
        etab[1][j+1]=(eta((j+1.5)*dt)-eta((j+0.5)*dt)-eta((j+1)*dt)+eta(j*dt))
        etab[2][j+1]=(eta((j+1)*dt)-eta((j+0.5)*dt)-eta((j+0.5)*dt)+eta(j*dt))
    if mod==1:
        for j in range(ntot+dk):
            etab[0][dk+1+j]=(eta((j+1.5)*dt)-eta((j+0.5)*dt)-eta(dk*dt)+eta((dk-1)*dt))
            etab[1][dk+1+j]=(eta((j+1)*dt)-eta((j+0.5)*dt)-eta((dk-0.5)*dt)+eta((dk-1)*dt))
    else:
        for j in range(ntot+dk):
            etab[0][dk+1+j]=etab[0][dk]
            etab[1][dk+1+j]=etab[1][dk]
    return etab
