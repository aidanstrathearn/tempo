from scipy.special import gamma
from cmath import sin, atan, log, cos
from mpmath import zeta,polygamma,harmonic,euler,lerchphi,coth,exp,psi
from mpmath import gamma as cgamma
import mpmath as mp
import scipy
from numpy import array,zeros,pi,inf


def lgam(x):
    return log(cgamma(x))

def sp3d_norm(mu,T,wc):
    d=1/mu
    return (-2*(d)**2*wc**6*(10+9*d**2*wc**2+3*d**4*wc**4)/(1+d**2*wc**2)**3 
            -1j*(T**3/d)*(polygamma(2,T/wc-1j*d*T)-polygamma(2,T/wc+1j*d*T))
            +2*T**4*polygamma(3,T/wc))


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
    return A*log(1+1j*wc*t)-1j*A*wc*t
    

#lineshape for super ohmic with spatial correlations: J=A*w^3*e^(-w/wc)*(1-sinc(w/mu))
def eta_sp_s3(t,T,wc,mu,A):
    return 0.5*A*T**2*(-2*wc**3*(-2j*t*mu**4+mu**2*wc+wc**3)*T**(-2)*(mu**2+wc**2)**(-2)
                       -1j*mu*T**(-1)*(euler-harmonic(T*(-1j*mu**(-1)*(1+t*mu)
                       +wc**(-1)))+polygamma(0,1+T*(-1j*t+1j*mu**(-1)+wc**(-1)))+2*polygamma(0,(mu-1j*wc)*T*(mu*wc)**(-1))-2*polygamma(0,(mu+1j*wc)*T*(mu*wc)**(-1))-polygamma(0,T*(mu*wc)**(-1)*(mu+1j*(-1+t*mu)*wc))+polygamma(0,T*(mu*wc)**(-1)*(mu+1j*(1+t*mu)*wc)))+4*polygamma(1,T*wc**(-1))-2*polygamma(1,T*wc**(-1)*(1+wc*T**(-1)-1j*t*wc))-2*polygamma(1,T*wc**(-1)*(1+1j*t*wc)))

#lineshape for ohmic with spatial correlations: J=A*w*e^(-w/wc)*(1-cos(w/mu))
def eta_sp_s1(t,T,wc,mu,A):
    return 0.5*A*(2*log(T/wc)+4*lgam(T/wc)
                 -lgam((T/wc)-1j*(T/mu))
                 -lgam((T/wc)+1-1j*(T/mu))
                 -lgam((T/wc)+1j*(T/mu))
                 -lgam((T/wc)+1+1j*(T/mu))
                 +lgam((T/wc)-1j*T*(-t+1/mu))
                 +lgam((T/wc)-1j*T*(-t-1/mu))
                 +lgam((T/wc)+1j*T*(-t+1/mu)+1)
                 +lgam((T/wc)-1j*T*(t+1/mu)+1)
                 -2*lgam((T/wc)-1j*T*t+1)
                 -2*lgam((T/wc)+1j*T*t)
                 )
                 
def neta_sp_s3(t,T,wc,mu,A):
    d=1/mu
    b=1/T
    return A*(d**2*wc**4*(-1-d**2*wc**2)/(1+d**2*wc**2)**2
            -1j*(1/(b*d))*polygamma(0,(1-1j*d*wc)/(b*wc))
            +1j*(1/(2*b*d))*(
                    2*polygamma(0,(1+1j*d*wc)/(b*wc))
                    -polygamma(0,(1+(b+1j*(d-t))*wc)/(b*wc))
                    +polygamma(0,(1-1j*(d-t)*wc)/(b*wc))
                    -polygamma(0,(1+1j*(d+t)*wc)/(b*wc))
                    +polygamma(0,(1+(b-1j*(d+t))*wc)/(b*wc)))
            -(1/b**2)*(
                    -2*polygamma(1,1/(b*wc))
                    +polygamma(1,(1+b*wc-1j*t*wc)/(b*wc))
                    +polygamma(1,(1+1j*t*wc)/(b*wc)))
            )
            

def eta_brownian(t,T,w0,al,G):
    b = 2*w0**2-G**2
    c = w0**4
    xi = (4*c-b**2)**0.5
    vn = 2*T*pi
    r1 = (b/2 + 1j/2*xi)**0.5
    r2 = (b/2 - 1j/2*xi)**0.5
    r1s = (-b/2 + 1j/2*xi)**0.5
    r2s = (-b/2 - 1j/2*xi)**0.5
    if t==0:
        eta = 0
    else:
        eta = -2*((-((-1 + coth(r1/(2.*T)))/r1**2) + (exp(1j*r1*t)*(-1 + coth(r1/(2.*T))))/r1**2 - (1 + coth(r2/(2.*T)))/r2**2 + 
        (1 + coth(r2/(2.*T)))/(exp(1j*r2*t)*r2**2) + t*((-1j*(-1 + coth(r1/(2.*T))))/r1 + 1j*((1 + coth(r2/(2.*T))))/r2))/
        (4.*xi) + T*(-(-(r2s**2*lerchphi(exp(-(t*vn)),1,(-r1s + vn)/vn))*exp(-t*vn) - 
        r2s**2*lerchphi(exp(-(t*vn)),1,(r1s + vn)/vn)*exp(-t*vn) + r1s**2*lerchphi(exp(-(t*vn)),1,(-r2s + vn)/vn)*exp(-t*vn) + 
        r1s**2*lerchphi(exp(-(t*vn)),1,(r2s + vn)/vn)*exp(-t*vn) + 2*r1s**2*log(1-exp(-t*vn)) - 
        2*r2s**2*log(1-exp(-t*vn)))/(2*r1s**2*(r1s - r2s)*r2s**2*(r1s + r2s)*vn) + 
        (t*(r2s*polygamma(0,-((-r1s - vn)/vn)) - r2s*polygamma(0,-((r1s - vn)/vn)) - r1s*polygamma(0,-((-r2s - vn)/vn)) + 
        r1s*polygamma(0,-((r2s - vn)/vn))))/(2.*r1s*r2s*(r1s**2 - r2s**2)*vn) - 
        (2*euler*r1s**2 - 2*euler*r2s**2 - r2s**2*polygamma(0,-((-r1s - vn)/vn)) - r2s**2*polygamma(0,-((r1s - vn)/vn)) + 
        r1s**2*polygamma(0,-((-r2s - vn)/vn)) + r1s**2*polygamma(0,-((r2s - vn)/vn)))/(2.*r1s**2*r2s**2*(r1s**2 - r2s**2)*vn)))
        eta = eta*G*w0**2*al
    print(t)
    return eta

#Time integrated part of the auto-correlation function
def fo(w,T,t,ct=0):
    #ct determines whether to include (1) counterterms or not (0)
    if ct==1:
        fo = w**(-2)*(coth(w/(2*T))*(1-cos(w*t))+1j*(sin(w*t)-w*t))
    else:
        fo = ((1-cos(w*t))+1j*(sin(w*t)))
    return fo

def numint(t,T,nin,ct=0): 
    numir = scipy.integrate.quad(lambda w: scipy.real(nin(w)*(fo(w,T,t,ct))),0,inf)
    numii = scipy.integrate.quad(lambda w: scipy.imag(nin(w)*(fo(w,T,t,ct))),0,inf)
    print(t)
    return numir[0]+1j*numii[0]

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
