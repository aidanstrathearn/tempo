#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:21:15 2017

@author: dominic
"""

from cmath import sin, log, cos
from mpmath import exp, sinc
from numpy import pi

def J_UDbrownian(w,w0,G,al):
    return al*G*w0**2*w/((w0**2-w**2)**2+G**2*w**2)

def J_ODbrownian(w,wc,al):
    return al*wc*w/(w**2+wc**2)

def J_structured(w,w0,G,al_UD,wc,al_OD):
    J = J_ODbrownian(w,wc,al_OD)+J_UDbrownian(w,w0,G,al_UD)
    return J

'''
spectral density with arbitrary number of Lorentzian peaks located at w0s with
couplings als and widths Gs
'''
def J_lorentzian(w,w0s,Gs,als):
    J = 0
    for i in range(len(w0s)):
        w0 = w0s[i]
        G = Gs[i]
        al = als[i]
        J += 1/pi*al*0.5*G/((w-w0)**2 + 0.25*G**2)
    return J


def J_g(w,s,wc,A):
    return A*w**s*exp(-w/wc)

#Adolphs and Renger spectral density
def J_AR(w,S0,Sh,w1,w2,wh,G):
    s1 = w**5*S0*(6.105*10**-5*exp(-(w/w1)**0.5)/w1**4 + 3.8156*10**-5*exp(-(w/w2)**0.5)/w2**4)
    s2 = Sh*w**2*0.5*G/((w-wh)**2 + 0.25*G**2)
    return s1+s2