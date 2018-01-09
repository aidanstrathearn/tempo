#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:21:15 2017

@author: dominic
"""

from scipy.special import gamma
from cmath import sin, atan, log, cos
from mpmath import zeta,polygamma,harmonic,euler,lerchphi,coth,exp,psi
from mpmath import gamma as cgamma
import mpmath as mp
import scipy
from numpy import array,zeros,pi,inf

def J_udbrownian(w,w0,G,al):
    return al/pi*G*w0**2*w/((w0**2-w**2)**2+G**2*w**2)
def J_odbrownian(w,wc,al):
    return al/pi*wc*w/(w**2+wc**2)

def J_g(w,T,s,wc,A):
    return A*w**s*exp(-w/wc)

