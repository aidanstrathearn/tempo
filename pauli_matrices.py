import math
import numpy as np

#### CONTAINS DEFINITIONS OF SPIN-1/2 OPERATORS for Hamiltonian dynamics #####

#identity op
eye = np.array([[1., 0.], [0.,1.]])

#Spin-x,y,z operators
z = 0.5*np.array([[1., 0.], [0.,-1.]])
x = 0.5*np.array([[0., 1.], [1., 0.]])
y = 0.5*np.array([[0., -1.0*1j], [1.0*1j, 0.]])

#Spin-up,down operators
up = x + 1j*y
dn = x - 1j*y


