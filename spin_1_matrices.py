import math
import numpy as np

#### CONTAINS DEFINITIONS OF SPIN-1 OPERATORS for Hamiltonian dynamics #####

#identity op
eye = np.array([[1., 0., 0.], [0.,1., 0.], [0., 0., 1.]])

#Spin-x,y,z operators
z = np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., -1.]])
x = np.sqrt(0.5)*np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
y = -1j*np.sqrt(0.5)*np.array([[0., 1., 0.], [-1., 0., 1.], [0., -1., 0.]])

#Spin-up,down operators
up = x + 1j*y
dn = x - 1j*y


