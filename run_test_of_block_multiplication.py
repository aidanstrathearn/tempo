#!/bin/env python

import math
import numpy as np
import sys

import definitions as defs
import block_multiplication as mult
import construction_of_mpo_and_mps as setup


### Specify the params for AKLT model###
N_sites=5; defs.local_dim=3; defs.bond_dim=2; defs.opdim=13*(N_sites-1)

#Default value - required accuracy not achieved
accuracy_achieved = False
increment = 2
max_bond_dim = 10

### Initialize wavefunctions & Hamiltonian
hamiltonian = defs.mpo_block(defs.opdim, N_sites) 
psi_ket = defs.mps_block(defs.bond_dim, N_sites)
psi_bra = defs.mps_block(defs.bond_dim, N_sites)
psi_ket_cpy_original = defs.mps_block(defs.bond_dim, N_sites)
psi_bra_cpy_original = defs.mps_block(defs.bond_dim, N_sites)

#Construct AKLT Hamiltonian
setup.construct_aklt_hamiltonian(hamiltonian, N_sites)

#Construct psi_ket and psi_bra = aklt ground state
setup.construct_aklt_ground_state_wavef(psi_ket, psi_bra, N_sites)
#Create a copy of the original mps block
psi_ket.copy_mps_block(psi_ket_cpy_original, N_sites, False)
psi_bra.copy_mps_block(psi_bra_cpy_original, N_sites, False)

#Compute normalization
c_norm = mult.contract_two_mps(psi_ket, psi_bra, N_sites)
#Apply mpo Hamiltonian to mps Wavefunction 
mult.multiply_block(psi_ket, hamiltonian, N_sites)
accuracy_achieved = mult.multiply_block.accuracy_achieved

#Keep incrementing bond_dim till we reach the required accuracy or the maximum bond_dim allowed
while( (accuracy_achieved == False) and (defs.bond_dim <= max_bond_dim) ):

   #Increment bond dim during each iteration
   defs.bond_dim = defs.bond_dim + increment
   print('bond dim = ', defs.bond_dim)
   print('main program - required accuracy achieved? ', accuracy_achieved)
   #Define new MPS block with incremented bond_dim
   psi_ket = defs.mps_block(defs.bond_dim, N_sites)
   psi_bra = defs.mps_block(defs.bond_dim, N_sites)
   #Insert the original MPS block (defined outside the loop) 
   #into the new one with incremented bond_dim  
   psi_ket.insert_mps_block(psi_ket_cpy_original, N_sites)
   psi_bra.insert_mps_block(psi_bra_cpy_original, N_sites)
   #Compute normalization
   c_norm = mult.contract_two_mps(psi_ket, psi_bra, N_sites)
   #Apply mpo Hamiltonian to mps Wavefunction 
   mult.multiply_block(psi_ket, hamiltonian, N_sites)
   accuracy_achieved = mult.multiply_block.accuracy_achieved


if(accuracy_achieved == False):
    sys.exit("Failed to achieve the required accuracy - EXIT")


print('Block multiplication complete ')
#Obtain the energy eval
eval_psi = mult.contract_two_mps(psi_ket, psi_bra, N_sites)
print('c_norm = ', c_norm)
print('Eigenvalue = ', eval_psi/c_norm)

if np.absolute(eval_psi/c_norm) > 0.001:
    print('Post-multiplication psi coefficient tensor: ')
    setup.verify_psi_coeff_tensor(psi_ket, eval_psi/c_norm, N_sites)



