#!/bin/env python

import math
import numpy as np
import definitions as defs
import block_multiplication as mult
import construction_of_mpo_and_mps as setup

run_aklt = True

### Specify the params ###
if run_aklt == True:
  #params for AKLT model
  N_sites=5; defs.local_dim=3; defs.bond_dim=5; defs.opdim=13*(N_sites-1)   #48
else:
  #params for AFM Heisenberg model
  N_sites=3; defs.local_dim=2; defs.bond_dim=2; defs.opdim=6


### Initialize wavefunctions & Hamiltonian
hamiltonian=defs.mpo_block(defs.opdim, N_sites) 
psi_ket=defs.mps_block(defs.bond_dim, N_sites)
psi_bra=defs.mps_block(defs.bond_dim, N_sites)

if run_aklt == True:
  #Construct AKLT Hamiltonian
  setup.construct_aklt_hamiltonian(hamiltonian, N_sites)
  #Construct psi_ket and psi_bra
  setup.construct_aklt_ground_state_wavef(psi_ket, psi_bra, N_sites)
else:
  #Construct AFM Heisenberg Hamiltonian
  setup.construct_afm_heisenberg_hamiltonian(hamiltonian, N_sites)
  #Construct psi_ket and psi_bra
  setup.construct_afm_heisenberg_ground_state_wavef_3site(psi_ket, psi_bra, N_sites)

#Compute normalization
c_norm = mult.contract_two_mps(psi_ket, psi_bra, N_sites)
print('c_norm = ', c_norm)
#Apply mpo Hamiltonian to mps Wavefunction 
mult.multiply_block(psi_ket, hamiltonian, N_sites)
print('Block multiplication complete ')
#Obtain the energy eval
eval_psi = mult.contract_two_mps(psi_ket, psi_bra, N_sites)
print('Eigenvalue = ', eval_psi/c_norm)

if np.absolute(eval_psi/c_norm) > 0.001:
    print('Post-multiplication psi coefficient tensor: ')
    setup.verify_psi_coeff_tensor(psi_ket, eval_psi/c_norm, N_sites)



