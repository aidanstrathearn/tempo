#!/bin/env python

import math
import numpy as np
import copy as cp
from MpsMpo_block_level_operations import mpo_site, mps_site
from MpsMpo_block_level_operations import mpo_block, mps_block
import construction_of_mpo_and_mps as setup

run_aklt = True

### Specify the params ###
if run_aklt == True:
  #params for AKLT model
  N_sites=5; local_dim=3; bond_dim=2; opdim=13*(N_sites-1)
else:
  #params for AFM Heisenberg model
  N_sites=3; local_dim=2; bond_dim=2; opdim=6

### Initialize wavefunctions & Hamiltonian
hamiltonian=mpo_block(local_dim, opdim, N_sites) 
psi_ket=mps_block(local_dim, bond_dim, N_sites)
psi_bra=mps_block(local_dim, bond_dim, N_sites)

if run_aklt == True:
  #Construct AKLT Hamiltonian
  setup.construct_aklt_hamiltonian(hamiltonian, local_dim, opdim, N_sites)
  #Construct psi_ket and psi_bra
  setup.construct_aklt_ground_state_wavef(psi_ket, psi_bra, N_sites)
else:
  #Construct AFM Heisenberg Hamiltonian
  setup.construct_afm_heisenberg_hamiltonian(hamiltonian, local_dim, opdim, N_sites)
  #Construct psi_ket and psi_bra
  setup.construct_afm_heisenberg_ground_state_wavef_3site(psi_ket, psi_bra, N_sites)

#Compute normalization
c_norm = psi_ket.contract_with_mps(psi_bra)
print('c_norm = ', c_norm)

#Apply mpo Hamiltonian to mps Wavefunction 
psi_ket.contract_with_mpo(hamiltonian)
print('Mpo-Mps multiplication complete ')

#Obtain the energy eval
eval_psi = psi_ket.contract_with_mps(psi_bra)
print('Eigenvalue = ', eval_psi/c_norm)

if np.absolute(eval_psi/c_norm) > 0.001:
    print('Post-multiplication psi coefficient tensor: ')
    setup.verify_psi_coeff_tensor(psi_ket, (eval_psi/c_norm), local_dim, N_sites)


