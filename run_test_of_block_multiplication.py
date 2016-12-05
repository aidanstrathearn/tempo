#!/bin/env python

import math
import numpy as np
import copy as cp
import definitions as defs
#import block_multiplication as mult
from block_multiplication import contract_two_mps
import construction_of_mpo_and_mps as setup

run_aklt = True

### Specify the params ###
if run_aklt == True:
  #params for AKLT model
  N_sites=5; N_mpo_sites=N_sites; local_dim=3; bond_dim=2; opdim=13*(N_mpo_sites-1)     #13*(N_sites-1)   #48
else:
  #params for AFM Heisenberg model
  N_sites=3; local_dim=2; bond_dim=2; opdim=6

### Initialize wavefunctions & Hamiltonian
hamiltonian=defs.mpo_block(local_dim, opdim, N_mpo_sites, N_sites) 
psi_ket=defs.mps_block(local_dim, bond_dim, N_sites)
psi_bra=defs.mps_block(local_dim, bond_dim, N_sites)

if run_aklt == True:
  #Construct AKLT Hamiltonian
  setup.construct_aklt_hamiltonian(hamiltonian, local_dim, opdim, N_mpo_sites)
  #Construct psi_ket and psi_bra
  setup.construct_aklt_ground_state_wavef(psi_ket, psi_bra, N_sites)
else:
  #Construct AFM Heisenberg Hamiltonian
  setup.construct_afm_heisenberg_hamiltonian(hamiltonian, local_dim, opdim, N_sites)
  #Construct psi_ket and psi_bra
  setup.construct_afm_heisenberg_ground_state_wavef_3site(psi_ket, psi_bra, N_sites)

#Compute normalization
c_norm = contract_two_mps(psi_ket, psi_bra)
print('c_norm = ', c_norm)
#Save a copy of psi_ket before block mult
psi_ket_0 = cp.deepcopy(psi_ket)

#Apply mpo Hamiltonian to mps Wavefunction 
psi_ket.multiply_block(hamiltonian)
print('Block multiplication complete ')

#Get c_norm for mps_blocks with incremented bond dims
c_norm = contract_two_mps(psi_ket_0, psi_bra)

if (N_mpo_sites > N_sites):
   psi_bra = defs.mps_block(local_dim, bond_dim, N_mpo_sites)
   psi_ket.copy_mps_block(psi_bra, N_mpo_sites, copy_conjugate=True)

#Obtain the energy eval
eval_psi = contract_two_mps(psi_ket, psi_bra)
print('Eigenvalue = ', eval_psi/c_norm)

if np.absolute(eval_psi/c_norm) > 0.001:
    print('Post-multiplication psi coefficient tensor: ')
    setup.verify_psi_coeff_tensor(psi_ket, (eval_psi/c_norm), local_dim, N_mpo_sites)


#TESTING append_mps_site function - works OK, but currently disabled
#psi_ket.append_mps_site(tensor_to_append=np.zeros((3,1,1), dtype=complex) )
#for site in np.arange(psi_ket.N_sites):
    #print('psi_ket', psi_ket.data[site].m)















