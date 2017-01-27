import math
import numpy as np
import sys
import copy as cp
import definitions
import pauli_matrices as Sigma
import spin_1_matrices as Spin
import block_multiplication as mult


#Construct the AKLT Hamiltonian
def construct_aklt_hamiltonian(mpo_hamiltonian, local_dim, opdim, N_sites):

  #pre-calculate spin terms:
  Sz_Sz = np.dot(Spin.z, Spin.z); Sup_Sup = np.dot(Spin.up, Spin.up); Sdn_Sdn = np.dot(Spin.dn, Spin.dn)
  Sz_Sup = np.dot(Spin.z, Spin.up); Sz_Sdn = np.dot(Spin.z, Spin.dn)
  Sup_Sz = np.dot(Spin.up, Spin.z); Sdn_Sz = np.dot(Spin.dn, Spin.z)
  Sup_Sdn = np.dot(Spin.up, Spin.dn); Sdn_Sup = np.dot(Spin.dn, Spin.up)

  term_cnt = -1

  for op_site in np.arange(N_sites-1):

     #First part of H:
     pre_fac=0.5
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Spin.up, Spin.dn, op_site, N_sites, True, 'spin1', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Spin.dn, Spin.up, op_site, N_sites, True, 'spin1', term_cnt)
     pre_fac=1.0
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Spin.z, Spin.z, op_site, N_sites, True, 'spin1', term_cnt)

     #Second part of H:
     pre_fac=1.0*(1.0/3.0)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sz_Sz, Sz_Sz, op_site, N_sites, True, 'spin1', term_cnt)
     pre_fac=0.25*(1.0/3.0)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sup_Sup, Sdn_Sdn, op_site, N_sites, True, 'spin1', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sdn_Sdn, Sup_Sup, op_site, N_sites, True, 'spin1', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sup_Sdn, Sdn_Sup, op_site, N_sites, True, 'spin1', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sdn_Sup, Sup_Sdn, op_site, N_sites, True, 'spin1', term_cnt)
     pre_fac=0.5*(1.0/3.0)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sz_Sup, Sz_Sdn, op_site, N_sites, True, 'spin1', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sz_Sdn, Sz_Sup, op_site, N_sites, True, 'spin1', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sup_Sz, Sdn_Sz, op_site, N_sites, True, 'spin1', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sdn_Sz, Sup_Sz, op_site, N_sites, True, 'spin1', term_cnt)

     #Third part of H - constant term: (set to 0.0 for non-zero gs eval - i.e. this is just an effective offset)
     pre_fac = 0.0*2.0/3.0
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Spin.eye, Spin.eye, op_site, N_sites, False, 'spin1', term_cnt)

  #(NB. initially we have zero terms in the mpo_hamiltonian - opcnt=0)
  #The last two-site term is: {op_cnt=N_sites-2, op_cnt=N_sites-1}



#Construct the AFM Heisenberg Hamiltonian
#Add two-site terms to the Hamiltonian
#Each step of the loop is adding an extra term to the sum
#O1 x O2 x I x I x I + I x O1 x O2 x I x I + I x I x O1 x O2 x I + ...
#Of two-site terms
def construct_afm_heisenberg_hamiltonian(mpo_hamiltonian, local_dim, opdim, N_sites):

  mpo_hamiltonian.data[1].m = np.zeros((local_dim, local_dim, opdim, opdim), dtype=complex)

  term_cnt = -1

  for op_site in np.arange(N_sites-1):
     pre_fac=0.5
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sigma.up, Sigma.dn, op_site, N_sites, True, 'pauli', term_cnt)
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sigma.dn, Sigma.up, op_site, N_sites, True, 'pauli', term_cnt)
     pre_fac=1.0
     term_cnt = mpo_add_extra_term(mpo_hamiltonian, pre_fac, Sigma.z, Sigma.z, op_site, N_sites, True, 'pauli', term_cnt)


#This function adds an extra term I x I x O1 x O2 x I to mpo_hamiltonian
#Each term I x I x O1 x O2 x I corresponds to an operator list
#op_list = {I,I,O1,O2,I}
def mpo_add_extra_term(mpo_hamiltonian, pre_fac, op1, op2, op_site, N_sites, is_two_site_term, which_spin, term_cnt):

   #Prepare op_list = {I,I,O1,O2,I} for a given term I x I x O1 x O2 x I
   op_list = prepare_operator_list(op1, op2, op_site, N_sites, is_two_site_term, which_spin)

   ensure_space_exists(mpo_hamiltonian.data[0], term_cnt)

   o = term_cnt+1
   op_list[0] = pre_fac*op_list[0]
   mpo_hamiltonian.data[0].m[:,:,0,o] = cp.deepcopy(op_list[0])
  
   for site in np.arange(1,N_sites-1): #FIXME change back (N-1 ---> if N_mpo = N_mps, N_mpo = N_mps + 1; N-2 ---> if N_mpo >= N_mps + 2;)
      mpo_hamiltonian.data[site].m[:,:,o,o] = cp.deepcopy(op_list[site])

   mpo_hamiltonian.data[N_sites-1].m[:,:,o,0] = cp.deepcopy(op_list[N_sites-1])

   #temp_op = op_list[N_sites-2]
   #mpo_hamiltonian.data[N_sites-2].m[:,0,o,o] = cp.deepcopy(temp_op[:,0]) #FIXME change back

   #temp_op = op_list[N_sites-1]
   #mpo_hamiltonian.data[N_sites-1].m[:,0,o,0] = cp.deepcopy(temp_op[:,0]) #FIXME change back

   term_cnt = o

   return term_cnt

#This function prepares a list of operators corresponding to a single term
#of the Hamiltonian: I x I x O1 x O2 x I
def prepare_operator_list(op1, op2, op_site, N_sites, is_two_site_term, which_spin):

   #by default, all elements in op_list = eye:
   op_list=[]
   for site in np.arange(N_sites):
       if which_spin == 'pauli':
           op_list.append(Sigma.eye)
       elif which_spin == 'spin1':
           op_list.append(Spin.eye)

   if is_two_site_term == True:
       #two-site terms with interactions:
       op_list[op_site] = op1
       op_list[op_site+1] = op2
   else:
       #single-site terms:
       op_list[op_site] = op1

   return op_list


#Check if mpo has enough space for an extra term
def ensure_space_exists(mpo, op_cnt):
    if (op_cnt + 1) > np.maximum(mpo.Edim, mpo.Wdim):
       sys.exit("Attempt to add too many elements to existing MPO")


#Construct the MPS form of the ground-state wavefunction in 3-site Heisenberg AFM 
def construct_afm_heisenberg_ground_state_wavef_3site(psi_ket, psi_bra, N_sites):
 
   #Construct psi_ket MPS 
   #using the exact matrices derived analytically:
   psi_ket.data[0].m[0,0,0:2] = np.array([1.0, 0.0]) 
   psi_ket.data[0].m[1,0,0:2] = np.array([0.0, 1.0]) 

   psi_ket.data[1].m[0,0:2,0:2] = np.array([[0.0, 1.0], [1.0, 0.0]]) 
   psi_ket.data[1].m[1,0:2,0:2] = np.array([[-2.0, 0.0], [0.0, 0.0]]) 

   psi_ket.data[2].m[0,0:2,0] = np.array([1.0, 0.0]) 
   psi_ket.data[2].m[1,0:2,0] = np.array([0.0, 1.0]) 

   #Construct psi_bra from psi_ket & normalize them:
   normalize_mps_blocks(psi_ket, psi_bra, N_sites)
  
   #Verify the psi_coefficient tensor of the ground state wavefunction
   print('Proof reading the initial psi coefficient tensor: ')
   local_dim = 2
   verify_psi_coeff_tensor(psi_ket, 1/np.sqrt(6.0), local_dim, N_sites) 
   #The quantum state psi coefficients should be: 
   #psi_tensor_ijk(:,:,:) = 0.0
   #psi_tensor_ijk(0,0,1) = 1.0 
   #psi_tensor_ijk(0,1,0) = -2.0) 
   #psi_tensor_ijk(1,0,0) = 1.0


def construct_aklt_ground_state_wavef(psi_ket, psi_bra, N_sites):

   #Construct psi_ket MPS 
   #using the exact matrices derived analytically:
   psi_ket.data[0].m[0,0,0:2] = np.array([0.0, np.sqrt(2.0/3.0)])
   psi_ket.data[0].m[1,0,0:2] = np.array([-np.sqrt(1.0/3.0), 0.0])
   psi_ket.data[0].m[2,0,0:2] = np.array([0.0, 0.0])

   for site in np.arange(1,N_sites-1):
      psi_ket.data[site].m[0,0:2,0:2] = np.array([[0.0, np.sqrt(2.0/3.0)], [0.0, 0.0]])
      psi_ket.data[site].m[1,0:2,0:2] = np.array([[-np.sqrt(1.0/3.0), 0.0], [0.0, np.sqrt(1.0/3.0)]])
      psi_ket.data[site].m[2,0:2,0:2] = np.array([[0.0, 0.0], [-np.sqrt(2.0/3.0), 0.0]])

   psi_ket.data[N_sites-1].m[0,0:2,0] = np.array([np.sqrt(2.0/3.0), 0.0])
   psi_ket.data[N_sites-1].m[1,0:2,0] = np.array([0.0, np.sqrt(1.0/3.0)])
   psi_ket.data[N_sites-1].m[2,0:2,0] = np.array([0.0, 0.0])

   #Construct psi_bra from psi_ket & normalize them:
   normalize_mps_blocks(psi_ket, psi_bra, N_sites)
   
   if (N_sites == 5) or (N_sites == 4) or (N_sites == 3):
       print('Proof reading the initial psi coefficient tensor: ')
       local_dim = 3
       verify_psi_coeff_tensor(psi_ket, 1.0, local_dim, N_sites)
       

   

      


#Find psi_bra from psi_ket, and normalize both wavefunctions
def normalize_mps_blocks(psi_ket, psi_bra, N_sites):
 
   #Get psi_bra from psi_ket
   psi_ket.copy_mps_block(psi_bra, N_sites, True)

   #Find normalization <psi_ket|psi_bra> = c_norm
   c_norm = mult.contract_two_mps(psi_ket, psi_bra)
   print('c_norm proofreading within: ', c_norm)

   #normalize psi_ket site-by-site
   for site in np.arange(N_sites):
       psi_ket.data[site].m = psi_ket.data[site].m/np.absolute(c_norm**(0.5/N_sites))
   #Get normalized psi_bra from psi_ket
   psi_ket.copy_mps_block(psi_bra, N_sites, True)
    


#Compute and print the psi coefficient tensor for verification 
#Computing psi_tensor amounts to multiplying together all MPS sites
#to produce a single tensor containing the coefficients of all quantum states:
#|Psi> = sum_ijk Psi_ijk |ijk>
def verify_psi_coeff_tensor(psi_mps_form, calibration_fac, local_dim, N_sites):

  if N_sites == 3:
     psi_tensor = np.zeros((local_dim, local_dim, local_dim), dtype=complex)
     for i in np.arange(local_dim):
       for j in np.arange(local_dim):
         for k in np.arange(local_dim):
            psi_tensor[i,j,k] = np.dot(psi_mps_form.data[0].m[i,0,:], np.dot(psi_mps_form.data[1].m[j,:,:], psi_mps_form.data[2].m[k,:,0]))/calibration_fac
  elif N_sites == 4:
     psi_tensor = np.zeros((local_dim, local_dim, local_dim, local_dim), dtype=complex)
     for i in np.arange(local_dim):
       for j in np.arange(local_dim):
         for k in np.arange(local_dim):
           for l in np.arange(local_dim):
              psi_tensor[i,j,k,l] = np.dot(psi_mps_form.data[0].m[i,0,:], np.dot(psi_mps_form.data[1].m[j,:,:], np.dot(psi_mps_form.data[2].m[k,:,:], psi_mps_form.data[3].m[l,:,0])))/calibration_fac
  elif N_sites == 5:
     psi_tensor = np.zeros((local_dim, local_dim, local_dim, local_dim, local_dim), dtype=complex)
     for i in np.arange(local_dim):
       for j in np.arange(local_dim):
         for k in np.arange(local_dim):
           for l in np.arange(local_dim):
             for n in np.arange(local_dim):
                 psi_tensor[i,j,k,l,n] = np.dot(psi_mps_form.data[0].m[i,0,:], np.dot(psi_mps_form.data[1].m[j,:,:], np.dot(psi_mps_form.data[2].m[k,:,:], np.dot(psi_mps_form.data[3].m[l,:,:], psi_mps_form.data[4].m[n,:,0]))))/calibration_fac
  
  #After the calculation, print out the psi_tensor
  if N_sites == 3:

    for i in np.arange(local_dim):
      for j in np.arange(local_dim):
        for k in np.arange(local_dim):

           if np.absolute(psi_tensor[i,j,k].imag) < 10**(-4):
                psi_tensor[i,j,k] = psi_tensor[i,j,k].real + 0j
           if np.absolute(psi_tensor[i,j,k].real) < 10**(-4):
                psi_tensor[i,j,k] = 0.0 + psi_tensor[i,j,k].imag*1j
            
           if np.absolute(psi_tensor[i,j,k]) > 0:
                print('At indices: ', i, j, k)
                print(psi_tensor[i,j,k])

  elif N_sites == 4:

    for i in np.arange(local_dim):
      for j in np.arange(local_dim):
        for k in np.arange(local_dim):
          for l in np.arange(local_dim):

             if np.absolute(psi_tensor[i,j,k,l].imag) < 10**(-4):
                 psi_tensor[i,j,k,l] = psi_tensor[i,j,k,l].real + 0j
             if np.absolute(psi_tensor[i,j,k,l].real) < 10**(-4):
                 psi_tensor[i,j,k,l] = 0.0 + psi_tensor[i,j,k,l].imag*1j
            
             if np.absolute(psi_tensor[i,j,k,l]) > 0:
                 print('At indices: ', i, j, k, l)
                 print(psi_tensor[i,j,k,l])

  elif N_sites == 5:

    for i in np.arange(local_dim):
      for j in np.arange(local_dim):
        for k in np.arange(local_dim):
          for l in np.arange(local_dim):
            for n in np.arange(local_dim):

               if np.absolute(psi_tensor[i,j,k,l,n].imag) < 10**(-4):
                   psi_tensor[i,j,k,l,n] = psi_tensor[i,j,k,l,n].real + 0j
               if np.absolute(psi_tensor[i,j,k,l,n].real) < 10**(-4):
                   psi_tensor[i,j,k,l,n] = 0.0 + psi_tensor[i,j,k,l,n].imag*1j
            
               if np.absolute(psi_tensor[i,j,k,l,n]) > 0.0:
                   print('At indices: ', i, j, k, l, n)
                   print(psi_tensor[i,j,k,l,n])
    




