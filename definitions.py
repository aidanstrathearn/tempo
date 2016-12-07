import math
import sys
import copy as cp
import numpy as np
from block_multiplication import lapack_multiply_each_site, arnoldi_multiply_each_site, sweep_over_free_mpo_sites

# Declare the size parameters for MPS/MPO
# We'll set the actual values in the main program

#Local Hilbert/Liouville space dimension of a given MPS/MPO site
#local_dim=None
#Bond dim of initial MPS 
#bond_dim=None
#Bond dim of MPO
#opdim=None

##########################################################################
#   Class mps_site    
# 
#   Attributes: 
#   SNdim = dimension of the South-North leg (i.e. 'local dim' of MPS site)
#   Wdim, Edim = dimensions of West & East legs (i.e. 'bond dims' of MPS site)
#
#   Synopsis:
#   Defines a single site of MPS - a rank-3 tensor with 3 legs
#
###########################################################################
class mps_site(object):

      def __init__(self, south_north_dim = None, west_dim = None, east_dim = None, input_tensor = None):

           if (input_tensor is None):
               #define dimensions 
               self.SNdim = south_north_dim
               self.Wdim = west_dim
               self.Edim = east_dim
               #initialize tensor on each mps site (all zeros - just an example):
               self.m = np.zeros((self.SNdim, self.Wdim, self.Edim), dtype=complex)
           elif len(input_tensor.shape) != 3:
               sys.exit("ERROR in mps_site - input_tensor must be a 3D object. Exiting...")
           elif (south_north_dim != None) or (west_dim != None) or (east_dim != None):
               sys.exit("ERROR in mps_site - if input_tensor is provided, it must be the only input of mps_site. Exiting...")
           else:
               #define dimensions 
               self.SNdim = input_tensor.shape[0]
               self.Wdim = input_tensor.shape[1]
               self.Edim = input_tensor.shape[2]
               #initialize tensor on each mps site = some input_tensor
               self.m = input_tensor

      def update_site(self, south_north_dim = None, west_dim = None, east_dim = None, input_tensor = None):

           if (input_tensor is None):
               #define dimensions 
               self.SNdim = south_north_dim
               self.Wdim = west_dim
               self.Edim = east_dim
               #initialize tensor on each mps site (all zeros - just an example):
               self.m = np.zeros((self.SNdim, self.Wdim, self.Edim), dtype=complex)
           elif len(input_tensor.shape) != 3:
               sys.exit("ERROR in mps_site - input_tensor must be a 3D object. Exiting...")
           elif (south_north_dim != None) or (west_dim != None) or (east_dim != None):
               sys.exit("ERROR in mps_site - if input_tensor is provided, it must be the only input of mps_site. Exiting...")
           else:
               #define dimensions 
               self.SNdim = input_tensor.shape[0]
               self.Wdim = input_tensor.shape[1]
               self.Edim = input_tensor.shape[2]
               #initialize tensor on each mps site = some input_tensor
               self.m = input_tensor

  
            



##########################################################################
#   Class mpo_site    
# 
#   Attributes: 
#   Sdim, Ndim = dimensions of the South & North legs 
#                      (i.e. 'local dims' of MPO site)
#   Wdim, Edim = dimensions of West & East legs (i.e. 'bond dims' of MPO site)
#
#   Synopsis:
#   Defines a single site of MPO - a rank-4 tensor with 4 legs
#
########################################################################### 
class mpo_site(object):

      def __init__(self, south_dim = None, north_dim = None, west_dim = None, east_dim = None, input_tensor = None):

         if (input_tensor is None):
            #define dimensions 
            self.Sdim = south_dim
            self.Ndim = north_dim
            self.Wdim = west_dim
            self.Edim = east_dim
            #initialize tensor on each mps site (all zeros - just an example):
            self.m = np.zeros((self.Sdim, self.Ndim, self.Wdim, self.Edim), dtype=complex)
         elif len(input_tensor.shape) != 4:
            sys.exit("ERROR in mpo_site - input_tensor must be a 4D object. Exiting...")
         elif (south_dim != None) or (north_dim != None) or (west_dim != None) or (east_dim != None):
            sys.exit("ERROR in mpo_site - if input_tensor is provided, it must be the only input of mpo_site. Exiting...")
         else:
            #define dimensions 
            self.Sdim = input_tensor.shape[0]
            self.Ndim = input_tensor.shape[1]
            self.Wdim = input_tensor.shape[2]
            self.Edim = input_tensor.shape[3]
            #initialize tensor on each mpo site = some input_tensor
            self.m = input_tensor


##########################################################################
#   Class mpo_block   
#
#   Synopsis:
#   Defines a block of MPO sites (i.e. a 4-leg tensor at each site)
#   with the total length of MPO block = N_sites.
#
#   Looping over all sites of the MPS block do the following:
#   Set Sdim,Ndim = local dimension of Hilbert/Liouville space at each site
#   Set Wdim,Edim = operator bond dim to the west & to the east of a site 
#   Note that operator bond dim = 1 at the ends of the MPO chain
########################################################################### 
class mpo_block(object):

      #the procedure w/ local_dim, op_dim should be an instance instead! (or not?)
      def __init__(self, local_dim, opdim, N_sites, length_of_mps_block = None):
         
          #Record the length of mpo_block
          self.N_sites = N_sites

          #initialize list of mpo_sites
          self.data = []

          #Note that Python numbers its lists from 0 to N-1!!!
          for site in np.arange(N_sites):

              if (length_of_mps_block is None) or (length_of_mps_block == N_sites):
                 if site == 0:
                    self.data.append(mpo_site(local_dim, local_dim, 1, opdim))
                 elif site == N_sites-1:
                    self.data.append(mpo_site(local_dim, local_dim, opdim, 1))
                 else:
                    self.data.append(mpo_site(local_dim, local_dim, opdim, opdim))
              else: 
                 if site == 0:
                    self.data.append(mpo_site(local_dim, local_dim, 1, opdim))
                 elif (site > length_of_mps_block-1) and (site < N_sites-1):
                    self.data.append(mpo_site(local_dim, 1, opdim, opdim))
                 elif site == N_sites-1:
                    self.data.append(mpo_site(local_dim, 1, opdim, 1))
                 else:
                    self.data.append(mpo_site(local_dim, local_dim, opdim, opdim))
          
      def append_site(self, tensor_to_append):
    #Append a new site
         self.data.append(mpo_site(input_tensor = tensor_to_append))
         self.N_sites = self.N_sites + 1 



##########################################################################
#   Class mps_block   
#
#   Synopsis:
#   Defines a block of MPS sites (i.e. a 3-leg tensor at each site)
#   with the total length of MPS block = N_sites.
#
#   Looping over all sites of the MPS block do the following:
#   Set SNdim = local dimension of Hilbert/Liouville space at each site
#   Set Wdim,Edim = the bond sizes to the west & to the east of a site 
#   
#   Since MPS is constructed by series of SVDs applied to the original state, 
#   the maximum (untruncated) bond size to the west of MPS site 
#   (appearing after SVD) can be: chi0 = min{d^{site}, d^(N-site)}. 
#   To avoid the exp growth of the bond size as we go further away from the ends
#   of MPS, we will truncate the bond size chi0 to the target bond size = bond_dim
#   (i.e. if chi0 > bond_dim, we truncate the full bond chi0 to bond_dim)
#   
#   Extra function - copy_mps_block:
#   Creates a copy of the MPS = mps_copy with length N_sites
#   if copy_conjugate = True, creates mps_copy = HC of the original MPS
#   else, creates mps_copy = original MPS 
#
#   CAVEAT: here, we assume that SNdim = uniform over all MPS sites
#   In the cases where this is not true, we'll need to make a slight 
#   modification to this part of the code. 
########################################################################### 
class mps_block(mpo_block):

      def __init__(self, local_dim, bond_dim, N_sites):

          #Record the length of mps_block
          self.N_sites = N_sites

          #initialize list of mpo_sites
          self.data = []

          #Note that Python numbers its lists from 0 to N-1!!!
          for site in np.arange(N_sites):

              #Distance from the chain end below which we should consider rescaling. 
              #We should not even try to calculate bond_dim**site if site is too far 
              #from the end as this might lead to an integer overflow.
              cutoff_distance = np.ceil(np.log(bond_dim)/np.log(local_dim))

              #Determine west_dim for each site
              west_distance = np.minimum(site, N_sites-site)
              if west_distance < cutoff_distance:
                     west_dim=np.minimum(bond_dim, local_dim**west_distance)
              else:
                     west_dim=bond_dim
              #Determine east_dim for each site
              east_distance = np.minimum(site+1, N_sites-(site+1))
              if east_distance < cutoff_distance:
                     east_dim=np.minimum(bond_dim, local_dim**east_distance)
              else:
                     east_dim=bond_dim
     
              #Create a new mps_site
              self.data.append(mps_site(local_dim, west_dim, east_dim))


      def copy_mps_block(self, mps_copy, N_sites, copy_conjugate=False): 
          #Note that Python numbers its lists from 0 to N-1!!!
          if copy_conjugate==False: 
             for site in np.arange(N_sites):
                #copy the old mps_site data to the new mps_site
                #SNdim = mps_copy.data[site].SNdim; Wdim = mps_copy.data[site].Wdim; Edim = mps_copy.data[site].Edim
                #mps_copy.data[site].m = cp.deepcopy(self.data[site].m[0:SNdim, 0:Wdim, 0:Edim])
                mps_copy.data[site].update_site(input_tensor = self.data[site].m)
          if copy_conjugate==True: 
             for site in np.arange(N_sites):
                #copy the old mps_site data to the new mps_site
                #SNdim = mps_copy.data[site].SNdim; Wdim = mps_copy.data[site].Wdim; Edim = mps_copy.data[site].Edim
                #mps_copy.data[site].update_site(input_tensor = np.conj(self.data[site].m[0:SNdim, 0:Wdim, 0:Edim]))
                mps_copy.data[site].update_site(input_tensor = np.conj(self.data[site].m))


      def expand_mps_block(self, mps_expanded, N_sites): 
          #Note that Python numbers its lists from 0 to N-1!!!
          for site in np.arange(N_sites):
             #copy the old mps_site data to the new mps_site
             SNdim = self.data[site].SNdim; Wdim = self.data[site].Wdim; Edim = self.data[site].Edim
             mps_expanded.data[site].m[0:SNdim, 0:Wdim, 0:Edim] = cp.deepcopy(self.data[site].m)

      def append_site(self, tensor_to_append):
          #Append a new site
          self.data.append(mps_site(input_tensor = tensor_to_append))
          self.N_sites = self.N_sites + 1 

      def readout(self):
          ns=self.N_sites
          rh=np.einsum('ijk->ik',self.data[0].m)
          for jj in range(1,ns-1):
            rh=np.einsum('ij,jk',rh,np.einsum('ijk->jk',self.data[jj].m))
   
          rh=np.einsum('ij,j',rh,np.einsum('ijk->j',self.data[ns-1].m))
          return rh
      ##############################################################################################################################################
      #
      #  multiply_block
      #
      #  Variables:
      #  mps_block1, mpo_block1 = input generic MPS/MPO blocks
      #  N_sites = length of MPS/MPO blocks
      #
      #  Synopsis:
      #  Sweep over all sites, multiplying MPS & MPO blocks site-by-site. 
      #  Near the ends of chain (smaller tensors) - use lapack SVD
      #  Deeper within the chain (larger tensors) - use arnoldi SVD 
      #
      #  As a final output, the original mps_block=MPS is changed to a
      #  new mps_block=MPS*MPO
      #
      #  Different modes in multiply_block: 'accuracy', 'chi', 'fraction'
      #
      #  Precision = required_accuracy in 'accuracy' mode
      #            = fraction of singular vals to calculate in 'fraction' mode 
      #            = bond_dim in 'chi' mode
      #
      #  eval_loop_start/end input only matters if which_mode='accuracy'
      #  otherwise, they're just set automatically.
      #
      #  Note that the actual eval_frac might be slightly different from the input (requested) eval_frac (e.g. 0.5555 or 0.6666 instead of 0.51)
      #  This is because eval_frac*nev_max is rounded to the nearest integer (sigma_dim = number of singular vals must be integer after all)
      #
      #  In mode = 'chi' we can set precision = None ---> the code will then use default values for each bond: chi = sdim_A, chi = opdim_A, etc
      #
      ##############################################################################################################################################
      def multiply_block(self, mpo_block, which_mode='accuracy', precision=0.3, delta_chi=1, eval_loop_start=None, eval_loop_end=None):
   
          #The code currently works in cases where MPS_length <= MPO_length
          if (self.N_sites > mpo_block.N_sites):
              sys.exit("ERROR in multiply_block: mpo_block should be equal to or longer than mps_block. Exiting...")

          #Sanity check of the input
          if (which_mode != 'accuracy') and (which_mode != 'chi') and (which_mode != 'fraction'):
              sys.exit("ERROR in multiply_block: which_mode must be 'accuracy', 'chi', or 'fraction'. Exiting...")

          #Sanity check of the input
          if (which_mode == "accuracy") or (which_mode == "fraction"):  
              #Note that both [fraction of evals] and [accuracy = ratio of smallest & largest sigma] must be between 0 and 1
              if (precision < 0) or (precision > 1) or (precision is None):
                 sys.exit("ERROR in multiply_block: precision must be between 0 and 1. Exiting...")
          elif (which_mode == "chi"):
             if not (precision is None):
                if not isinstance(precision,int) or not (precision > 0): 
                  sys.exit("ERROR in multiply_block: precision must be a positive integer or equal to None. Exiting...")
                
          #Sanity check of the input
          if which_mode == "accuracy":

             if not (eval_loop_start is None):
                if not isinstance(eval_loop_start,int) or not (eval_loop_start > 0):
                   sys.exit("ERROR in multiply_block: eval_loop_start must be a positive integer or equal to None. Exiting...")

             if not (eval_loop_end is None): 
                if not isinstance(eval_loop_end,int) or not (eval_loop_end > 0):
                   sys.exit("ERROR in multiply_block: eval_loop_end must be a positive integer or equal to None. Exiting...")


          #Check that ends of mps/mpo have bond_dim = 1
          if (self.data[0].Wdim != 1):
              sys.exit("ERROR in multiply_block: first site of MPS must have West dim = 1. Exiting...")
          if (mpo_block.data[0].Wdim != 1):
              sys.exit("ERROR in multiply_block: first site of MPO must have West dim = 1. Exiting...")
          #if (self.data[self.N_sites - 1].Edim != 1):
          #    sys.exit("ERROR in multiply_block: last site of MPS must have East dim = 1. Exiting...")
          #if (mpo_block.data[mpo_block.N_sites - 1].Edim != 1):
          #    sys.exit("ERROR in multiply_block: last site of MPO must have East dim = 1. Exiting...")

       
          #Note that Python numbers its lists from 0 to N-1!!!
          for site in np.arange(mpo_block.N_sites):

              #Verify that MPS/MPO have correct South & North dims 
              if (site < self.N_sites):
                 if (mpo_block.data[site].Ndim == 1):
                     print('Error at site = ', site)
                     sys.exit("ERROR in multiply_block: mpo_block has been set up incorrectly - should have North dim > 1. Exiting...")
                 if (mpo_block.data[site].Sdim == 1):
                     print('Error at site = ', site)
                     sys.exit("ERROR in multiply_block: mpo_block has been set up incorrectly - should have South dim > 1. Exiting...")
                 if (self.data[site].SNdim == 1):
                     print('Error at site = ', site)
                     sys.exit("ERROR in multiply_block: mpo_block has been set up incorrectly - should have South-North dim > 1. Exiting...")

              ############## PERFORM BLOCK MULTIPLICATION AND SVD #######################################################
    
              if (site == 0):
                  #Simple mult at site=0, no SVD
                  intermediate_mps = mps_site(self.data[site].SNdim, self.data[site].Wdim, self.data[site].Edim*mpo_block.data[site].Edim)
                  intermediate_mps = lapack_multiply_each_site(self, mpo_block, intermediate_mps, site, which_mode, precision, eval_loop_start, eval_loop_end)
              else:
                  if (site == 1) or (site == self.N_sites - 1):
                      #use lapack at site=1,N_sites-1
                      intermediate_mps = lapack_multiply_each_site(self, mpo_block, intermediate_mps, site, which_mode, precision, eval_loop_start, eval_loop_end)
                  elif (site > self.N_sites - 1):
                      #add the free mpo sites to mps
                      sweep_over_free_mpo_sites(self, mpo_block, site, which_mode, precision, eval_loop_start, eval_loop_end)
                  else:
                      #on other sites, use arnoldi by default, but switch to lapack if eval_frac is too large (see arnoldi_multiply_each_site function)
                      intermediate_mps = arnoldi_multiply_each_site(self, mpo_block, intermediate_mps, site, which_mode, precision, delta_chi, eval_loop_start, eval_loop_end)

              ######################################################################################################

          #Check that ends of output MPS have bond_dim = 1
          if (self.data[0].Wdim != 1):
              sys.exit("OUTPUT ERROR in multiply_block: first site of OUTPUT MPS must have West dim = 1. Exiting...")
          #if (self.data[self.N_sites - 1].Edim != 1):
          #   sys.exit("OUTPUT ERROR in multiply_block: last site of OUTPUT MPS must have East dim = 1. Exiting...")

          #return mps_block1










