import math
import sys
import copy as cp
import numpy as np

# Declare the size parameters for MPS/MPO
# We'll set the actual values in the main program

#Local Hilbert/Liouville space dimension of a given MPS/MPO site
local_dim=None
#Bond dim of initial MPS 
bond_dim=None
#Bond dim of MPO
opdim=None

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
class mps_site:

      def __init__(self, south_north_dim = None, west_dim = None, east_dim = None, input_tensor = None):

           if (input_tensor == None):
               #define dimensions 
               self.SNdim = south_north_dim
               self.Wdim = west_dim
               self.Edim = east_dim
               #initialize tensor on each mps site (all zeros - just an example):
               self.m = np.zeros((self.SNdim, self.Wdim, self.Edim), dtype=complex)
          
           elif (south_north_dim != None) or (west_dim != None) or (east_dim != None):
               sys.exit("ERROR in mps_site - if input_tensor is provided, it must be the only input of mps_site. Exiting...")
           else:
               #define dimensions 
               self.SNdim = np.shape(input_tensor)[0]
               self.Wdim = np.shape(input_tensor)[1]
               self.Edim = np.shape(input_tensor)[2]
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
class mpo_site:

      def __init__(self, south_dim = None, north_dim = None, west_dim = None, east_dim = None, input_tensor = None):

         if (input_tensor == None):
            #define dimensions 
            self.Sdim = south_dim
            self.Ndim = north_dim
            self.Wdim = west_dim
            self.Edim = east_dim
            #initialize tensor on each mps site (all zeros - just an example):
            self.m = np.zeros((self.Sdim, self.Ndim, self.Wdim, self.Edim), dtype=complex)
         elif np.shape(np.shape(input_tensor)) != 4:
            sys.exit("ERROR in mpo_site - input_tensor must be a 4D object. Exiting...")
         elif (south_dim != None) or (north_dim != None) or (west_dim != None) or (east_dim != None):
            sys.exit("ERROR in mpo_site - if input_tensor is provided, it must be the only input of mpo_site. Exiting...")
         else:
            #define dimensions 
            self.Sdim = np.shape(input_tensor,1)
            self.Ndim = np.shape(input_tensor,2)
            self.Wdim = np.shape(input_tensor,3)
            self.Edim = np.shape(input_tensor,4)
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
class mpo_block(mpo_site, list):

      #the procedure w/ local_dim, op_dim should be an instance instead! (or not?)
      def __init__(self, opdim, N_sites, length_of_mps_block = None):
         
          #Record the length of mpo_block
          self.N_sites = N_sites

          #Note that Python numbers its lists from 0 to N-1!!!
          for site in np.arange(N_sites):

              if (length_of_mps_block == None) or (length_of_mps_block == N_sites):
                 if site == 0:
                    self.append(mpo_site(local_dim, local_dim, 1, opdim))
                 elif site == N_sites-1:
                    self.append(mpo_site(local_dim, local_dim, opdim, 1))
                 else:
                    self.append(mpo_site(local_dim, local_dim, opdim, opdim))
              else: 
                 if site == 0:
                    self.append(mpo_site(local_dim, local_dim, 1, opdim))
                 elif (site > length_of_mps_block-1) and (site < N_sites-1):
                    self.append(mpo_site(local_dim, 1, opdim, opdim))
                 elif site == N_sites-1:
                    self.append(mpo_site(local_dim, 1, opdim, 1))
                 else:
                    self.append(mpo_site(local_dim, local_dim, opdim, opdim))




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
class mps_block(mps_site, list):

      def __init__(self, bond_dim, N_sites):

          #Record the length of mps_block
          self.N_sites = N_sites

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
              self.append(mps_site(local_dim, west_dim, east_dim))


      def copy_mps_block(self, mps_copy, N_sites, copy_conjugate=False): 
          #Note that Python numbers its lists from 0 to N-1!!!
          if copy_conjugate==False: 
             for site in np.arange(N_sites):
                #copy the old mps_site data to the new mps_site
                SNdim = mps_copy[site].SNdim; Wdim = mps_copy[site].Wdim; Edim = mps_copy[site].Edim
                mps_copy[site].m = cp.deepcopy(self[site].m[0:SNdim, 0:Wdim, 0:Edim])
          if copy_conjugate==True: 
             for site in np.arange(N_sites):
                #copy the old mps_site data to the new mps_site
                SNdim = mps_copy[site].SNdim; Wdim = mps_copy[site].Wdim; Edim = mps_copy[site].Edim
                mps_copy[site].m = cp.deepcopy(np.conj(self[site].m[0:SNdim, 0:Wdim, 0:Edim]))


      def expand_mps_block(self, mps_expanded, N_sites): 
          #Note that Python numbers its lists from 0 to N-1!!!
          for site in np.arange(N_sites):
             #copy the old mps_site data to the new mps_site
             SNdim = self[site].SNdim; Wdim = self[site].Wdim; Edim = self[site].Edim
             mps_expanded[site].m[0:SNdim, 0:Wdim, 0:Edim] = cp.deepcopy(self[site].m)














