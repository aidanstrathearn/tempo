import math
import numpy as np

# General size parameters for MPS/MPO:

#local dim
local_dim = 2
#standard MPS bond dim
bond_dim = 10
#standard MPO bond dim
op_dim = 4
#length of MPS block
N_sites = 3

# Define individual sites of mps_block
class mps_site:

      def __init__(self, south_north_dim, west_dim, east_dim):
            #define dimensions 
            self.SNdim = south_north_dim
            self.Wdim = west_dim
            self.Edim = east_dim
            #initialize tensor on each mps site (all zeros - just an example):
            self.m = np.zeros((self.SNdim, self.Wdim, self.Edim), dtype=complex)


# Define individual sites of mpo_block 
class mpo_site:

      def __init__(self, south_dim, north_dim, west_dim, east_dim):
            #define dimensions 
            self.Sdim = south_dim
            self.Ndim = north_dim
            self.Wdim = west_dim
            self.Edim = east_dim
            #initialize tensor on each mps site (all zeros - just an example):
            self.m = np.zeros((self.Sdim, self.Ndim, self.Wdim, self.Edim), dtype=complex)




# Combine all mps_sites to form an mps_block 
def setup_mps_block(mps_block, N_sites):

   #Note that Python numbers its lists from 0 to N-1!!!
   for site in np.arange(N_sites):
      if site == 0:
           mps_block.append(mps_site(local_dim, 1, bond_dim))
      elif site == N_sites-1:
           mps_block.append(mps_site(local_dim, bond_dim, 1))
      else:
           mps_block.append(mps_site(local_dim, bond_dim, bond_dim))



# Combine all mpo_sites to form an mpo_block 
def setup_mpo_block(mpo_block, N_sites):

   #Note that Python numbers its lists from 0 to N-1!!!
   for site in np.arange(N_sites):
      if site == 0:
           mpo_block.append(mpo_site(local_dim, local_dim, 1, bond_dim))
      elif site == N_sites-1:
           mpo_block.append(mpo_site(local_dim, local_dim, bond_dim, 1))
      else:
           mpo_block.append(mpo_site(local_dim, local_dim, bond_dim, bond_dim))


######################### Verify if we can access MPS/MPO data and sizes: ############################
       
#initialize new mps block
SOME_VEC = []     
setup_mps_block(SOME_VEC, N_sites)

print('This is MPS:')
print(' ')
#access data on each MPS site
print(SOME_VEC[0].m)
#access dimensions of each MPS site
print(SOME_VEC[0].SNdim)
print(SOME_VEC[0].Wdim)
print(SOME_VEC[0].Edim)
#access specific elements of the tensor on any MPS site
print(SOME_VEC[2].m[0,0,0])
print(' ')


#initialize new mpo block
SOME_OPERATOR = []
setup_mpo_block(SOME_OPERATOR, N_sites)

print('This is MPO:')
print(' ')
#access data on each MPO site
print(SOME_OPERATOR[0].m)
#access dimensions of each MPO site
print(SOME_OPERATOR[0].Sdim)
print(SOME_OPERATOR[0].Ndim)
print(SOME_OPERATOR[0].Wdim)
print(SOME_OPERATOR[0].Edim)
#access specific elements of the tensor on any MPO site
print(SOME_OPERATOR[2].m[0,0,0,0])
print(' ')













