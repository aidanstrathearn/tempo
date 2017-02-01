from __future__ import print_function
import math
import sys
import copy as cp
import numpy as np
import ErrorHandling as err
from tensor_algebra import *
from MpsMpo_site_level_operations import mps_site, mpo_site


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
 def __init__(self, local_dim, opdim, N_sites):
         
    #Record the length of mpo_block
    self.N_sites = N_sites

    #initialize list of mpo_sites
    self.data = []

    #Note that Python numbers its lists from 0 to N-1!!!
    for site in range(N_sites):

       if site == 0:
            self.data.append(mpo_site(local_dim, local_dim, 1, opdim))
       elif site == N_sites-1:
            self.data.append(mpo_site(local_dim, local_dim, opdim, 1))
       else:
            self.data.append(mpo_site(local_dim, local_dim, opdim, opdim))

          
 def append_site(self, tensor_to_append):

    #Append a new site
    self.data.append(mpo_site(tens_in = tensor_to_append))
    self.N_sites = self.N_sites + 1 


 def reverse_mpo(self):

    self.data.reverse()

    for site in range(self.N_sites):
        MpoSiteT=np.transpose(self.data[site].m, (0,1,3,2))
        self.data[site].update_site(tens_in = MpoSiteT)



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
class mps_block():

 def __init__(self, local_dim, bond_dim, N_sites):

    #set the length of mps_block
    self.N_sites = N_sites

    #initialize list of mps_sites
    self.data = []

    #keep track of sites multiplied by MPO
    self.is_multiplied=[]

    #Note that Python numbers its lists from 0 to N-1!!!
    for site in range(N_sites):

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



 def append_site(self, tensor_to_append):

    try:
       if len(tens_in.shape) != 3: raise err.MpsSiteInputError
       if tensor_to_append.shape[1] != 1: raise err.MpsAppendingError

       #Append a new site
       self.data.append(mps_site(tens_in = tensor_to_append))
       self.N_sites = self.N_sites + 1 

    except err.MpsSiteInputError as e:
       print("append_site: ", e.msg)
       sys.exit()

    except err.MpsAppendingError as e:
       print("append_site: ", e.msg)
       sys.exit()



 def copy_mps(self, mps_copy, copy_conj=False): 

    #Initialize mps_copy
    mps_copy.data=[]

    #Note that Python numbers its lists from 0 to N-1!!!
    if copy_conj==False: 
       #copy the old mps_site data to the new mps_site
       for site in range(self.N_sites):
           mps_copy.data.append(mps_site(tens_in = self.data[site].m))

    elif copy_conj==True: 
       #copy the CONJ of old mps_site data to the new mps_site
       for site in range(self.N_sites):
           mps_copy.data.append(mps_site(tens_in = np.conj(self.data[site].m)))



 def contract_with_mps(self, other):
 
    mps_overlap = TensMul(self.data[0].m, other.data[0].m)

    for site in range(1,self.N_sites):
        tmpMpsMps = TensMul(self.data[site].m, other.data[site].m)
        mps_overlap = np.einsum('km,mn->kn', mps_overlap, tmpMpsMps)

    expval = mps_overlap[0,0]

    return expval



 def normalize_mps(self):

    self.copy_mps(mps_hc, 'conj')
    Cnorm = self.contract_with_mps(mps_hc)

    for site in range(self.N_sites):
        self.data[site].m = (Cnorm**(-0.5/self.N_sites)) * self.data[site].m




 def canonicalize_mps(self, orth_centre, prec, trunc_mode): 

    print('Canonicalizing MPS block')

    #Left sweep
    if (orth_centre > 0):
        self.left_sweep_mps(orth_centre, prec, trunc_mode)
   
    #Right sweep = [left sweep through the reversed mps_block]
    if (orth_centre < self.N_sites):
        #Reverse the mps_block, perform left sweep, reverse back
        self.reverse_mps()
        self.left_sweep_mps(self.N_sites - orth_centre + int(orth_centre != 0), prec, trunc_mode)
        self.reverse_mps()


 def reverse_mps(self):

    self.data.reverse()

    for site in range(self.N_sites):
        MpsSiteT=np.transpose(self.data[site].m, (0,2,1))
        self.data[site].update_site(tens_in = MpsSiteT)



 def left_sweep_mps(self, orth_centre, prec, trunc_mode): 

    for site in range(1,orth_centre):
        self.data[site-1].svd_mps_site(self.data[site], prec, trunc_mode)




 ########## Modes: 'accuracy', 'chi', 'fraction' ################################

 #Note the term "+ int(orth_centre != 0)" --> 
 #on [R->L] sweep we reverse Oc to N - Oc + 1, so that all bonds have been through svd 
 #(in particular, the bond to the right of Oc, which is skipped if we reverse Oc to N - Oc instead)
 #If Oc=0 --> reverse to N - Oc instead, cause there's no bond (and no sites) to the right of Oc (so nothing to svd)
 def contract_with_mpo(self, mpo_block, orth_centre=None, prec=0.01, trunc_mode='accuracy'):          

    #default val of orth_centre
    if orth_centre == None: orth_centre=int(np.ceil(0.5*self.N_sites))

    #Initialize a boolean list (must re-init each time!) 
    #to keep track which mps sites have been multiplied by the corresponding mpo sites
    self.is_multiplied=[]
    for i in range(self.N_sites):
        self.is_multiplied.append(False)

    if (orth_centre > 0):
        self.left_sweep_mps_mpo(mpo_block, orth_centre, prec, trunc_mode) 


    if (orth_centre < self.N_sites):
        self.reverse_mps_mpo_network(mpo_block) 
        self.left_sweep_mps_mpo(mpo_block, self.N_sites - orth_centre + int(orth_centre != 0), prec, trunc_mode)
        self.reverse_mps_mpo_network(mpo_block)

    #Canonicalize MPS 
    #(if Oc=N --> do backward sweep with Oc=0; if Oc=0 --> do backward sweep with Oc=N; else --> do both sweeps)
    if (orth_centre > 0): self.canonicalize_mps(0, prec, trunc_mode)
    if (orth_centre < self.N_sites): self.canonicalize_mps(self.N_sites, prec, trunc_mode)




 def reverse_mps_mpo_network(self, mpo_block): 

    self.reverse_mps() 
    mpo_block.reverse_mpo() 
    self.is_multiplied.reverse()



 def left_sweep_mps_mpo(self, mpo_block, orth_centre, prec, trunc_mode): 

    print('MULT at site ', 0)

    self.data[0].update_site(tens_in = TensMul(mpo_block.data[0].m, self.data[0].m))
    self.is_multiplied[0] = True

    for site in range(1,orth_centre):
        
        print('MULT & SVD at site ', site)

        if not self.is_multiplied[site]:
            self.data[site-1].zip_mps_mpo_sites(self.data[site], mpo_block.data[site], prec, trunc_mode)
            self.is_multiplied[site] = True
        else:
            print('SVD (no mult) at site ', site)
            self.data[site-1].svd_mps_site(self.data[site], prec, trunc_mode)




 def readout(self):

    ns=self.N_sites
    rh=np.einsum('ijk->ik',self.data[0].m)

    for jj in range(1,ns-1):
       rh=np.einsum('ij,jk',rh,np.einsum('ijk->jk',self.data[jj].m))   

    rh=np.einsum('ij,j',rh,np.einsum('ijk->j',self.data[ns-1].m))
    return rh


