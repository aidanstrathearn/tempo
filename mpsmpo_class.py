from __future__ import print_function
import sys
import numpy as np
import ErrorHandling as err
from svd_functions import *

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

 def __init__(self, Sdim = None, Ndim = None, Wdim = None, Edim = None, tens_in = None):

    if (tens_in is None):
        #define dims & init mpo_site tensor to all-zeros array
        self.Sdim = Sdim; self.Ndim = Ndim; self.Wdim = Wdim; self.Edim = Edim
        self.m = np.zeros((self.Sdim, self.Ndim, self.Wdim, self.Edim), dtype=complex)

    else:
        try:
           if len(tens_in.shape) != 4: raise err.MpoSiteShapeError
           if (Sdim is not None) or (Ndim is not None) or (Wdim is not None) or (Edim is not None): raise err.MpoSiteInputError

           #get dims from tens_in & set mpo_site to tens_in 
           self.Sdim = tens_in.shape[0]; self.Ndim = tens_in.shape[1]
           self.Wdim = tens_in.shape[2]; self.Edim = tens_in.shape[3]
           self.m = tens_in

        except err.MpoSiteShapeError as e: 
           print("mpo_site: ", e.msg)
           sys.exit()

        except err.MpoSiteInputError as e:
           print("mpo_site: ", e.msg)
           sys.exit()

 def update_site(self, Sdim = None, Ndim = None, Wdim = None, Edim = None, tens_in = None):

    if (tens_in is None):
        #define dims & set mpo_site tensor to all-zeros array
        self.Sdim = Sdim; self.Ndim = Ndim; self.Wdim = Wdim; self.Edim = Edim
        self.m = np.zeros((self.Sdim, self.Ndim, self.Wdim, self.Edim), dtype=complex)

    else:
        try:
           if len(tens_in.shape) != 4: raise err.MpoSiteShapeError
           if (Sdim is not None) or (Ndim is not None) or (Wdim is not None) or (Edim is not None): raise err.MpoSiteInputError

           #get dims from tens_in & set mpo_site to tens_in 
           self.Sdim = tens_in.shape[0]; self.Ndim = tens_in.shape[1]
           self.Wdim = tens_in.shape[2]; self.Edim = tens_in.shape[3]
           self.m = tens_in

        except err.MpoSiteShapeError as e: 
           print("mpo: update_site: ", e.msg)
           sys.exit()

        except err.MpoSiteInputError as e:
           print("mpo: update_site: ", e.msg)
           sys.exit()

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

 def __init__(self, SNdim = None, Wdim = None, Edim = None, tens_in = None):

    if (tens_in is None):
        #define dims & init mpo_site tensor to all-zeros array
        self.SNdim = SNdim; self.Wdim = Wdim; self.Edim = Edim
        self.m = np.zeros((self.SNdim, self.Wdim, self.Edim), dtype=complex)

    else:
        try:
           if len(tens_in.shape) != 3: raise err.MpsSiteShapeError
           if (SNdim is not None) or (Wdim is not None) or (Edim is not None): raise err.MpsSiteInputError

           #get dims from tens_in & set mps_site to tens_in 
           self.SNdim = tens_in.shape[0]; self.Wdim = tens_in.shape[1]; self.Edim = tens_in.shape[2]
           self.m = tens_in

        except err.MpsSiteShapeError as e: 
           print("mps_site: ", e.msg)
           sys.exit()

        except err.MpsSiteInputError as e:
           print("mps_site: ", e.msg)
           sys.exit()

 def update_site(self, SNdim = None, Wdim = None, Edim = None, tens_in = None):

    if (tens_in is None):
        #define dims & init mpo_site tensor to all-zeros array
        self.SNdim = SNdim; self.Wdim = Wdim; self.Edim = Edim
        self.m = np.zeros((self.SNdim, self.Wdim, self.Edim), dtype=complex)

    else:
        try:
           if len(tens_in.shape) != 3: raise err.MpsSiteShapeError
           if (SNdim is not None) or (Wdim is not None) or (Edim is not None): raise err.MpsSiteInputError

           #get dims from tens_in & set mps_site to tens_in 
           self.SNdim = tens_in.shape[0]; self.Wdim = tens_in.shape[1]; self.Edim = tens_in.shape[2]
           self.m = tens_in

        except err.MpsSiteShapeError as e: 
           print("mps: update_site: ", e.msg)
           sys.exit()

        except err.MpsSiteInputError as e:
           print("mps: update_site: ", e.msg)
           sys.exit()

 def svd_mps_site(self, other, prec, trunc_mode): 

    #Set dims of theta & construct theta matrix
    dimT = [self.SNdim * self.Wdim, self.Edim]
    theta = reshape_tens3d_into_matrix(self.m, dimT)

    #Set trunc params
    chi, eps = set_trunc_params(prec, trunc_mode, sigma_dim(dimT))

    #Compute SVD
    U, Udag, chi, accuracy_OK = compute_lapack_svd(theta, chi, eps)

    #Copy back svd results
    self.update_site(tens_in = reshape_matrix_into_tens3d(U, [self.SNdim, self.Wdim, chi]))

    #Contract: Udag*theta*(mpsB)     
    dimst=other.m.shape
    tmpMps = np.reshape(np.swapaxes(other.m,0,1),(dimst[1],dimst[0]*dimst[2]))
    tmpMps = np.dot(Udag,np.dot(theta,tmpMps))
    tmpMps = np.reshape(tmpMps,(chi, dimst[0],dimst[2]))
    tmpMps = np.swapaxes(tmpMps,0,1)
    other.update_site(tens_in = tmpMps)

 def zip_mps_mpo_sites(self, other, other_mpo, prec, trunc_mode): #mpsA = self, mpsB = other

    #Set dims of theta & construct theta matrix
    dimT = [self.SNdim * self.Wdim, self.Edim]
    theta = reshape_tens3d_into_matrix(self.m, dimT)

    #Set trunc params
    chi, eps = set_trunc_params(prec, trunc_mode, sigma_dim(dimT))

    #initialize to False, loop until True
    accuracy_OK=False

    while(not accuracy_OK):
      U, Udag, chi, accuracy_OK = compute_lapack_svd(theta, chi, eps)

    #Copy back svd results
    self.update_site(tens_in = reshape_matrix_into_tens3d(U, [self.SNdim, self.Wdim, chi]))

    #Contract: Udag*theta*(mpoB--mpsB)
    tmpMpsMpo = TensMul(other_mpo.m, other.m)
      
    dimst=tmpMpsMpo.shape
    tmpMpsMpo = np.reshape(np.swapaxes(tmpMpsMpo,0,1),(dimst[1],dimst[0]*dimst[2]))
    tmpMpsMpo = np.dot(Udag,np.dot(theta,tmpMpsMpo))
    tmpMpsMpo = np.reshape(tmpMpsMpo,(chi, dimst[0],dimst[2]))
    tmpMpsMpo = np.swapaxes(tmpMpsMpo,0,1)

    other.update_site(tens_in = tmpMpsMpo)
    
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
 
 def append_mposite(self,mposite):

    #Append a new site
    self.data.append(mposite)
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
       
 def insert_site(self, axis, tensor_to_append):

    try:
       if len(tensor_to_append.shape) != 3: raise err.MpsSiteInputError
       #Append a new site
       self.data.insert(axis,mps_site(tens_in = tensor_to_append))
       self.N_sites = self.N_sites + 1 

    except err.MpsSiteInputError as e:
       print("append_site: ", e.msg)
       sys.exit()

    except err.MpsAppendingError as e:
       print("append_site: ", e.msg)
       sys.exit()

 def contract_end(self):
    #contracts one leg of ADT/mps as described in paper
    ns=self.N_sites
    #first contract local leg of last site and store site as matrix, then delete site from MPS
    tens=np.einsum('ijk->j',self.data[ns-1].m)
    del self.data[ns-1]
    self.N_sites=self.N_sites-1
    #multiply in last site with matrix to give new site, stored as tens
    tens=np.einsum('i,jki',tens,self.data[ns-2].m)
    #give tens 1d dummy leg and update last site of MPS
    tens=np.expand_dims(tens,-1)            
    self.data[ns-2].update_site(tens_in=tens)

 def readout(self):
     #contracts all but the 'present time' leg of ADT/mps and returns 1-leg reduced density matrix
    l=len(self.data)
    #for special case of rank-1 ADT just sum over 1d dummy legs and return
    if l==1:
        out=np.sum(np.sum(self.data[0].m,-1),-1)
        return out
    #other wise sum over all but 1-leg of last site, store as out, then successively
    #sum legs of new end sites to make matrices then multiply into vector 'out'
    out=np.sum(np.sum(self.data[l-1].m,0),-1)
    for jj in range(l-2):
        out=np.dot(np.sum(self.data[l-2-jj].m,0),out)
    out=np.dot(np.sum(self.data[0].m,1),out)
    #after the last site, 'out' should now be the reduced density matrix
    return out           

 def canonicalize_mps(self, orth_centre, prec, trunc_mode): 
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
        
 def delete_site(self,site):
     del self.data[site]
     self.N_sites=self.N_sites-1

 ########## Modes: 'accuracy', 'chi', 'fraction' ################################

 #Note the term "+ int(orth_centre != 0)" --> 
 #on [R->L] sweep we reverse Oc to N - Oc + 1, so that all bonds have been through svd 
 #(in particular, the bond to the right of Oc, which is skipped if we reverse Oc to N - Oc instead)
 #If Oc=0 --> reverse to N - Oc instead, cause there's no bond (and no sites) to the right of Oc (so nothing to svd)
 def contract_with_mpo(self, mpo_block, orth_centre=None, prec=0.0001, trunc_mode='accuracy'):          

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
    
 def left_sweep_mps_mpo(self, mpo_block, orth_centre, prec, trunc_mode): 

    self.data[0].update_site(tens_in = TensMul(mpo_block.data[0].m, self.data[0].m))
    self.is_multiplied[0] = True
                  
    for site in range(1,orth_centre):
        if not self.is_multiplied[site]:
            self.data[site-1].zip_mps_mpo_sites(self.data[site], mpo_block.data[site], prec, trunc_mode)
            self.is_multiplied[site] = True
        else:
            self.data[site-1].svd_mps_site(self.data[site], prec, trunc_mode)

 def reverse_mps_mpo_network(self, mpo_block): 
    self.reverse_mps() 
    mpo_block.reverse_mpo() 
    self.is_multiplied.reverse()

 def bonddims(self):
     bond=[]                
     for ss in range(self.N_sites):
          bond.append(self.data[ss].m.shape[2])
     return bond
          
 def totsize(self):
     size=0
     for ss in range(self.N_sites):
         size=self.data[ss].m.shape[0]*self.data[ss].m.shape[1]*self.data[ss].m.shape[2]+size
     return size     