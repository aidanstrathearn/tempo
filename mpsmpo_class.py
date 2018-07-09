from __future__ import print_function
import sys
import ErrorHandling as err
from svd_functions import tensor_to_matrix, matrix_to_tensor, set_trunc_params, compute_lapack_svd
from numpy import dot, swapaxes, transpose, ceil, expand_dims, reshape, eye
from numpy import sum as nsum

#==============================================================================
# Note we refer to tensors here as having North/South/East/West legs -- graphically these labels
# have the usual meaning (North=Up, West=Left etc.) when the tensor network 
# diagrams in the paper are rotated 180 degrees
# 
# That is, our ADT/MPS legs point downwards here and the 'present' timepoint leg is 
# to the left -- python list element 0 in list language
#==============================================================================



######################################################################################################
##########################################  SITE CLASSES  ###########################################
######################################################################################################


##########################################################################
#   Class mpo_site    
# 
#   Attributes: 
#   Sdim, Ndim = dimensions of the South & North legs 
#                      (i.e. 'local dims' of MPO site)
#   Wdim, Edim = dimensions of West & East legs (i.e. 'bond dims' of MPO site)
#   m = the multi-dimensional numpy array representing the tensor
#
#   Synopsis:
#   Defines a single site of MPO - a rank-4 tensor with 4 legs
#
########################################################################### 
class mpo_site(object):

 def __init__(self, tens_in = None):

    try:
       if len(tens_in.shape) != 4: raise err.MpoSiteShapeError

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

 def update_site(self, tens_in = None):

    try:
       if len(tens_in.shape) != 4: raise err.MpoSiteShapeError

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
#   m = the multi-dimensional numpy array representing the tensor
#   Synopsis:
#   Defines a single site of MPS - a rank-3 tensor with 3 legs
#
###########################################################################
class mps_site(object):

 def __init__(self,tens_in = None):

    try:
       if len(tens_in.shape) != 3: raise err.MpsSiteShapeError

       #get dims from tens_in & set mps_site to tens_in 
       self.SNdim = tens_in.shape[0]; self.Wdim = tens_in.shape[1]; self.Edim = tens_in.shape[2]
       self.m = tens_in

    except err.MpsSiteShapeError as e: 
       print("mps_site: ", e.msg)
       sys.exit()

    except err.MpsSiteInputError as e:
       print("mps_site: ", e.msg)
       sys.exit()

 def update_site(self, tens_in = None):

    try:
       if len(tens_in.shape) != 3: raise err.MpsSiteShapeError

       #get dims from tens_in & set mps_site to tens_in 
       self.SNdim = tens_in.shape[0]; self.Wdim = tens_in.shape[1]; self.Edim = tens_in.shape[2]
       self.m = tens_in

    except err.MpsSiteShapeError as e: 
       print("mps: update_site: ", e.msg)
       sys.exit()

    except err.MpsSiteInputError as e:
       print("mps: update_site: ", e.msg)
       sys.exit()

 def contract_with_mpo_site(self,mposite):
     #this contracts mps site with an mpo site to give another mps site with larger bond dims
     #                          
     #     MPS site        W1 --O-- E1
     #                          \                                 
     #                                      ---->     (W1 x W2) --O-- (E1 x E2)   MPS site
     #                          \                                 \
     #     MPO site        W2 --O-- E2
     #                          \
     #
     #first swap axes of MPS/MPO sites to be in the right order to contract using numpy dot
     tensO=dot(swapaxes(mposite.m,1,3),swapaxes(self.m,0,1))
     #get the dimensions of resulting 5-leg tensor
     sh=tensO.shape
     #swap axis to put west legs together and east legs together then reshape into
     #new 3-leg MPS and update
     tensO=reshape(swapaxes(swapaxes(tensO,1,2),2,3),(sh[0],sh[2]*sh[3],sh[1]*sh[4]))
     self.update_site(tens_in=tensO)

######################################################################################################
#########################################  BLOCK CLASSES  #########################################
######################################################################################################


class mpo_block(object):

 def __init__(self):
    #keep track of how long the block is 
    self.N_sites = 0
    self.data = []
         
 def append_mposite(self,mposite):
    #Append a new site
    self.data.append(mposite)
    self.N_sites = self.N_sites + 1

 def reverse_mpo(self):

    self.data.reverse()
    for site in range(self.N_sites):
        MpoSiteT=transpose(self.data[site].m, (0,1,3,2))
        self.data[site].update_site(tens_in = MpoSiteT)

class mps_block():

 def __init__(self,prec,trunc='accuracy'):
    #initialise an mps by stating what precision and what truncation method
    #are going to be used to store it
    
    #set the length of mps_block
    self.N_sites = 0

    #initialize list of mps_sites
    self.data = []
     
    self.precision=prec
    
    self.trunc_mode=trunc
    
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
 
 def truncate_bond(self,k):
    
    #truncates the k'th bond of the MPS using an SVD using method self.trunc_mode and precision self.precision
    #
    #                     
    #          ---O---  Edim  ---O---                   
    #             \              \   
    #                    ^
    #  (k-1)'th site     ^       k'th site
    #                    ^
    #                k'th bond
    #
    #If there are N sites then there are only N-1 bonds so...
    if k<1 or k>self.N_sites-1:
        return 0
    #start by combining south and west legs of (k-1)'th site to give 2-leg tensor which is 
    #the rectangular matrix we will perform the SVD on
    #
    #
    #
    #     Wdim  --O--  Edim  ----->      (Wdim x SNdim) --M-- Edim
    #             \
    #           SNdim                                  'theta'
    #
    dims = [self.data[k-1].SNdim * self.data[k-1].Wdim, self.data[k-1].Edim]
    theta = tensor_to_matrix(self.data[k-1].m, dims)

    #Set trunc params
    chi, eps = set_trunc_params(self.precision, self.trunc_mode, min(dims[0],dims[1]))

    #Now perfom the SVD and truncate - both of these happen inside 'compute_lapack_svd'
    
    #SVD:
    #
    #  dim1 --M-- dim2  ---->  dim1 --U-- min(dim1,dim2) --S-- min(dim1,dim2) --V-- dim2
    #
    #  U, V unitary are matrices (U.Udag=I, V.Vdag=I)
    #  S is diagonal matrix of the min(dim1,dim2) singular values
    
    #The truncation is actually on the unitary U rather than the singular values S
    #We throw away columns of U that get multiplied into entries of the diagonal matrix S which 
    #are smaller than a specified value to leave chi columns. 
    #Then we can approximate the dim1-dimensional identity matrix
    #
    #   dim1 --I-- dim1  ---->   dim1 --U-- chi --Udag-- dim1  
    #
    #so we can express theta as
    #
    #     (Wdim x SNdim) --M-- Edim    ------>   (Wdim x SNdim) --U-- chi --Udag.M-- Edim
    #
    U, Udag, chi = compute_lapack_svd(theta, chi, eps)

    #now retain  (Wdim x SNdim) --U-- chi to become the new (k-1)'th site after reshaping to 
    #separate out west and south legs
    self.data[k-1].update_site(tens_in = matrix_to_tensor(U, [self.data[k-1].SNdim, self.data[k-1].Wdim, chi]))
    
    #multiply chi --Udag.M-- Edim into the k'th site, practically carried out by converting to a matrix and
    #then using numpy dot
    
    tmpMps=tensor_to_matrix(self.data[k].m, (self.data[k].Wdim, self.data[k].SNdim * self.data[k].Edim))
    tmpMps = dot(Udag,dot(theta,tmpMps))
    tmpMps=matrix_to_tensor(tmpMps, (self.data[k].SNdim,chi,self.data[k].Edim))
    self.data[k].update_site(tens_in = tmpMps)
    #Overall then we are left with:
    #                     
    #          ---U---  chi  ---Udag.M---O---    
    #             \                      \   
    #                    ^
    #  (k-1)'th site     ^               k'th site
    #                    ^
    #             truncated k'th bond
    #
    #
    
 def reverse_mps(self):
    #reverse the entire mps bock 
    #first reverse the list of sites
    self.data.reverse()  
    #then site by site reverse the swap the east and west legs of the sites
    for mpssite in self.data:
        mpssite.update_site(tens_in = transpose(mpssite.m, (0,2,1)))
    
 def canonicalize_mps(self, orth_centre): 
    #systematically truncate all bonds of mps - orth_centre is point in mps to stop
    #before truncating from the other end of mps - acheived by reversing mps then using exact same
    #truncation procedure until the orth_centre. then reverses back to original order
    for jj in range(1,orth_centre):
        self.truncate_bond(jj)
    self.reverse_mps()
    for jj in range(1,self.N_sites - orth_centre+1):
        self.truncate_bond(jj)
    self.reverse_mps()
    
 def contract_with_mpo(self, mpo_block, orth_centre=None):          
    #function to contract an mps with an mpo site by site performing truncations at each site
    
    #default val of orth_centre is the actual centre of the mps
    if orth_centre == None: orth_centre=int(ceil(0.5*self.N_sites))
    #contract first sites of MPS/MPO together
    self.data[0].contract_with_mpo_site(mpo_block.data[0])
    #iteratively contract in mpo sites and immediately truncate the 
    #bond connecting to the previous mps site up until the orth_centre
    for site in range(1,orth_centre):
        self.data[site].contract_with_mpo_site(mpo_block.data[site])
        self.truncate_bond(site)
    
    #now reverse the mps and mpo and repeat as above up until all mpo sites have been contracted in
    self.reverse_mps() 
    mpo_block.reverse_mpo()
    #if statement for special case of a 1 site mps
    if self.N_sites>1: self.data[0].contract_with_mpo_site(mpo_block.data[0])   
    for site in range(1,self.N_sites - orth_centre - int(orth_centre == 0)):
        self.data[site].contract_with_mpo_site(mpo_block.data[site])
        self.truncate_bond(site)
    #truncate last bond that links the two halfs of the mps we have seperately swept through above  
    self.truncate_bond(self.N_sites - orth_centre - int(orth_centre == 0))
    #reverse mps and mpo back to original order
    self.reverse_mps() 
    mpo_block.reverse_mpo() 

    #final truncation sweep through mps from one side to the other and back again
    if (orth_centre > 0): self.canonicalize_mps(0)
    if (orth_centre < self.N_sites): self.canonicalize_mps(self.N_sites)
    #if mps has a loose west leg with dim!=1 then grow this leg into new site using a delta function
    if self.data[0].Wdim != 1: self.insert_site(0,expand_dims(eye(self.data[0].Wdim),1))

 def contract_end(self):
    #contracts one leg of ADT/mps as described in paper
    #first sum over south and east legs of end site then dot in to second last site and make
    #3d again by giving 1d dummy index wth expand_dims
    self.data[-2].update_site(tens_in=expand_dims(
            dot(self.data[-2].m,nsum(self.data[-1].m,(0,2)))
            ,-1) )
    #delete the useless end site and change N_sites accordingly
    del self.data[-1]
    self.N_sites=self.N_sites-1

 def readout(self):
     #contracts all but the 'present time' leg of ADT/mps and returns 1-leg reduced density matrix
    l=len(self.data)
    #for special case of rank-1 ADT just sum over 1d dummy legs and return
    if l==1:
        out=nsum(nsum(self.data[0].m,-1),-1)
        return out
    #other wise sum over all but 1-leg of last site, store as out, then successively
    #sum legs of new end sites to make matrices then multiply into vector 'out'
    out=nsum(nsum(self.data[l-1].m,0),-1)
    for jj in range(l-2):
        out=dot(nsum(self.data[l-2-jj].m,0),out)
    out=dot(nsum(self.data[0].m,1),out)
    #after the last site, 'out' should now be the reduced density matrix
    return out    

 def bonddims(self):
     #returns a list of the bond dimensions along the mps
     bond=[]                
     for ss in range(self.N_sites):
          bond.append(self.data[ss].m.shape[2])
     return bond
          
 def totsize(self):
     #returns the total number of elements (i.e. complex numbers) which make up the mps
     size=0
     for ss in range(self.N_sites):
         size=self.data[ss].m.shape[0]*self.data[ss].m.shape[1]*self.data[ss].m.shape[2]+size
     return size
