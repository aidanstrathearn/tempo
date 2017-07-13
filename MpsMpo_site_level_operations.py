from __future__ import print_function
import math
import sys
import copy as cp
import numpy as np
import ErrorHandling as err
from tensor_algebra import *



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

 def zip_mpo_mpo_sites(self, other, other_mpo, prec, trunc_mode): #mpsA = self, mpsB = other

    #Set dims of theta & construct theta matrix
    dimT = [self.SNdim * self.Wdim, self.Edim]
    theta = reshape_tens4d_into_matrix(self.m, dimT)

    #Set trunc params
    chi, eps = set_trunc_params(prec, trunc_mode, sigma_dim(dimT))

    #initialize to False, loop until True
    accuracy_OK=False

    while(not accuracy_OK):

       if lapack_preferred(dimT, other.Edim, chi):
          U, Udag, chi, accuracy_OK = compute_lapack_svd(theta, chi, eps)
       else:
          U, Udag, chi, accuracy_OK = compute_arnoldi_svd(theta, chi, eps)

    #Copy back svd results
    self.update_site(tens_in = reshape_matrix_into_tens4d(U, [self.Sdim, self.Ndim, self.Wdim, chi]))

    #Contract: Udag*theta*(mpoB--mpsB)
    tmpMpsMpo = TensMul(other_mpo.m, other.m)
    
    #tmpMpsMpo = np.einsum('km,imn->ikn', theta, tmpMpsMpo)
    tmpMpsMpo = np.swapaxes(np.dot(theta, tmpMpsMpo),0,1)
    
    #tmpMpsMpo = np.einsum('km,imn->ikn', Udag, tmpMpsMpo)
    tmpMpsMpo = np.swapaxes(np.dot(Udag, tmpMpsMpo),0,1)
    
    other.update_site(tens_in = tmpMpsMpo)

 def svd_mpo_site(self, other, prec, trunc_mode): 

    #Set dims of theta & construct theta matrix
    dimT = [self.Sdim*self.Ndim * self.Wdim, self.Edim]
    
    theta = reshape_tens4d_into_matrix(self.m, dimT)

    #Set trunc params
    chi, eps = set_trunc_params(prec, trunc_mode, sigma_dim(dimT))

    #Compute SVD
    U, Udag, chi, accuracy_OK = compute_lapack_svd(theta, chi, eps)

    #Copy back svd results
    self.update_site(tens_in = reshape_matrix_into_tens3d(U, [self.Sdim, self.Ndim, self.Wdim, chi]))

    #Contract: Udag*theta*(mpsB)
    
    #tmpMps = np.einsum('km,imn->ikn', theta, other.m)
    tmpMps = np.swapaxes(np.dot(theta,other.m),0,1)
    #tmpMps = np.einsum('km,imn->ikn', Udag, tmpMps)
    tmpMps = np.swapaxes(np.dot(Udag,tmpMps),0,1)
    other.update_site(tens_in = tmpMps)

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
    
    #tmpMps = np.einsum('km,imn->ikn', theta, other.m)
    tmpMps = np.swapaxes(np.dot(theta,other.m),0,1)
    #tmpMps = np.einsum('km,imn->ikn', Udag, tmpMps)
    tmpMps = np.swapaxes(np.dot(Udag,tmpMps),0,1)
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

       if lapack_preferred(dimT, other.Edim, chi):
          U, Udag, chi, accuracy_OK = compute_lapack_svd(theta, chi, eps)
       else:
          U, Udag, chi, accuracy_OK = compute_arnoldi_svd(theta, chi, eps)

    #Copy back svd results
    self.update_site(tens_in = reshape_matrix_into_tens3d(U, [self.SNdim, self.Wdim, chi]))

    #Contract: Udag*theta*(mpoB--mpsB)
    tmpMpsMpo = TensMul(other_mpo.m, other.m)
    
    #tmpMpsMpo = np.einsum('km,imn->ikn', theta, tmpMpsMpo)
    tmpMpsMpo = np.swapaxes(np.dot(theta, tmpMpsMpo),0,1)
    
    #tmpMpsMpo = np.einsum('km,imn->ikn', Udag, tmpMpsMpo)
    tmpMpsMpo = np.swapaxes(np.dot(Udag, tmpMpsMpo),0,1)
    
    other.update_site(tens_in = tmpMpsMpo)








