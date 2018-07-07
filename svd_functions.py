from __future__ import print_function
import sys
import copy as cp
import numpy as np
import scipy as sp
import ErrorHandling as err
from numpy import linalg

def TensMul(tensA_in, tensB_in):
  #Prepare tensA, tensB --> both should be 4D arrays (should create copies to prevent unwanted modification)
  if (tensA_in.ndim==3): tensA = tensA_in[np.newaxis, ...] 
  else: tensA=tensA_in      
  if (tensB_in.ndim==3): tensB = tensB_in[:, np.newaxis, ...] 
  else: tensB=tensB_in
  
  tensO=np.dot(np.swapaxes(tensA,1,3),np.swapaxes(tensB,0,2))
  rs=tensO.shape
  tensO=np.reshape(np.swapaxes(tensO,1,4),(rs[0],rs[4],rs[2]*rs[3],rs[1]*rs[5]))
  if (tensO.shape[0] == 1) and (tensO.shape[1] == 1): tensO = tensO[0,0,:,:]                                                      
  elif (tensO.shape[0] == 1): tensO = tensO[0,:,:,:] 
  elif (tensO.shape[1] == 1): tensO = tensO[:,0,:,:]
  return tensO

##### reshape matrix into tensor-3d with dims = dimOut ####
def reshape_matrix_into_tens3d(matIn, dimOut):

  if (matIn.shape[1] == dimOut[2]):
     matIn=np.reshape(matIn,(dimOut[0], dimOut[1], dimOut[2]))

  elif (matIn.shape[0] == dimOut[1]):
     matIn=np.swapaxes(np.reshape(matIn.T,(dimOut[0], dimOut[2], dimOut[1])),1,2)
  return matIn

##### reshape tensor-3d into matrix with dims = dimMat ####
def reshape_tens3d_into_matrix(tensIn, dimOut):

  #Initialize matrixOut to zeros
  matOut=np.zeros((dimOut[0], dimOut[1]), dtype=complex)

  #dims of tensIn
  dimIn = np.asarray(tensIn.shape)

  if (dimIn[2] == dimOut[1]):

     for i in range(dimIn[0]):
         matOut[i*dimIn[1] : (i+1)*dimIn[1] , :] = tensIn[i,:,:]

  elif (dimIn[1] == dimOut[0]):

     for i in range(dimIn[0]):
         matOut[: , i*dimIn[2] : (i+1)*dimIn[2]] = tensIn[i,:,:]

  return matOut

#fraction: prec=fraction, set chi=int(prec*sigma_dim), eps=1.0
#chi: prec=chi, set chi=chi, eps=1.0
#accuracy: prec=eps, set eps=eps
def set_trunc_params(prec, trunc_mode, sigma_dim):

  try:

    if (trunc_mode == 'accuracy'):
        ### Fixed accuracy mode ###
        try: 
           if (prec < 0) or (prec > 1) or (prec == None): raise err.EpsModeError

           #Set eps && chi
           eps=prec; chi=2

        except err.EpsModeError as e:
           print("set_trunc_params: ", e.msg)
           sys.exit()
        ##########################


    elif (trunc_mode == 'chi'):
        ### Fixed chi mode ###
        try:
           if not isinstance(prec,int) or not (prec > 0): raise err.ChiModeError
           if prec > sigma_dim: raise err.SigmaDimError

           #Set eps && chi
           chi=prec; eps=1.1
           
        except err.ChiModeError as e:
           print("set_trunc_params: ", e.msg)
           sys.exit()

        except err.SigmaDimError as e:
           print("set_trunc_params: ", e.msg)
           chi=sigma_dim; eps=1.1
        ##########################

    elif (trunc_mode == 'fraction'):
        ### Fixed fraction mode ###
        try:
           if (prec < 0) or (prec > 1) or (prec == None): raise err.FracModeError

           #Set eps && chi
           chi=int(round(prec*sigma_dim)); eps=1.1

           if (chi<1): raise err.ChiError

        except err.FracModeError as e:
           print("set_trunc_params: ", e.msg)
           sys.exit()

        except err.ChiError as e:
           print("set_trunc_params: ", e.msg)
           sys.exit()
        ##########################

    #if user-specified trunc_mode is none of accuracy, chi, fraction:
    if (trunc_mode != 'accuracy') and (trunc_mode != 'chi') and (trunc_mode != 'fraction'): raise err.TruncModeError

  except err.TruncModeError as e:
        print("set_trunc_params: ", e.msg)
        sys.exit()

  return chi, eps

#Find sigma_dim = the total number of sigmas (i.e. untruncated)
def sigma_dim(dimT):
  return min(dimT[0], dimT[1])

#Compute SVD using Lapack
def compute_lapack_svd(theta, chi, eps):

  #Create a copy to prevent an accidental modification of theta
  ThetaTmp = cp.deepcopy(theta)
  #Compute Lapack SVD
  try:
      U, Sigma, VH = sp.linalg.svd(ThetaTmp, full_matrices=True,lapack_driver='gesvd')
  except(linalg.LinAlgError):
      U, Sigma, VH = sp.linalg.svd(ThetaTmp, full_matrices=True,lapack_driver='gesdd')

  #Truncate SVD matrices (accuracy_OK = True cause Lapack returns all sigmas and we'll always be able to reach 
  #sufficiently small trunc error or end up keeping all sigmas)
  U, chi = truncate_svd_matrices(U, Sigma, chi, eps)
  accuracy_OK=True

  return U, U.conj().T, chi, accuracy_OK

#Check if truncErr < eps
def truncErr_below_eps(Sigma, eps):
    #Check if truncErr < eps
    truncErr_vs_eps = (np.min(Sigma)/np.max(Sigma)) < eps
    return truncErr_vs_eps

#Truncate SVD matrices as given by chi, eps
def truncate_svd_matrices(U, Sigma, chi, eps):

  #Find size of sigma (not equal to sigma_dim in general - e.g. arnoldi returns SdimTmp < sigma_dim)
  sdimTmp=Sigma.shape[0]

  #Only proceed to trunc-err loop if (eps<1.0)
  #i.e. only if we've chosen 'accuracy mode' but not 'chi' or 'frac'
  if (eps < 1.0):
     for i in range(2, sdimTmp + 1):
        if truncErr_below_eps(Sigma[0:i], eps):
            chi = (i-1)
            break
        elif (i == sdimTmp): 
            chi = sdimTmp

  return U[:, 0:chi], chi