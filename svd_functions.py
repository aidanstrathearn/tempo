from __future__ import print_function
import sys
import copy as cp
import scipy as sp
import ErrorHandling as err
from numpy import linalg
from numpy import reshape, swapaxes, asarray
import numpy as np

def matrix_to_tensor(matIn, dimOut):
    #reshapes a 2-leg tensor
  if (matIn.shape[1] == dimOut[2]):
     matIn=reshape(matIn,(dimOut[0], dimOut[1], dimOut[2]))

  elif (matIn.shape[0] == dimOut[1]):
     #matIn=swapaxes(reshape(matIn.T,(dimOut[0], dimOut[2], dimOut[1])),1,2)
     matIn=swapaxes(reshape(matIn,(dimOut[1], dimOut[0], dimOut[2])),0,1)
  return matIn

def tensor_to_matrix(tensIn, dimOut):

  #dims of tensIn
  dimIn = asarray(tensIn.shape)

  if (dimIn[2] == dimOut[1]):
     #matOut=reshape(tensIn,(dimIn[0]*dimIn[1],dimIn[2]))
     matOut=reshape(tensIn,(-1,dimIn[2]))
     #matOut=reshape(swapaxes(swapaxes(tensIn,1,2),0,1),(dimIn[2],dimIn[0]*dimIn[1]))
     #matOut=matOut.T
  elif (dimIn[1] == dimOut[0]):
     matOut=reshape(swapaxes(tensIn,0,1),(dimIn[1],dimIn[0]*dimIn[2]))
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

    if (trunc_mode != 'accuracy') and (trunc_mode != 'chi') and (trunc_mode != 'fraction'): raise err.TruncModeError

  except err.TruncModeError as e:
        print("set_trunc_params: ", e.msg)
        sys.exit()

  return chi, eps

def compute_lapack_svd2(theta, chi, eps):
  #Compute SVD using Lapack
  #Create a copy to prevent an accidental modification of theta
  ThetaTmp = cp.deepcopy(theta)
  try:
      U, Sigma, VH = sp.linalg.svd(ThetaTmp, full_matrices=True,lapack_driver='gesvd')
  except(linalg.LinAlgError):
      U, Sigma, VH = sp.linalg.svd(ThetaTmp, full_matrices=True,lapack_driver='gesdd')

  #Truncate SVD matrices  
  #sufficiently small trunc error or end up keeping all sigmas)
  U, chi = truncate_svd_matrices(U, Sigma, chi, eps)

  return U, U.conj().T, chi

#Truncate SVD matrices as given by chi, eps
def truncate_svd_matrices(U, Sigma, chi, eps):

  #Find size of sigma (not equal to sigma_dim in general - e.g. arnoldi returns SdimTmp < sigma_dim)
  sdimTmp=Sigma.shape[0]
  #Only proceed to trunc-err loop if (eps<1.0)
  if (eps < 1.0):
     for i in range(2, sdimTmp + 1):
        if (min(Sigma[0:i])/max(Sigma[0:i])) < eps:
            chi = (i-1)
            break
        elif (i == sdimTmp): 
            chi = sdimTmp

  return U[:, 0:chi], chi

def compute_lapack_svd(theta, chi, eps):
  #Compute SVD using Lapack
  #Create a copy to prevent an accidental modification of theta
  ThetaTmp = cp.deepcopy(theta)
  try:
      U, Sigma, VH = sp.linalg.svd(ThetaTmp, full_matrices=True,lapack_driver='gesvd')
  except(linalg.LinAlgError):
      U, Sigma, VH = sp.linalg.svd(ThetaTmp, full_matrices=True,lapack_driver='gesdd')
  
  try:
      chi=next(i for i in range(len(Sigma)) if Sigma[i]/max(Sigma) < eps)
  except(StopIteration):
      chi=len(Sigma)
      
  U=U[:, 0:chi]
  #Truncate SVD matrices  
  #sufficiently small trunc error or end up keeping all sigmas)
  #U, chi = truncate_svd_matrices(U, Sigma, chi, eps)

  return U, U.conj().T, chi

eps=0
Sigma=[0,1,8,5,4,5,6,3,8,9,-1,3,6,3,6,7]
chi=next(i for i in range(len(Sigma)) if Sigma[i]/max(Sigma) < eps)
print(chi)
print(Sigma[0:2])
chi=0
sdimTmpw=len(Sigma)
for i in range(2, sdimTmpw + 1):
    if (min(Sigma[0:i])/max(Sigma[0:i])) < eps:
        chi = (i-1)
        break
    elif (i == sdimTmpw): 
        chi = sdimTmpw
print(chi)

arr=np.array([[[[1,2,9],[3,4,10]]],[[[1,2,9],[3,4,10]]],[[[1,2,9],[3,4,10]]],[[[1,2,9],[3,4,10]]]])
print(arr.shape)
print(np.transpose(arr).shape)


        