import math
import numpy as np
import copy as cp
import sys
from scipy.sparse.linalg import svds, LinearOperator
from definitions import mps_block, mpo_block, mps_site, mpo_site

#Global MPS/MPO variables needed in Arnoldi SVD (called iteratively inside arnoldi svd routine):
intermediate_mps = None
mpoX = None; mpsX = None
#Module level definitions of matrices/vecs needed in Arnoldi SVD (called iteratively inside arnoldi svd routine):
Block_Ltemp = None; Block_Rtemp = None
Block_Lvec = None; Block_Rvec = None


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
def multiply_block(mps_block1, mpo_block1, which_mode='accuracy', precision=3.0e-01, delta_chi=1, eval_loop_start=None, eval_loop_end=None):

   global intermediate_mps
   
   #The code currently works in cases where MPS_length <= MPO_length
   if (mps_block1.N_sites > mpo_block1.N_sites):
        sys.exit("ERROR in multiply_block: mpo_block should be equal to or longer than mps_block. Exiting...")

   #Sanity check of the input
   if (which_mode != 'accuracy'):
     if (which_mode != 'chi'):
        if (which_mode != 'fraction'):
           sys.exit("ERROR in multiply_block: which_mode must be 'accuracy', 'chi', or 'fraction'. Exiting...")

   #Sanity check of the input
   if (which_mode == "accuracy") or (which_mode == "fraction"):  
      #Note that both [fraction of evals] and [accuracy = ratio of smallest & largest sigma] must be between 0 and 1
      if (precision < 0) or (precision > 1) or (precision == None):
          sys.exit("ERROR in multiply_block: precision must be between 0 and 1. Exiting...")
   elif (which_mode == "chi"):
      if not (precision == None):
         if not isinstance(precision,int) or not (precision > 0): 
            sys.exit("ERROR in multiply_block: precision must be a positive integer or equal to None. Exiting...")
                
   #Sanity check of the input
   if which_mode == "accuracy":

      if not (eval_loop_start == None):
         if not isinstance(eval_loop_start,int) or not (eval_loop_start > 0):
            sys.exit("ERROR in multiply_block: eval_loop_start must be a positive integer or equal to None. Exiting...")

      if not (eval_loop_end == None): 
         if not isinstance(eval_loop_end,int) or not (eval_loop_end > 0):
            sys.exit("ERROR in multiply_block: eval_loop_end must be a positive integer or equal to None. Exiting...")


   #Check that ends of mps/mpo have bond_dim = 1
   if (mps_block1[0].Wdim != 1):
       sys.exit("ERROR in multiply_block: first site of MPS must have West dim = 1. Exiting...")
   if (mpo_block1[0].Wdim != 1):
       sys.exit("ERROR in multiply_block: first site of MPO must have West dim = 1. Exiting...")
   if (mps_block1[mps_block1.N_sites - 1].Edim != 1):
       sys.exit("ERROR in multiply_block: last site of MPS must have East dim = 1. Exiting...")
   if (mpo_block1[mpo_block1.N_sites - 1].Edim != 1):
       sys.exit("ERROR in multiply_block: last site of MPO must have East dim = 1. Exiting...")

       
   #Note that Python numbers its lists from 0 to N-1!!!
   for site in np.arange(mpo_block1.N_sites):

       #Verify that MPS/MPO have correct South & North dims 
       if (site < mps_block1.N_sites):
          if (mpo_block1[site].Ndim == 1):
              print('Error at site = ', site)
              sys.exit("ERROR in multiply_block: mpo_block has been set up incorrectly - should have North dim > 1. Exiting...")
          if (mpo_block1[site].Sdim == 1):
              print('Error at site = ', site)
              sys.exit("ERROR in multiply_block: mpo_block has been set up incorrectly - should have South dim > 1. Exiting...")
          if (mps_block1[site].SNdim == 1):
              print('Error at site = ', site)
              sys.exit("ERROR in multiply_block: mpo_block has been set up incorrectly - should have South-North dim > 1. Exiting...")

       ############## PERFORM BLOCK MULTIPLICATION AND SVD #############################################################################################
    
       if (site == 0):
           #Simple mult at site=0, no SVD
           lapack_multiply_each_site(mps_block1, mpo_block1, site, which_mode, precision, eval_loop_start, eval_loop_end)
       else:
           if (site == 1) or (site == mps_block1.N_sites - 1):
                #use lapack at site=1,N_sites-1
                lapack_multiply_each_site(mps_block1, mpo_block1, site, which_mode, precision, eval_loop_start, eval_loop_end)
           elif (site > mps_block1.N_sites - 1):
                #add the free mpo sites to mps
                sweep_over_free_mpo_sites(mps_block1, mpo_block1, site, which_mode, precision, eval_loop_start, eval_loop_end)
           else:
                #on other sites, use arnoldi by default, but switch to lapack if eval_frac is too large (see arnoldi_multiply_each_site function)
                arnoldi_multiply_each_site(mps_block1, mpo_block1, site, which_mode, precision, delta_chi, eval_loop_start, eval_loop_end)

      ##################################################################################################################################################


   #Check that ends of output MPS have bond_dim = 1
   if (mps_block1[0].Wdim != 1):
       sys.exit("OUTPUT ERROR in multiply_block: first site of OUTPUT MPS must have West dim = 1. Exiting...")
   if (mps_block1[mps_block1.N_sites - 1].Edim != 1):
       sys.exit("OUTPUT ERROR in multiply_block: last site of OUTPUT MPS must have East dim = 1. Exiting...")

   return mps_block1
   



###########################################################################
#
#  lapack_multiply_each site
#
#  Variables:
#  mps_block, mpo_block = input generic MPS/MPO blocks
#  site = the site at which we multiply MPS & MPO 
#  N_sites = length of MPS/MPO blocks
#
#  Synopsis:
#  This function multiplies a single site of MPS by a single site of MPO.
#  The multiplication produces an intermediate_mps 3-leg tensor at site=site
#  (i.e. intermed_mps = MPS site 
#  with untruncated east bond - combo dim of the state & operator bond dims)
#
#  The steps of multiplication:
#  0)Contract MPS(site) with MPO(site): calc MPS*MPO product at site=site
#  1)Take the intermediate_mps from site=site-1
#  2)Combine it with MPS*MPO product at site=site into a Theta matrix
#  3)Subsequently perform Lapack SVD of Theta matrix
#  4)Write U*sqrt(S) to form the MPS(site-1)
#  5)Write sqrt(S)*VH to form the new intermediate_mps at site=site
#   
############################################################################
def lapack_multiply_each_site(mps_block, mpo_block, site, which_mode, precision, eval_loop_start, eval_loop_end):

   global intermediate_mps
   global mpoX, mpsX

   #initialize intermediate_mps
   if site==0:
      intermediate_mps = mps_site(mps_block[site].SNdim, mps_block[site].Wdim, mps_block[site].Edim*mpo_block[site].Edim)

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mps_block[site], mpo_block[site])
   mpoX=mpo_block[site]; mpsX=mps_block[site]

   #Multiply mps & mpo at site=site
   mps_mpo_product_at_site = np.zeros((SNdim_B, sdim_A*opdim_A, sdim_B*opdim_B), dtype=complex)
   for oB in np.arange(opdim_B):
     for sB in np.arange(sdim_B):
       for oA in np.arange(opdim_A):
         for sA in np.arange(sdim_A):
            mps_mpo_product_at_site[:, sA + oA*sdim_A, sB + oB*sdim_B] = np.dot(mpo_block[site].m[:,:,oA,oB], mps_block[site].m[:,sA,sB])
   
   if site==0:
     #at site=0 initialize intermediate_mps & move to the next step
     intermediate_mps.m = cp.deepcopy(mps_mpo_product_at_site)
   else:
     #construct theta matrix for svd
     theta=np.zeros((Wdim_A*SNdim_A, sdim_B*opdim_B*SNdim_B), dtype=complex)
     temp_prod=np.zeros((Wdim_A, sdim_B*opdim_B), dtype=complex)

     for iA in np.arange(SNdim_A):
       for iB in np.arange(SNdim_B):        
         temp_prod = np.dot(intermediate_mps.m[iA,:,:], mps_mpo_product_at_site[iB,:,:])
         theta[iA*Wdim_A : (iA+1)*Wdim_A , iB*sdim_B*opdim_B : (iB+1)*sdim_B*opdim_B] = temp_prod[0:Wdim_A,0:sdim_B*opdim_B]

     #MAX number of singular vals
     nev_max = min(SNdim_A*Wdim_A, SNdim_B*sdim_B*opdim_B)
     #current bond dim of mps
     current_chi = sdim_A

     ##### PERFORM LAPACK SVD ##########
     print('starting LAPACK SVD at site = ', site)
     U, S, VH, sigma_dim = perform_lapack_svd(theta, which_mode, precision, eval_loop_start, eval_loop_end, current_chi, nev_max)
     
     #Copy back SVD results to mps_block
     copy_back_svd_results(mps_block, mpo_block, site, U, S, VH, sigma_dim)      

     #At the last site, copy intermediate_mps to mps_block[N_sites-1]
     if site == (mps_block.N_sites-1):
         mps_block[mps_block.N_sites-1] = mps_site(SNdim_B, sigma_dim, sdim_B*opdim_B)
         mps_block[mps_block.N_sites-1].m = cp.deepcopy(intermediate_mps.m)



############################################################################
#
# perform_lapack_svd
#
# Variables: 
#       
#
############################################################################
def perform_lapack_svd(theta, which_mode, precision, eval_loop_start, eval_loop_end, current_chi, nev_max):

     #Specify start & end pts of lapack eval loop
     if which_mode == "accuracy":
          if eval_loop_start == None:
               eval_loop_start = current_chi
          if eval_loop_end == None:
               eval_loop_end = nev_max
     elif which_mode == "chi":
          if precision == None:
               sigma_dim = current_chi
          else:
               sigma_dim = precision
     elif which_mode == "fraction":
               sigma_dim = int(round(precision*nev_max))         

     ##### PERFORM LAPACK SVD USING INPUT THETA ##########
     U, S, VH = np.linalg.svd(theta, full_matrices=True)

     if (which_mode == 'fraction') or (which_mode == 'chi'):

          #If requested sigma_dim > nev_max ---> set sigma_dim = nev_max 
          if (sigma_dim > nev_max):
               sigma_dim = nev_max
               print('LAPACK: the requested number of singular values is larger than nev_max. Setting sigma_dim = nev_max')

          print('LAPACK completed, number of singular vals: ', sigma_dim, 'eval frac = ', sigma_dim/float(nev_max))
          print(' ')
          #return svd matrices & exit the loop
          return U, S, VH, sigma_dim
     
     elif (which_mode == 'accuracy'):
          #If eval_loop_start > eval_loop_end ---> set eval_loop_start = eval_loop_end
          if (eval_loop_start > eval_loop_end):
             eval_loop_start = eval_loop_end
             print('LAPACK: eval_loop_start is larger than eval_loop_end. Setting eval_loop_start = eval_loop_end')
 
          #lapack SVD finds all singular vals - just loop over S, and decide how many singular vals we should keep 
          #i.e. no need to repeat SVD multiple times (in contrast with Arnoldi)
          for sigma_dim in np.arange(eval_loop_start, eval_loop_end + 1):

             #calc the ratio b/n smallest & largest singular vals
             sigma_max = np.amax(S[0:sigma_dim]); sigma_min = np.amin(S[0:sigma_dim]); sigma_ratio = sigma_min/sigma_max
             #eval frac
             eval_frac = sigma_dim/float(nev_max)
             print('LAPACK frac of singular vals to calc', eval_frac)

             #If we haven't achieved the required accuracy yet - increment the num of singular vals we keep
             if (sigma_ratio > precision) and (eval_frac < 1.0):
                 print('LAPACK increasing sigma_dim to: ', sigma_dim + 1)
                 print(' ')
             else:
                 print('LAPACK required accuracy achieved or eval_frac = 1.0 with sigma_dim = : ', sigma_dim)
                 print(' ')
                 #return svd matrices & exit the loop
                 return U, S, VH, sigma_dim
                 break



############################################################################
#
# copy_back_svd_results
#
# Variables: 
#       
#
############################################################################
def copy_back_svd_results(mps_block, mpo_block, site, U, S, VH, sigma_dim):

   global intermediate_mps

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mps_block[site], mpo_block[site])

   #copy U to mps_block[site-1]
   mps_block[site-1] = mps_site(SNdim_A, Wdim_A, sigma_dim)
   #print('sigma_dim is: ', sigma_dim)
   #print('shape of mps_block(site-1): ', np.shape(mps_block[site-1].m))
   #print('shape of U is: ', np.shape(U))
   for i in np.arange(SNdim_A):
       mps_block[site-1].m[i,:,0:sigma_dim] = U[i*Wdim_A : (i+1)*Wdim_A , 0:sigma_dim] 

   #Construct new intermediate_mps at site=site & copy VH to it
   #(i.e. we shift by site ---> site+1)
   intermediate_mps = mps_site(SNdim_B, sigma_dim, sdim_B*opdim_B)
   for i in np.arange(SNdim_B):
       intermediate_mps.m[i, 0:sigma_dim, :] = VH[0:sigma_dim , i*sdim_B*opdim_B : (i+1)*sdim_B*opdim_B]

   #absorb S to mps_block[site-1] and the new intermediate_mps at site=site
   for i in np.arange(sigma_dim):
       fac=np.sqrt(S[i]) 
       mps_block[site-1].m[:,:,i] = mps_block[site-1].m[:,:,i]*fac
       intermediate_mps.m[:,i,:] = fac*intermediate_mps.m[:,i,:]



###########################################################################
#
#  arnoldi_multiply_each site
#
#  Variables:
#  mps_block, mpo_block = input generic MPS/MPO blocks
#  site = the site at which we multiply MPS & MPO 
#
#  Synopsis:
#  This function performs an implicit MPS*MPO multiplication at site=site: 
#  we combine intermediate_mps at site=site-1 and MPS*MPO product at site=site
#  within the LinearOperator function called in Arnoldi method which performs 
#  SVD. Copying back the SVD results:
#  (a) Write U*sqrt(S) to form the MPS(site-1)
#  (b) Write sqrt(S)*VH to form the new intermediate_mps at site=site
#  
#   
############################################################################
def arnoldi_multiply_each_site(mps_block, mpo_block, site, which_mode, precision, delta_chi, eval_loop_start, eval_loop_end):

   global intermediate_mps
   global mpoX, mpsX

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mps_block[site], mpo_block[site])
   mpoX=mpo_block[site]; mpsX=mps_block[site]

   #Def theta = linear operator that acts on an input vector
   theta = LinearOperator(matvec=arnoldi_app_op_to_vec, rmatvec=arnoldi_app_HC_op_to_vec, matmat=arnoldi_mult_mat_by_mat, shape=(SNdim_A*Wdim_A, SNdim_B*sdim_B*opdim_B))

   #max number of singular vals
   nev_max = min(SNdim_A*Wdim_A, SNdim_B*sdim_B*opdim_B)
   #Right dim of theta & VH matrix - required in Arnoldi
   VH_rdim = opdim_B*sdim_B*SNdim_B
   #Current bond dim of mps:
   current_chi = sdim_A

   ##### PERFORM ARNOLDI SVD ##########
   print('starting ARNOLDI SVD at site = ', site)
   U, S, VH, sigma_dim = perform_arnoldi_svd(theta, which_mode, precision, delta_chi, eval_loop_start, current_chi, VH_rdim, nev_max)

   if (S[0] == 'switch-to-lapack') and (len(S) == 1):
     ##### SWITCH TO LAPACK SVD
     #Continue starting with the same [eval_loop_start = sigma_dim] that Arnoldi has got to ##########
     eval_loop_start = sigma_dim
     lapack_multiply_each_site(mps_block, mpo_block, site, which_mode, precision, eval_loop_start, eval_loop_end)
   else:
     #Copy back SVD results to mps_block
     copy_back_svd_results(mps_block, mpo_block, site, U, S, VH, sigma_dim)



############################################################################
#
# perform_arnoldi_svd
#
# Variables: 
#       
#
############################################################################
def perform_arnoldi_svd(theta, which_mode, precision, delta_chi, sigma_dim, current_chi, VH_rdim, nev_max):

   #if the fraction of singular vals > critical fraction -- use lapack, else -- use arnoldi
   threshold_frac = 0.5
   #count the number of iterations in the while loop when sigma_dim was increased 
   #(so that we know if we should try decreasing sigma_dim or not)
   num_iters = 1

   #Specify starting pt of arnoldi eval loop (must be outside while loop)
   if which_mode == "accuracy":
      if sigma_dim == None:
         sigma_dim = current_chi
   elif which_mode == "chi":
      if precision == None:
         sigma_dim = current_chi
      else:
         sigma_dim = precision 
   elif which_mode == "fraction":
      sigma_dim = int(round(precision*nev_max))

   #If which_mode == accuracy ---> keep incrementing sigma_dim until (sigma_ratio < precision)
   #Else ---> while loop stops after one iteration
   while(True):

      #eval frac to decide between using Arnoldi & switching to Lapack (must be inside while loop)
      eval_frac = sigma_dim/float(nev_max)
      print('ARNOLDI - frac of singular vals to calc', eval_frac)

      if (eval_frac > threshold_frac):

         #If we exceed the threshold fraction of singular vals - switch to lapack instead 
         print('ARNOLDI - eval_frac exceeded the threshold at sigma_dim', sigma_dim ,'switching to LAPACK: ')
         print(' ')
         #Return U,S,VH = 'switch-to-lapack' as a signal that we should switch to lapack; 
         #IMPORTANT: return sigma_dim as is, since it will be needed in lapack svd;
         U = []; U.append('switch-to-lapack'); S = []; S.append('switch-to-lapack'); VH = []; VH.append('switch-to-lapack')             
         return U, S, VH, sigma_dim
         break

      else:

         ##### PERFORM ARNOLDI SVD USING INPUT THETA ##########
         U, S, VH = svds(theta, k=sigma_dim, ncv=np.minimum(3*sigma_dim+1,VH_rdim), tol=10**(-5), which='LM', v0=None, maxiter=10*sigma_dim, return_singular_vectors=True)

         if (which_mode == 'fraction') or (which_mode == 'chi'):

             print('ARNOLDI completed, number of singular vals: ', sigma_dim)
             print(' ')
             #exit the loop & return svd matrices
             return U, S, VH, sigma_dim
             break
                     
         elif (which_mode == 'accuracy'):

              #calc the ratio b/n smallest & largest singular vals
              sigma_max = np.amax(S[0:sigma_dim]); sigma_min = np.amin(S[0:sigma_dim]); sigma_ratio = sigma_min/sigma_max

              if (sigma_ratio > precision):
                  #If we haven't achieved the required accuracy yet - increment the num of singular vals
                  sigma_dim = sigma_dim + delta_chi
                  print('ARNOLDI - increasing sigma_dim to: ', sigma_dim)
                  print(' ')
                  #count the number of iters when sigma_dim was increased
                  num_iters = num_iters + 1
              elif (sigma_ratio < precision) and (num_iters == 1) and (sigma_dim > 2):
                  #If the 1st iteration of while loop gives sigma_dim = already sufficient to achieve required accuracy
                  #Then try reducing sigma_dim (to prevent mps_block from getting stuck in an intermediate high-entanglement state)
                  #During transient t-evol, intermediate states often have higher entanglement than the final SS & initial state
                  #sigma_dim = sigma_dim - 1
                  sigma_dim = sigma_dim - 1
                  print('ARNOLDI - required accuracy already achieved by 1st iter of while loop.')
                  print('Try reducing sigma_dim to', sigma_dim)
                  print(' ')
                  #Note that if we decrease sigma_dim -> sigma_dim - 1 and find that sigma_dim - 1 is not sufficient anymore
                  #We'll get (sigma_ratio > precision) and we'll have to increase sigma_dim -> sigma_dim + delta_chi
                  #After this, we'll have num_iters = 1 --> num_iters = 2
                  #This will prevent us from coming back to the (sigma_ratio < precision) and (num_iters == 1) case  
                  #Thus the code won't try to decrease sigma_dim anymore, 
                  #And won't remain stuck in a perpetual back-n-forth cycle of increasing-decreasing sigma_dim
              else:
                  print('ARNOLDI - required accuracy achieved with: ', sigma_dim)
                  print(' ')
                  #exit the loop & return svd matrices
                  #Note: we only return U, S, VH, sigma_dim if the required accuracy has already been achieved
                  return U, S, VH, sigma_dim
                  break




###########################################################################
#
#  arnoldi_app_op_to_vec
#
#  Variables:
#  Vi = general input vector on which the LinearOperator function acts
#  Vo = the output vector produced by the LinearOperator function acting on Vi
#
#  Synopsis:
#  This function constructs a LinearOperator that represents the action of
#  Theta matrix on an input vector: Vo = Theta * Vi.
# 
#  For each pair of indices {sA,oA} of the bond connecting site-1 and site:
#  1) We take intermediate_mps at site=site-1 and MPS*MPO product at site=site
#  and produce a vector pair |Block_Lvec> <Block_Rvec|. 
#  2) We apply the entity |Block_Lvec> <Block_Rvec| to Vi. 
#  Then, we sum the outcomes of all operations |Block_Lvec> <Block_Rvec| |Vi> 
#  at different pairs of indices {sA,oA} to produce the output vector:
#  Vo = SUM_{sA,oA} (|Block_Lvec> <Block_Rvec|)_{sA,oA} |Vi>.
#
#  This way, we perform the potentially expensive Theta*Vi operation 
#  in small chunks & sum the outcomes, instead of doing it all at once.
############################################################################
def arnoldi_app_op_to_vec(Vi):

   global Block_Ltemp, Block_Rtemp 
   global Block_Lvec, Block_Rvec

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mpsX, mpoX)

   Block_Ltemp = np.zeros((SNdim_A,Wdim_A), dtype=complex)
   Block_Rtemp = np.zeros((SNdim_B,sdim_B*opdim_B), dtype=complex)
   Block_Lvec = np.zeros((SNdim_A*Wdim_A), dtype=complex) 
   Block_Rvec = np.zeros((SNdim_B*sdim_B*opdim_B), dtype=complex)

   #initialize output vec
   Vo=0.0
   for oA in np.arange(opdim_A):
     for sA in np.arange(sdim_A):
       Block_Ltemp = intermediate_mps.m[:,:,sA+oA*sdim_A]
       for oB in np.arange(opdim_B):
         for sB in np.arange(sdim_B):      
            Block_Rtemp[:, sB+oB*sdim_B] = np.dot(mpoX.m[:,:,oA,oB], mpsX.m[:,sA,sB])

       Block_Lvec = np.reshape(Block_Ltemp, (SNdim_A*Wdim_A))
       Block_Rvec = np.reshape(Block_Rtemp, (SNdim_B*sdim_B*opdim_B))
      
       Vo = Vo + Block_Lvec*np.dot(Vi, Block_Rvec)

   return Vo


###########################################################################
#
#  arnoldi_app_HC_op_to_vec
#
#  Variables:
#  Vi = general input vector on which the LinearOperator function acts
#  Vo = the output vector produced by the LinearOperator function acting on Vi
#
#  Synopsis:
#
#  Construct a LinearOperator analogous to arnoldi_app_op_to_vec, but this
#  time it represents the action of Theta^HC matrix on an input vector: 
#  Vo = Theta^HC * Vi. 
# 
#  We follow these additional steps: 
#  a) Compute Vi^HC
#  b) Peform Vo = Vi^HC * Theta
#  c) HC back: Vo = Vo^HC = (Vi^HC * Theta)^HC = Theta^HC * Vi
#
#  All the other steps are the same as in arnoldi_app_op_to_vec.
#
############################################################################
def arnoldi_app_HC_op_to_vec(Vi):

   global Block_Ltemp, Block_Rtemp 
   global Block_Lvec, Block_Rvec, Block_vc

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mpsX, mpoX)

   Block_Ltemp = np.zeros((SNdim_A,Wdim_A), dtype=complex)
   Block_Rtemp = np.zeros((SNdim_B,sdim_B*opdim_B), dtype=complex)
   Block_Lvec = np.zeros((SNdim_A*Wdim_A), dtype=complex) 
   Block_Rvec = np.zeros((SNdim_B*sdim_B*opdim_B), dtype=complex)

   Vc = np.reshape(np.conj(Vi), (-1))

   #initialize output vec
   Vo=0.0

   for oA in np.arange(opdim_A):
     for sA in np.arange(sdim_A):
       Block_Ltemp = intermediate_mps.m[:,:,sA+oA*sdim_A]
       for oB in np.arange(opdim_B):
         for sB in np.arange(sdim_B):      
            Block_Rtemp[:, sB+oB*sdim_B] = np.dot(mpoX.m[:,:,oA,oB], mpsX.m[:,sA,sB])

       Block_Lvec = np.reshape(Block_Ltemp, (SNdim_A*Wdim_A))
       Block_Rvec = np.reshape(Block_Rtemp, (SNdim_B*sdim_B*opdim_B)) 
   
       Vo = Vo + Block_Rvec*np.dot(Vc, Block_Lvec)

   Vo=np.conj(Vo)

   return Vo


###########################################################################
#
#  arnoldi_mult_mat_by_mat
#
#  Variables:
#  Vi = general input matrix on which the LinearOperator function acts
#  Vo = the output matrix produced by the LinearOperator function acting on Vi
#
#  Synopsis:
#
#  All the steps are the same as in arnoldi_app_op_to_vec, but this
#  time it represents the multiplication of Theta matrix with an input matrix: 
#  Vo = Theta*Vi, where Vo, Theta, Vi are all matrices. 
#
#
############################################################################
def arnoldi_mult_mat_by_mat(Vi):

   global Block_Ltemp, Block_Rtemp 
   global Block_Lvec, Block_Rvec

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mpsX, mpoX)

   Block_Ltemp = np.zeros((SNdim_A,Wdim_A), dtype=complex)
   Block_Rtemp = np.zeros((SNdim_B,sdim_B*opdim_B), dtype=complex)
   Block_Lvec = np.zeros((SNdim_A*Wdim_A), dtype=complex) 
   Block_Rvec = np.zeros((SNdim_B*sdim_B*opdim_B), dtype=complex)

   #initialize the output matrix
   Vo=0.0
   for oA in np.arange(opdim_A):
     for sA in np.arange(sdim_A):
       Block_Ltemp = intermediate_mps.m[:,:,sA+oA*sdim_A]
       for oB in np.arange(opdim_B):
         for sB in np.arange(sdim_B):      
            Block_Rtemp[:, sB+oB*sdim_B] = np.dot(mpoX.m[:,:,oA,oB], mpsX.m[:,sA,sB])

       Block_Lvec = np.reshape(Block_Ltemp, (SNdim_A*Wdim_A))
       Block_Rvec = np.reshape(Block_Rtemp, (SNdim_B*sdim_B*opdim_B))

       Vo = Vo + np.outer(Block_Lvec, np.dot(Block_Rvec, Vi))

   return Vo





############################################################################
#
# sweep_over_free_mpo_sites
#
# Variables: 
#       
#
############################################################################
def sweep_over_free_mpo_sites(mps_block, mpo_block, site, which_mode, precision, eval_loop_start, eval_loop_end):

     #Verify that mpo_block input has an mps-like tail 
     if (mpo_block[site].Ndim != 1):
        print('Error at site = ', site)
        sys.exit("ERROR in sweep_over_free_mpo_sites: mpo_block has been set up incorrectly - should have North dim = 1. Exiting...")

     #find dims of the tensor objects we're dealing with
     SNdim_A = mpo_block[site-1].Sdim; SNdim_B = mpo_block[site].Sdim
     opdim_A = mpo_block[site].Wdim; opdim_B = mpo_block[site].Edim
     Wdim_A = mps_block[site-1].Wdim

     #construct theta matrix for svd
     theta=np.zeros((Wdim_A*SNdim_A, opdim_B*SNdim_B), dtype=complex)
     temp_prod=np.zeros((Wdim_A, opdim_B), dtype=complex)

     for iA in np.arange(SNdim_A):
       for iB in np.arange(SNdim_B): 
         temp_prod = np.dot(mps_block[site-1].m[iA,:,:], mpo_block[site].m[iB,0,:,:])
         theta[iA*Wdim_A : (iA+1)*Wdim_A , iB*opdim_B : (iB+1)*opdim_B] = temp_prod[0:Wdim_A , 0:opdim_B]

     #tot number of singular vals
     nev_max = min(SNdim_A*Wdim_A, SNdim_B*opdim_B)
     #current bond dim of mps
     current_chi = opdim_A

     ##### PERFORM LAPACK SVD ##########
     print('starting LAPACK SVD at site = ', site)
     U, S, VH, sigma_dim = perform_lapack_svd(theta, which_mode, precision, eval_loop_start, eval_loop_end, current_chi, nev_max)

     #copy U to mps_block[site-1]
     mps_block[site-1] = mps_site(SNdim_A, Wdim_A, sigma_dim)
     for i in np.arange(SNdim_A):
         mps_block[site-1].m[i,:,0:sigma_dim] = U[i*Wdim_A : (i+1)*Wdim_A , 0:sigma_dim] 

     #Append extra site to mps_block
     mps_block.append(mps_site(SNdim_B, sigma_dim, opdim_B))
     mps_block.N_sites = mps_block.N_sites + 1
     for i in np.arange(SNdim_B):
         mps_block[site].m[i, 0:sigma_dim, :] = VH[0:sigma_dim , i*opdim_B : (i+1)*opdim_B]

     
     #absorb S to mps_block[site-1] and the new intermediate_mps at site=site
     for i in np.arange(sigma_dim):
         fac=np.sqrt(S[i]) 
         mps_block[site-1].m[:,:,i] = mps_block[site-1].m[:,:,i]*fac
         mps_block[site].m[:,i,:] = fac*mps_block[site].m[:,i,:]








###########################################################################
#
#  contract_two_mps
#
#  Variables:
#  mps_block1, mps_block2 = input generic MPS blocks
#  N_sites = length of the MPS blocks
#
#  Synopsis:
#  This function contracts two MPS blocks to produce a scalar expec value.
#  a) At site=1, we initialize the overlap variable between the two MPS:
#     MPS1(site)*MPS2(site) = mps_OVERLAP    
#  b) We then sweep over all sites as follows:
#  c) At each site, multiply MPS1(site)*MPS2(site) = Single_Site_Overlap 
#    (zip the chains together at site=site)
#  d) Then absorb Single_Site_Overlap (the zipped element) into mps_OVERLAP
#  e) Continue until we reach the end of MPS(1,2) chains - 
#     the last absorption gives a scalar value (0-leg tensor)
#
############################################################################
def contract_two_mps(mps_block1, mps_block2):

   #Note that Python numbers its lists from 0 to N-1!!!
   #Sweep thru all sites, absorbing each site into mps_overlap
   # via 3-step process ZIP-ABSORB-UPDATE
   for site in np.arange(mps_block1.N_sites):
        #Get east & west dims of each site
        Wdim1 = cp.copy(mps_block1[site].Wdim); Wdim2 = cp.copy(mps_block2[site].Wdim)
        Edim1 = cp.copy(mps_block1[site].Edim); Edim2 = cp.copy(mps_block2[site].Edim)
        if site==0:
           mps_overlap = np.zeros((Edim1*Edim2), dtype=complex)
           
           #At site=0, initialize mps_overlap by multiplying the first pair of sites (here Wdim=1)
           for e1 in np.arange(Edim1):
             for e2 in np.arange(Edim2):
                mps_overlap[e2 + e1*Edim2] = np.dot(mps_block1[site].m[:,0,e1], mps_block2[site].m[:,0,e2])
        else:
           single_site_overlap = np.zeros((Wdim1*Wdim2, Edim1*Edim2), dtype=complex)
           #Zip the chain at site=site to form a zipped-up segment = single_site overlap
           for e1 in np.arange(Edim1):
             for e2 in np.arange(Edim2):
               for w1 in np.arange(Wdim1):
                 for w2 in np.arange(Wdim2):
                     single_site_overlap[w2 + w1*Wdim2, e2 + e1*Edim2] = np.dot(mps_block1[site].m[:,w1,e1], mps_block2[site].m[:,w2,e2])
            #At site=N_sites-1 the chain terminates and 
            #single_site_overlap has only one leg (cause Edim=1)
            #Thus reshape single_site_overlap as a vector

           #Absorb the zipped up segment at site=site into mps_overlap
           if (site == mps_block1.N_sites-1):
                mps_overlap = np.dot(mps_overlap, single_site_overlap[:,0])
           else:
                mps_overlap = np.dot(mps_overlap, single_site_overlap)

   #Once all sites of the two mps chains have been absorbed into mps_overlap
   #mps_overlap is just a scalar value - write it to expec_value
   expec_value = cp.deepcopy(mps_overlap)

   return expec_value



##################################################################################################
#
#  get_dims_of_mpo_mps
#
#  Variables:
#  mps_in_B, mpo_in_B - site-B of MPS & MPO blocks --> mps_in_B(site) = MPS(site), mpo_in_B = MPO(site) 
#  prior to MPS*MPO multiplication at site=B.
#
#  mps_in_A - site-A of MPS block --> can be either intermediate_mps tensor or mps_block(site-1)
#  whichever is more convenient to determine Wdim_A, SNdim_A.
#
#  Synopsis:
#  This function returns the dimensions of MPS, MPO sites = (site-1 and site)
#  involved in multiplication & SVD. 
#  The dims to calculate: 
#  Wdim, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B
###################################################################################################
def get_dims_of_mpo_mps(mps_in_A, mps_in_B, mpo_in_B):

   SNdim_B = mpo_in_B.Sdim

   opdim_A = mpo_in_B.Wdim; sdim_A = mps_in_B.Wdim 
   opdim_B = mpo_in_B.Edim; sdim_B = mps_in_B.Edim 

   Wdim_A = mps_in_A.Wdim; SNdim_A = mps_in_A.SNdim 

   return Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B



#NOTE: use mps_in_A, mps_in_B rather than mps_block[site-1], mps_block[site] 
#since it's used in arnoldi_app_op_to_vec(Vi) routine which is a linear operator called iteratively by arnoldi svd
#we need to get_dims_of_mpo_mps inside arnoldi_app_op_to_vec(Vi) each time we run it.

#NOTE: we use mpsX, mpoX module-level copies cause they're called iteratively inside scipy arnoldi svd function! See the following lines from arnoldi_multiply_each_site:
#Def theta = linear operator that acts on an input vector
#theta = LinearOperator(matvec=arnoldi_app_op_to_vec, rmatvec=arnoldi_app_HC_op_to_vec, matmat=arnoldi_mult_mat_by_mat, shape=(SNdim_A*Wdim_A, SNdim_B*sdim_B*opdim_B))
##### PERFORM ARNOLDI SVD ##########
#U, S, VH, sigma_dim = perform_arnoldi_svd(theta, site, required_accuracy, delta_chi, arnoldi_start_dim, nev_TOT, threshold_frac)





#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################
#################### TESTING MODE ONLY: TEST THETA MATRIX ACTION ON VEC FOR ARNOLDI VS LAPACK ###################################


#test input & output vecs: (vo=theta*vi) VS. (vo=L*R*vi)
def test_multiply_each_site(mps_block, mpo_block, site):

   global intermediate_mps
   global mpoX, mpsX

   #introduce intermediate_mps
   if site==0:
      intermediate_mps = mps_site(mps_block[site].SNdim, mps_block[site].Wdim, mps_block[site].Edim*mpo_block[site].Edim)

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mps_block[site], mpo_block[site])

   #For LAPACK, Multiply mps & mpo at site=site
   mps_mpo_product_at_site = np.zeros((SNdim_B, sdim_A*opdim_A, sdim_B*opdim_B), dtype=complex)
   for oB in np.arange(opdim_B):
     for sB in np.arange(sdim_B):
       for oA in np.arange(opdim_A):
         for sA in np.arange(sdim_A):
            mps_mpo_product_at_site[:, sA + oA*sdim_A, sB + oB*sdim_B] = np.dot(mpo_block[site].m[:,:,oA,oB], mps_block[site].m[:,sA,sB])
   
   if site==0:
     #at site=0 initialize intermediate_mps & move to the next step
     intermediate_mps.m = cp.deepcopy(mps_mpo_product_at_site)
   else:

     #CONSTRUCT LAPACK THETA MATRIX #####
  
     #Combine intermediate_mps at site-1 with mps_mpo_product at site
     #to form theta matrix that contains the data of site-1 and site
     theta=np.zeros((Wdim_A*SNdim_A, sdim_B*opdim_B*SNdim_B), dtype=complex)
     temp_prod=np.zeros((Wdim_A, sdim_B*opdim_B), dtype=complex)
     for iA in np.arange(SNdim_A):
       for iB in np.arange(SNdim_B):        
         temp_prod = np.dot(intermediate_mps.m[iA,:,:], mps_mpo_product_at_site[iB,:,:])
         theta[iA*Wdim_A : (iA+1)*Wdim_A , iB*sdim_B*opdim_B : (iB+1)*sdim_B*opdim_B] = temp_prod[0:Wdim_A,0:sdim_B*opdim_B]

     #CONSTRUCT ARNOLDI IMPLICIT THETA ######

     #Write MPO/MPS_block[site] to the module-level storage mpoX/mpsX
     mpoX=mpo_block[site]; mpsX=mps_block[site]
     #Construct the implicit theta
     theta_arnoldi = LinearOperator(matvec=arnoldi_app_op_to_vec, rmatvec=arnoldi_app_HC_op_to_vec, matmat=arnoldi_mult_mat_by_mat, shape=(SNdim_A*Wdim_A, SNdim_B*sdim_B*opdim_B))
    
     #TEST THE ACTION OF LAPACK THETA VS ARNOLDI THETA

     #test vo = theta * vi
     compare_lapack_vs_arnoldi(theta, theta_arnoldi, 'matvec') 
     #HC: test vo = theta^H * vi
     compare_lapack_vs_arnoldi(theta, theta_arnoldi, 'HC_matvec') 
     #mat-mat: test vo = theta*vi where vo,vi = matrices
     compare_lapack_vs_arnoldi(theta, theta_arnoldi, 'matmat') 




#Compare the results of lapack & arpack theta acting on the same vi
#If which_test = False: vo = theta * vi
#If which_test = True: test vo = theta^H * vi
def compare_lapack_vs_arnoldi(theta, theta_arnoldi, which_test): 

  #Generate Vi_test input vec
  Vi_test = generate_input_vec(which_test)
  #lapack vo=theta*vi
  Vo_lapack = test_apply_lapack_theta(theta, Vi_test, which_test)

  if which_test == 'matvec':
      #arnoldi vo = theta * vi: Vo_arnoldi = test_arnoldi_app_op_to_vec(Vi_test)
      Vo_arnoldi = theta_arnoldi.matvec(Vi_test)
  elif which_test == 'HC_matvec':
      #arnoldi vo = theta^H * vi: Vo_arnoldi = test_arnoldi_app_HC_op_to_vec(Vi_test)
      Vo_arnoldi = theta_arnoldi.rmatvec(Vi_test)
  elif which_test == 'matmat':
      #arnoldi matrices vo = theta * vi: Vo_arnoldi = test_arnoldi_mult_mat_by_mat(Vi_test)
      Vo_arnoldi = theta_arnoldi.matmat(Vi_test)


  #### Now test arnoldi & lapack output ####
  
  #sanity check if Vo_arnoldi & Vo_lapack dims are the same
  if np.shape(Vo_arnoldi) == np.shape(Vo_lapack):
     print('Constructing Delta_Vo')
     Delta_Vo = np.zeros(np.shape(Vo_arnoldi), dtype=complex)
  else:
     print('arnoldi dim: ', np.shape(Vo_arnoldi) , 'lapack dim: ', np.shape(Vo_lapack))
     sys.exit('Vo_arnoldi & Vo_lapack dims dont match - exit')
     
  #set entries to zero if very small
  if which_test == 'matmat':
     for i in np.arange(Delta_Vo.shape[0]):
       for j in np.arange(Delta_Vo.shape[1]):
          Delta_Vo[i,j] = Vo_arnoldi[i,j] - Vo_lapack[i,j]
          if np.absolute(Delta_Vo[i,j]) < 10**(-5):
              Delta_Vo[i,j] = 0.0
  else:
     for i in np.arange(np.size(Delta_Vo)):
        Delta_Vo[i] = Vo_arnoldi[i] - Vo_lapack[i]
        if np.absolute(Delta_Vo[i]) < 10**(-5):
            Delta_Vo[i] = 0.0

  #print the diff vector Delta_Vo
  if which_test == 'matvec':

     if np.absolute(np.sum(Delta_Vo)) == 0.0:
        if np.absolute(np.sum(Vo_arnoldi)) != 0.0:
           if np.absolute(np.sum(Vo_lapack)) != 0.0:
              print('TEST PASSED: ARNOLDI = LAPACK')
           else:
              sys.exit('TEST FAILED: Vo_lapack is 0')   
        else:
          sys.exit('TEST FAILED: Vo_arnoldi is 0')
     else:
       sys.exit('TEST FAILED: Vo_arnoldi & Vo_lapack not equal')

  elif which_test == 'HC_matvec':

     if np.absolute(np.sum(Delta_Vo)) == 0.0:
        if np.absolute(np.sum(Vo_arnoldi)) != 0.0:
           if np.absolute(np.sum(Vo_lapack)) != 0.0:
              print('TEST PASSED: ARNOLDI = LAPACK (HC)')
           else:
              sys.exit('TEST FAILED: Vo_lapack is 0 (HC)')   
        else:
          sys.exit('TEST FAILED: Vo_arnoldi is 0 (HC)')
     else:
       sys.exit('TEST FAILED: Vo_arnoldi & Vo_lapack not equal (HC)')

  elif which_test == 'matmat':

     if np.absolute(np.sum(Delta_Vo)) == 0.0:
        if np.absolute(np.sum(Vo_arnoldi)) != 0.0:
           if np.absolute(np.sum(Vo_lapack)) != 0.0:
              print('TEST PASSED: ARNOLDI = LAPACK (MATMAT)')
           else:
              sys.exit('TEST FAILED: Vo_lapack is 0 (MATMAT)')   
        else:
          sys.exit('TEST FAILED: Vo_arnoldi is 0 (MATMAT)')
     else:
       sys.exit('TEST FAILED: Vo_arnoldi & Vo_lapack not equal (MATMAT)')



 

#Construct the input vec Vi
def generate_input_vec(which_test):

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mpsX, mpoX)

   #Determine the dims of Vi & Vo:
   if which_test == 'matvec':
       vdim = SNdim_B*sdim_B*opdim_B
   elif which_test == 'HC_matvec':
       vdim = SNdim_A*Wdim_A
   elif which_test == 'matmat':
       vdim = np.array([SNdim_B*sdim_B*opdim_B, sdim_A])

   #Generate a random Vi input vec:
   if which_test == 'matmat':
      Vi_test = np.random.rand(vdim[0], vdim[1]) + np.random.rand(vdim[0], vdim[1])*1j
   else:
      Vi_test = np.random.rand(vdim) + np.random.rand(vdim)*1j

   return Vi_test


 #Apply lapack theta to a vector
def test_apply_lapack_theta(theta, Vi, which_test):

  #act with theta on vi
  if which_test == 'matvec':
      print('Act with lapack theta')
      Vo = np.dot(theta, Vi)
  elif which_test == 'HC_matvec':
      print('Act with lapack theta_HC')
      theta_HC = np.conj(theta.T)
      Vo = np.dot(theta_HC, Vi)
  elif which_test == 'matmat':
      print('Mult theta with Vi matrix')
      Vo = np.dot(theta, Vi)

  return Vo


def test_arnoldi_app_op_to_vec(Vi):

   global Block_Ltemp, Block_Rtemp 
   global Block_Lvec, Block_Rvec

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mpsX, mpoX)

   Block_Ltemp = np.zeros((SNdim_A,Wdim_A), dtype=complex)
   Block_Rtemp = np.zeros((SNdim_B,sdim_B*opdim_B), dtype=complex)
   Block_Lvec = np.zeros((SNdim_A*Wdim_A), dtype=complex) 
   Block_Rvec = np.zeros((SNdim_B*sdim_B*opdim_B), dtype=complex)

   #initialize output vec
   Vo=0.0
   for oA in np.arange(opdim_A):
     for sA in np.arange(sdim_A):
       Block_Ltemp = intermediate_mps.m[:,:,sA+oA*sdim_A]
       for oB in np.arange(opdim_B):
         for sB in np.arange(sdim_B):      
            Block_Rtemp[:, sB+oB*sdim_B] = np.dot(mpoX.m[:,:,oA,oB], mpsX.m[:,sA,sB])

       Block_Lvec = np.reshape(Block_Ltemp, (SNdim_A*Wdim_A))
       Block_Rvec = np.reshape(Block_Rtemp, (SNdim_B*sdim_B*opdim_B))
      
       Vo = Vo + Block_Lvec*np.dot(Vi, Block_Rvec)

   return Vo

def test_arnoldi_app_HC_op_to_vec(Vi):

   global Block_Ltemp, Block_Rtemp 
   global Block_Lvec, Block_Rvec, Block_vc

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mpsX, mpoX)

   Block_Ltemp = np.zeros((SNdim_A,Wdim_A), dtype=complex)
   Block_Rtemp = np.zeros((SNdim_B,sdim_B*opdim_B), dtype=complex)
   Block_Lvec = np.zeros((SNdim_A*Wdim_A), dtype=complex) 
   Block_Rvec = np.zeros((SNdim_B*sdim_B*opdim_B), dtype=complex)

   Vc = np.reshape(np.conj(Vi), (-1))

   #initialize output vec
   Vo=0.0

   for oA in np.arange(opdim_A):
     for sA in np.arange(sdim_A):
       Block_Ltemp = intermediate_mps.m[:,:,sA+oA*sdim_A]
       for oB in np.arange(opdim_B):
         for sB in np.arange(sdim_B):      
            Block_Rtemp[:, sB+oB*sdim_B] = np.dot(mpoX.m[:,:,oA,oB], mpsX.m[:,sA,sB])

       Block_Lvec = np.reshape(Block_Ltemp, (SNdim_A*Wdim_A))
       Block_Rvec = np.reshape(Block_Rtemp, (SNdim_B*sdim_B*opdim_B)) 
   
       Vo = Vo + Block_Rvec*np.dot(Vc, Block_Lvec)

   Vo=np.conj(Vo)

   return Vo

def test_arnoldi_mult_mat_by_mat(Vi):

   global Block_Ltemp, Block_Rtemp 
   global Block_Lvec, Block_Rvec

   #find dims of the tensor objects we're dealing with
   Wdim_A, SNdim_A, SNdim_B, opdim_A, sdim_A, opdim_B, sdim_B = get_dims_of_mpo_mps(intermediate_mps, mpsX, mpoX)

   Block_Ltemp = np.zeros((SNdim_A,Wdim_A), dtype=complex)
   Block_Rtemp = np.zeros((SNdim_B,sdim_B*opdim_B), dtype=complex)
   Block_Lvec = np.zeros((SNdim_A*Wdim_A), dtype=complex) 
   Block_Rvec = np.zeros((SNdim_B*sdim_B*opdim_B), dtype=complex)

   #initialize the output matrix
   Vo=0.0
   for oA in np.arange(opdim_A):
     for sA in np.arange(sdim_A):
       Block_Ltemp = intermediate_mps.m[:,:,sA+oA*sdim_A]
       for oB in np.arange(opdim_B):
         for sB in np.arange(sdim_B):      
            Block_Rtemp[:, sB+oB*sdim_B] = np.dot(mpoX.m[:,:,oA,oB], mpsX.m[:,sA,sB])

       Block_Lvec = np.reshape(Block_Ltemp, (SNdim_A*Wdim_A))
       Block_Rvec = np.reshape(Block_Rtemp, (SNdim_B*sdim_B*opdim_B))      
       Vo = Vo + np.outer(Block_Lvec, np.dot(Block_Rvec, Vi))

   return Vo













































        
