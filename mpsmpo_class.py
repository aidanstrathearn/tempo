#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:15:33 2018

@author: aidan, dainius
"""
import sys
from numpy import dot, swapaxes, ceil, expand_dims, reshape, linalg
from numpy import sum as nsum
from scipy.linalg import svd as la_svd
from multiprocessing.dummy import Queue, Process
from time import time
from pathos.multiprocessing import ProcessingPool as Pool

#==============================================================================
# Note we refer to tensors here as having North/South/East/West legs -- graphically these labels
# have the usual meaning (North=Up, West=Left etc.) when the tensor network 
# diagrams in the paper are rotated 180 degrees
# 
# That is, our ADT/MPS legs point downwards here and the 'present' timepoint leg is 
# to the left -- python list element 0 in list language
#==============================================================================

#
#
#NOTE: The reshape function retains the order of axes whithin combined axes
#example: 
#the 3-leg(axis) tensor Tabc with dimensions da,db,dc can be reshaped to a 2-leg(axis)matrix M{ab}c=reshape(Tabc,(da*db,dc))
#where {ab} denotes the combined a/b legs(axes). Now take the transpose Mc{ab}=M{ab}c.T
#Because reshape retains the order of indices, if we want to shape the transpose back into a
#3-leg tensor we have to do it like Tcab=reshape(Mc{ab},(dc,da,db)), which is not the same order 
#of a,b,c we started with! to get back to the same order you would then need to use
#the swapaxes function Tabc=swapaxes(swapaxes(Tcab,(0,1)),(1,2)), also plenty of these below
#
#ALSO NOTE: a '-1' in reshape argument is telling it to work out what the dimension should be
#given the other dimensions involved. This still retains the ordering!


######################################################################################################
##########################################  SITE CLASSES  ###########################################
######################################################################################################

class mpo_site(object):

 def __init__(self, tens = None):

   if len(tens.shape) != 4:
       print('__init__: mpo site needs 4 legs')
       sys.exit()
   #get dims from tens & set mpo_site to tens 
   [self.Sdim,self.Ndim,self.Wdim,self.Edim]=tens.shape
   self.m = tens

 def update(self, tens = None):

   if len(tens.shape) != 4: 
       print('update: mpo site needs 4 legs')
       sys.exit()
   #get dims from tens & set mpo_site to tens 
   [self.Sdim,self.Ndim,self.Wdim,self.Edim]=tens.shape
   self.m = tens

class mps_site(object):

 def __init__(self,tens = None):

   if len(tens.shape) != 3: 
       print('__init__: mps site needs 3 legs')
       sys.exit()
   #get dims from tens & set mps_site to tens 
   [self.SNdim,self.Wdim,self.Edim]=tens.shape
   self.m = tens

 def update(self, tens = None):

   if len(tens.shape) != 3:
       print('update: mps site needs 3 legs')
       sys.exit()
   #get dims from tens & set mps_site to tens 
   [self.SNdim,self.Wdim,self.Edim]=tens.shape
   self.m = tens

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
     #first reshape each tensor into a matrix and contract legs using dot: the '-1's in reshape tell it to work out the
     #size of one leg of the matrix given the size of the other and of the original tensor
     tensO=dot(reshape(swapaxes(mposite.m,1,3),(-1,mposite.Ndim)),reshape(self.m,(self.SNdim,-1)))
     #reshape into 4-leg tensor: south leg, 2 east legs, and a single combine west leg
     #east legs need to be beside each other to combine so swap first east with combined west
     tensO=swapaxes(reshape(tensO,(-1, mposite.Edim, mposite.Wdim*self.Wdim, self.Edim)),1,2)
     #finally reshape into 3-leg tensor to combine the easts
     tensO=reshape(tensO,(-1, mposite.Wdim*self.Wdim, mposite.Edim*self.Edim))

     #update the mps site object 
     self.update(tens=tensO)

######################################################################################################
#########################################  BLOCK CLASSES  #########################################
######################################################################################################


class mpo_block(object):

 def __init__(self):
    #set the length of mpo_block
    self.N_sites = 0
    #initialize list of mpo_sites
    self.sites = []
 
 def insert_site(self, axis, tensor_to_append):
   #insert mpo site at given position 'axis' in block
   self.sites.insert(axis,mpo_site(tens = tensor_to_append))
   #record that the block now has 1 more site
   self.N_sites = self.N_sites + 1 

 def reverse_mpo(self):
    #reverse the entire mpo bock 
    #first reverse the list of sites
    self.sites.reverse()
    #then site by site reverse the swap the east and west legs of the sites
    for site in self.sites:
        site.update(tens=swapaxes(site.m,2,3))
 
 def split(self,k):
     #splits the mpo retaining first k sites and returning rest as a new mpo
     newmpo=mpo_block()
     newmpo.sites=self.sites[k:]
     newmpo.N_sites=self.N_sites-k
     self.sites=self.sites[:k]
     self.N_sites=k
     newmpo.reverse_mpo()
     return newmpo
 
 def connect(self,mpo_block):
     #connects on another mpo
     mpo_block.reverse_mpo()
     self.sites=self.sites+mpo_block.sites
     self.N_sites=self.N_sites+mpo_block.N_sites
 
    
class mps_block(object):

 def __init__(self,prec):
    #initialise an mps by stating what precision we are going to keep its bonds truncated to
    self.precision=prec
    #set the length of mps_block
    self.N_sites = 0
    #initialize list of mps_sites
    self.sites = []
 
 def split(self,k):
     #splits the mps retaining first k sites and returning rest as a new mps
     newmps=mps_block(self.precision)
     newmps.sites=self.sites[k:]
     newmps.N_sites=self.N_sites-k
     self.sites=self.sites[:k]
     self.N_sites=k
     newmps.reverse_mps()
     return newmps
 
 def connect(self,mps_block):
     #connects on another mps
     mps_block.reverse_mps()
     self.sites=self.sites+mps_block.sites
     self.N_sites=self.N_sites+mps_block.N_sites
     
 def insert_site(self, axis, tensor_to_append):
   #insert mps site at given position 'axis' in block
   self.sites.insert(axis,mps_site(tens = tensor_to_append))
   #record that the block now has 1 more site
   self.N_sites = self.N_sites + 1 
       
 def reverse_mps(self):
    #reverse the entire mps bock 
    #first reverse the list of sites
    self.sites.reverse()  
    #then site by site reverse the swap the east and west legs of the sites
    for site in self.sites:
        site.update(tens = swapaxes(site.m, 1,2))
        
 def truncate_bond(self,k):
    #If there are N sites then there are only N-1 bonds so...
    if k<1 or k>self.N_sites-1: return 0
    
    #truncates the k'th bond of the MPS using an SVD 
    #                     
    #          ---O---  Edim  ---O---                   
    #             \              \   
    #                    ^
    #  (k-1)'th site     ^       k'th site
    #                    ^
    #                k'th bond
    
    #start by combining south and west legs of (k-1)'th site to give 2-leg tensor which is 
    #the rectangular matrix we will perform the SVD on
    #
    #     Wdim  --O--  Edim  ----->      (Wdim x SNdim) --M-- Edim
    #             \
    #           SNdim                                  'theta'
    #
    theta = reshape(self.sites[k-1].m,(-1,self.sites[k-1].Edim))
    
    #SVD:
    #
    #  dim1 --M-- dim2  ---->  dim1 --U-- min(dim1,dim2) --S-- min(dim1,dim2) --V-- dim2
    #
    #  U, V unitary are matrices (U.Udag=I, V.Vdag=I), S is diagonal matrix of min(dim1,dim2) singular values
    
    #this try and except is because sometimes 'gesvd' fails
    #Could also use Arnoldi SVD here but we found the truncation error was worse
    try:
        U, Sigma, _ = la_svd(theta, full_matrices=False,lapack_driver='gesvd')
    except(linalg.LinAlgError):
        U, Sigma, _ = la_svd(theta, full_matrices=False,lapack_driver='gesdd')

    #Sigma here is list of singular values in non-increasing order rather than a diagonal matrix
        
    #TRUNCATION: this is actually on the unitary U rather than the singular values S
    #We throw away columns of U that get multiplied into entries of the diagonal matrix S which 
    #are smaller than a specified value, to leave chi columns. 
    #Then we can approximate the dim1-dimensional identity matrix
    #
    #   dim1 --I-- dim1  ---->   dim1 --U-- chi --Udag-- dim1  
    #
    #so we can express theta as
    #
    #     (Wdim x SNdim) --M-- Edim    ------>   (Wdim x SNdim) --U-- chi --Udag.M-- Edim
    #
    #loop through finding the position of the first singular value that falls below precision
    #since python ordering starts from 0 the position is the number of SV's we want to keep = chi
    try: chi=next(i for i in range(len(Sigma)) if Sigma[i]/max(Sigma) < self.precision)
    #If no singular values are small enough then keep em all
    except(StopIteration): chi=len(Sigma)
    #Chop off the rows of U which dont get multiplied into the chi values we are keeping
    U=U[:, 0:chi]
    theta=dot(U.conj().T,theta)
    #now retain  (Wdim x SNdim) --U-- chi to become the new (k-1)'th site after reshaping to 
    #separate out west and south legs
    self.sites[k-1].update(tens = reshape(U,(-1,self.sites[k-1].Wdim, chi)))
    #multiply chi --Udag.M-- Edim into the k'th site, practically carried out by converting to a matrix and
    #then using numpy dot
    theta=dot(theta,reshape(swapaxes(self.sites[k].m,0,1), (self.sites[k].Wdim,-1)))
    self.sites[k].update(tens = swapaxes(reshape(theta, (chi,self.sites[k].SNdim,-1)),0,1))
    #Overall then we are left with:
    #                     
    #          ---U---  chi  ---Udag.M---O---    
    #             \                      \   
    #                    ^
    #  (k-1)'th site     ^               k'th site
    #                    ^
    #             truncated k'th bond
    
 def trunc_sweep(self,k,mpo=None):
     #sweep from left to right truncating up to and including kth bond
     #if given an mpo then contract in its sites and then truncate
     if type(mpo)==mpo_block:
         self.sites[0].contract_with_mpo_site(mpo.sites[0])
         for jj in range(1,k+1): 
             self.sites[jj].contract_with_mpo_site(mpo.sites[jj])
             self.truncate_bond(jj)
     else:
         for jj in range(1,k+1): 
             self.truncate_bond(jj)
             
 def contract_with_mpo(self, mpoblk, orth_centre=None):          
    #function to contract an mps with an mpo site by site performing truncations at each site
    #for special case of 1-site mps then just contract in mpo site - no truncations
    if self.N_sites==1:
        self.sites[0].contract_with_mpo_site(mpoblk.sites[0])
        return 0
      
    #default val of orth_centre is the actual centre of the mps
    if orth_centre == None: orth_centre=int(ceil(0.5*self.N_sites))

    #sweep along contracting in the mpo and truncating to orth_centre-1'th bond
    self.trunc_sweep(orth_centre - 1,mpoblk) 
    #now reverse the mps and mpo and repeat as above up until all mpo sites have been contracted in
    #and all but one bond has been truncated
    self.reverse_mps() 
    mpoblk.reverse_mpo()    
    self.trunc_sweep(self.N_sites - orth_centre - 1,mpoblk)
    #truncate last bond that links the two halfs of the mps we have seperately swept through above  
    self.truncate_bond(self.N_sites - orth_centre)
    #before reversing back to another sweep along whole mps (not contracting in mpos this time)
    self.trunc_sweep(self.N_sites)  
    #reverse back and perform another sweep along whole mps
    self.reverse_mps() 
    mpoblk.reverse_mpo()    
    self.trunc_sweep(self.N_sites)
    
 def contract_end(self):
    #sums over one leg of ADT/mps as indicated in figures by contraction with a 1 leg semi circle
    #first sum over south and east legs of end site then dot in to second last site and make
    #3d again by giving 1d dummy index wth expand_dims
    self.sites[-2].update(tens=expand_dims(
            dot(self.sites[-2].m,nsum(self.sites[-1].m,(0,2)))
            ,-1) )
    #delete the useless end site and change N_sites accordingly
    del self.sites[-1]
    self.N_sites=self.N_sites-1

 def readout(self):
    #print(self.bonddims())
     #contracts all but the 'present time' leg of ADT/mps and returns 1-leg reduced density matrix
    #l=len(self.sites)
    #for special case of rank-1 ADT just sum over 1d dummy legs and return
    if self.N_sites==1: return nsum(nsum(self.sites[0].m,-1),-1)
    #other wise sum over all but 1-leg of last site, store as out, then successively
    #sum legs of new end sites to make matrices then multiply into vector 'out'
    out=nsum(nsum(self.sites[self.N_sites-1].m,0),-1)
    for jj in range(self.N_sites-2):
        out=dot(nsum(self.sites[self.N_sites-2-jj].m,0),out)
    out=dot(nsum(self.sites[0].m,1),out)
    #after the last site, 'out' should now be the reduced density matrix
    return out

 def bonddims(self):
     #returns a list of the bond dimensions along the mps
     bond=[]
     for site in self.sites: bond.append([site.SNdim,site.Wdim,site.Edim])
     return bond
          
 def totsize(self):
     #returns the total number of elements (i.e. complex numbers) which make up the mps
     size=0
     for site in self.sites: size = size + site.SNdim*site.Wdim*site.Edim
     return size
 
 def contract_with_mpo2(self, mpoblk):  
    #multithreading/multiprocessing version of contract_with_mpo
    #set for multithreading, to use multiprocessing remove .dummy from 'import multiprocessing.dummy'
    #only works for length 4 or greater
    if self.N_sites<4:
        self.contract_with_mpo(mpoblk)
        return 0
    
    #find the mid point of the mps and mpo and split them in 2
    cen=int(ceil(0.5*self.N_sites))
    mps2=self.split(cen) 
    mpo2=mpoblk.split(cen)
    
    #contract in the first mpo site with the first mps site for each of mps/mpo
    self.sites[0].contract_with_mpo_site(mpoblk.sites[0])
    mps2.sites[0].contract_with_mpo_site(mpo2.sites[0])
    
    #define a truncation function to use with multiprocessing
    #this is basically the same bond truncation function as above so no comments
    #'pre' and 'cur' are the (k-1)th and kth sites of the mps, mpo is the kth mpo site
    #we use 'mark' so that we can tell which of the 2 mps we doing manipulations on
    #q is the multithreading queue where the results end up
    def trun(pre,cur,mpo,mark,q):
        cur.contract_with_mpo_site(mpo)
        theta = reshape(pre.m,(-1,pre.Edim))
        try:
            U, Sigma, _ = la_svd(theta, full_matrices=False,lapack_driver='gesvd')
        except(linalg.LinAlgError):
            U, Sigma, _ = la_svd(theta, full_matrices=False,lapack_driver='gesdd')  
        try: chi=next(i for i in range(len(Sigma)) if Sigma[i]/max(Sigma) < self.precision)
        except(StopIteration): chi=len(Sigma)
        U=U[:, 0:chi]
        theta=dot(U.conj().T,theta)
        pretens = reshape(U,(-1,pre.Wdim, chi))
        theta=dot(theta,reshape(swapaxes(cur.m,0,1), (cur.Wdim,-1)))
        curtens = swapaxes(reshape(theta, (chi,cur.SNdim,-1)),0,1)
        #pre.update(tens=pretens)
        #cur.update(tens=curtens)
        q.put([pretens, curtens,mark])
    
    for jj in range(1,len(mps2.sites)):
        #simultaneoulsy sweep along both mps submitting bond truncations on each as separate processes
        q=Queue()
        p1=Process(target=trun,args=([self.sites[jj-1],self.sites[jj],mpoblk.sites[jj],1,q]))
        p2=Process(target=trun,args=([mps2.sites[jj-1],mps2.sites[jj],mpo2.sites[jj],2,q]))
        #start the processes
        p1.start(); p2.start()
        #get the results from the queue
        sites1=q.get(); sites2=q.get()
        #no guarantee of what order results go into the queue so check the mark
        #and set sites1 and sites2 accordingly
        if sites1[2]==2:
            sites1,sites2=sites2,sites1
        #done with processes so join 
        p1.join(); p2.join()
        #update the sites of both mps with the results of the truncation
        self.sites[jj-1].update(tens=sites1[0])
        self.sites[jj].update(tens=sites1[1])
        
        mps2.sites[jj-1].update(tens=sites2[0])
        mps2.sites[jj].update(tens=sites2[1])
    
    #when the full mps has an odd number of sites there is still one bond
    #left to truncate on the longer of the 2 mps, whish is always 'self'
    if len(self.sites)!=len(mps2.sites):
        q=Queue()
        p=Process(target=trun,args=([self.sites[-2],self.sites[-1],mpoblk.sites[-1],3,q]))
        p.start()
        p.join()
        sites1=q.get()
        self.sites[-2].update(tens=sites1[0])
        self.sites[-1].update(tens=sites1[1])
    
    #connect back together the mps's and mpo's
    self.connect(mps2)
    mpoblk.connect(mpo2)
    
    #finish of the contraction by truncating the linking bond and sweeping back and forth the same as before
    self.reverse_mps()
    self.truncate_bond(self.N_sites - cen-1)
    self.trunc_sweep(self.N_sites)
    self.reverse_mps() 
    self.trunc_sweep(self.N_sites)
