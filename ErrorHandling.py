from __future__ import print_function
import math
import sys
import copy as cp
import numpy as np



### Classes for Error Handling ###
class Error(Exception):
    # Base class for exceptions
    pass


class MpoSiteShapeError(Error):

 def __init__(self):
     self.msg="tens_in must be a 4D object"



class MpoSiteInputError(Error):

 def __init__(self):
     self.msg="must provide either tens_in or a set of dimensions but not both"



class MpsSiteShapeError(Error):

 def __init__(self):
     self.msg="tens_in must be a 3D object"



class MpsSiteInputError(Error):

 def __init__(self):
     self.msg="must provide either tens_in or a set of dimensions but not both"



class MpsAppendingError(Error):

 def __init__(self):
     self.msg="tensor_to_append must have Wdim = 1"




class TruncModeError(Error):

 def __init__(self):
     self.msg="trunc_mode must be 'accuracy', 'chi', or 'fraction'"



class EpsModeError(Error):

 def __init__(self):
     self.msg="prec must be between 0 and 1 in 'accuracy' mode"



class SigmaDimError(Error):

 def __init__(self):
     self.msg="prec must not exceed sigma_dim in 'chi' mode: setting chi = sigma_dim"



class ChiModeError(Error):

 def __init__(self):
     self.msg="prec must be a positive integer in 'chi' mode"



class FracModeError(Error):

 def __init__(self):
     self.msg="prec must be between 0 and 1 in 'fraction' mode"



class ChiError(Error):

 def __init__(self):
     self.msg="chi must be a positive integer"
 




