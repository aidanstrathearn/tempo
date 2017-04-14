from pytensor import sptensor, sptenmat

import numpy


#create sparse tensor
vals = numpy.array([[0.5 + 3.2*1j], [1.5], [2.5], [3.5 + 0.3*1j], [4.5], [5.5]])
subs = numpy.array([[1,1,1], [0,0,0], [1,2,3], [1,0,1], [1,1,2], [3,1,1]])

sptest = sptensor(subs, vals)

#convert to matrix by flattening dimensions 1 and 2 (and keeping dimension 0)
mattest = sptenmat(sptest, [0])

#convert to scipy sparse matrix
coo_test = mattest.tosparsemat()

print(coo_test.data)
print(coo_test.col)
print(coo_test.row)
