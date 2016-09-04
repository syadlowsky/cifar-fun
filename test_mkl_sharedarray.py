import numpy as np
from numpy.random import rand as nprand
import scipy.linalg
import SharedArray as sa
import time

n = 50000
m = 10000
k = 1024

try:
    sa.delete("shm://xbatch")
    sa.delete("shm://trainKernel")
    sa.delete("shm://testKernel")
except:
    pass
XBatchShared = sa.create("shm://xbatch", (n+m,k*8), dtype='float32')
trainKernelShared = sa.create("shm://trainKernel", (n, n), dtype='float32')
testKernelShared = sa.create("shm://testKernel", (m, m), dtype='float32')

trainKernel = np.zeros((n, n), dtype='float32')
testKernel= np.zeros((m, n), dtype='float32')
for i in range(100):
    print i
    t1 = time.time()
    XBatchShared = sa.attach("shm://xbatch")
    trainKernelShared = sa.attach("shm://trainKernel")
    testKernelShared = sa.attach("shm://testKernel")
    np.copyto(XBatchShared, nprand(*XBatchShared.shape).astype(np.float32))
    
    # Local copy
    XBatchLocal = np.zeros(XBatchShared.shape, dtype='float32')
    np.copyto(XBatchLocal, XBatchShared)
    XBatchTrain = XBatchLocal[:n,:]
    XBatchTest = XBatchLocal[n:,:]
    trainKernel += XBatchTrain.dot(XBatchTrain.T)
    testKernel += XBatchTest.dot(XBatchTrain.T)
    print time.time() - t1
print "done"
