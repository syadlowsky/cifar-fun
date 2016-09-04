import time
import logging
import math
import os
import argparse

import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe, Lock

from theano import function, config, shared, sandbox
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import theano.tensor as T

from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool, AvgPool

import numpy as np
from numpy.random import rand as nprand
import scipy.linalg
import SharedArray as sa

from sklearn import metrics

#WARNING FOR AVERAGE POOLING THIS RELIES ON THIS FORK OF PYLEARN2:
# https://github.com/Vaishaal/pylearn2

#logging.getLogger('theano.gof.cmodule').setLevel(logging.DEBUG)

from proto import data_pb2

def load_dataset(base_dir):
    train_filename = os.path.join(base_dir, 'train.binaryproto')
    test_filename = os.path.join(base_dir, 'test.binaryproto')

    def load_from_binaryproto(filename):
        X = []
        y = []
        dataset = data_pb2.Data()
        with open(filename, 'rb') as f:
            dataset.ParseFromString(f.read())
        for datum in dataset.entry:
            image = np.frombuffer(datum.data, dtype=np.float32) \
                      .reshape(datum.channels, datum.height, datum.width)
            X.append(image)
            y.append(datum.label.ordinal_label)
        return np.array(X), np.array(y).astype(np.int32)

    # We can now download and read the training and test set images and labels.
    X_train, y_train = load_from_binaryproto(train_filename)
    X_test, y_test = load_from_binaryproto(test_filename)

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def conv(data, filter_gen, feature_batch_size, num_feature_batches,
         data_batch_size, cuda_convnet=True, symmetric_relu=True,
         start_feature_batch=0, pool_type='avg', pool_size=14, pad=0, bias=1.0,
         ps=6):
    cuda_convnet = True
    outX = int(math.ceil((data.shape[2] - ps + 1)/float(pool_size)))
    outY = int(math.ceil((data.shape[3] - ps + 1)/float(pool_size)))
    outFilters = feature_batch_size*num_feature_batches
    if (symmetric_relu):
        outFilters = 2*outFilters

    print "Out Shape ", outX, "x", outY, "x", outFilters
    XFinal = nprand(data.shape[0], outFilters, outX, outY).astype(np.float32)
    filters = []
    numImages = data.shape[0]
    # Convert to cuda-convnet order
    if (cuda_convnet):
        data = data.transpose(1,2,3,0)

    # POOL OP CREATION
    if (cuda_convnet):
        if (pool_type == 'avg'):
            pool_op = AvgPool(ds=pool_size, stride=pool_size)
        elif (pool_type == 'max'):
            pool_op = MaxPool(ds=pool_size, stride=pool_size)
        else:
            raise Exception('Unsupported pool type')

    else:
        pool_op = lambda X: T.signal.pool.pool_2d(X, (pool_size, pool_size), ignore_border=False, mode='max')

    if (cuda_convnet):
        conv_op = FilterActs(pad=pad)
    else:
        conv_op = lambda X, F: T.nnet.conv2d(X, F)

    CHANNEL_AXIS = 1
    for j in range(num_feature_batches):
        F = filter_gen(feature_batch_size)
        if (cuda_convnet):
            F = F.transpose(1,2,3,0)
            CHANNEL_AXIS = 0

        filters.append(F)

        start_filters = j*feature_batch_size
        end_filters = (j+1)*feature_batch_size

        if symmetric_relu:
            start_filters *= 2
            end_filters *= 2

        for i in range(int(np.ceil(numImages/float(data_batch_size)))):
                start = i*data_batch_size
                end = min((i+1)*data_batch_size, numImages)
                print "FEATURE BATCH #", (j + start_feature_batch), "DATA BATCH #", i,  " SIZE IS ", end - start

    filters = np.concatenate(filters,axis=0)
    return (XFinal, filters)

def preprocess(train, test, min_divisor=1e-8, zca_bias=0.1):
    origTrainShape = train.shape
    origTestShape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1)
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1)


    print "PRE PROCESSING"
    nTrain = train.shape[0]

    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:,np.newaxis]
    test = test - np.mean(test, axis=1)[:,np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train, axis=1)/55.0
    test_norms = np.linalg.norm(test, axis=1)/55.0

    # Get rid of really small norms
    train_norms[np.where(train_norms < min_divisor)] = 1
    test_norms[np.where(test_norms < min_divisor)] = 1

    # Make features unit norm
    train = train/train_norms[:,np.newaxis]
    test = test/test_norms[:,np.newaxis]


    whitening_means = np.mean(train, axis=0)
    data_means = np.mean(train, axis=1)


    zeroCenterTrain = (train - whitening_means[np.newaxis, :])

    trainCovMat = 1.0/nTrain * zeroCenterTrain.T.dot(zeroCenterTrain)

    (E,V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    print global_ZCA[:4,:4]

    train = (train - whitening_means).dot(global_ZCA)
    test = (test - whitening_means).dot(global_ZCA)

    return (train.reshape(origTrainShape), test.reshape(origTestShape))


def featurizeTrainAndEvaluateDualModelAsync(
        XTrain, XTest, labelsTrain, labelsTest, filter_gen, solve=False,
        num_feature_batches=1, solve_every_iter=1, regs=[0.1], pool_size=14,
        FEATURE_BATCH_SIZE=1024, CUDA_CONVNET=True, DATA_BATCH_SIZE=1280):
    print("RELOADING MOTHER FUCKER 3")
    n_train = XTrain.shape[0]
    X = np.vstack((XTrain, XTest))
    parent, child = Pipe()
    try:
        sa.delete("shm://xbatch")
        sa.delete("shm://trainKernel")
        sa.delete("shm://testKernel")
    except:
        pass
    XBatchShared = sa.create("shm://xbatch", (X.shape[0],FEATURE_BATCH_SIZE*8), dtype='float32')
    trainKernelShared = sa.create("shm://trainKernel", (XTrain.shape[0], XTrain.shape[0]), dtype='float32')
    testKernelShared = sa.create("shm://testKernel", (XTest.shape[0], XTrain.shape[0]), dtype='float32')

    trainKernelLocal = np.zeros((XTrain.shape[0], XTrain.shape[0]))
    testKernelLocal = np.zeros((XTest.shape[0], XTrain.shape[0]))

    finish_lock = Lock()

    p = Process(target=accumulateGramAndSolveAsync,
                args=(child,XTrain.shape[0], XTest.shape[0], regs, labelsTrain,
                      labelsTest, solve, finish_lock))
    p.start()
    try:
        for i in range(1, (num_feature_batches + 1)):
            print "A"
            print i
            print("Convolving features")
            time1 = time.time()
            (XBatch, filters) = conv(
                X, filter_gen, FEATURE_BATCH_SIZE, 1, DATA_BATCH_SIZE,
                CUDA_CONVNET, symmetric_relu=True, start_feature_batch=i-1,
                pool_size=pool_size)
            time2 = time.time()
            print 'Convolving features took {0} seconds'.format((time2-time1))
            print("Sending features")
            time1 = time.time()
            np.copyto(XBatchShared, XBatch.reshape(XBatch.shape[0], -1))
            parent.send(i)
            time2 = time.time()
            print 'Sending features took {0} seconds'.format((time2-time1))
            print "B"
        print "C"
        parent.send(-1)
        print "D"
        finish_lock.acquire()
        #child.recv()
        print "E"
        print("Receiving kernel from child")
        print "F"
        np.copyto(trainKernelLocal, trainKernelShared)
        np.copyto(testKernelLocal, testKernelShared)
        print "G"
        parent.close()
        child.close()
        sa.delete("shm://xbatch")
        sa.delete("shm://trainKernel")
        sa.delete("shm://testKernel")
        return trainKernelLocal, testKernelLocal
    except (KeyboardInterrupt, SystemExit):
        sa.delete("shm://xbatch")
        sa.delete("shm://trainKernel")
        sa.delete("shm://testKernel")
        parent.send(-1)
        parent.close()
        child.close()
        raise


def accumulateGramAndSolveAsync(
        pipe, numTrain, numTest, regs, labelsTrain, labelsTest, solve=False, finish_lock=None):
    finish_lock.acquire()
    trainKernel = np.zeros((numTrain, numTrain), dtype='float32')
    testKernel= np.zeros((numTest, numTrain), dtype='float32')
    XBatchShared = sa.attach("shm://xbatch")
    trainKernelShared = sa.attach("shm://trainKernel")
    testKernelShared = sa.attach("shm://testKernel")

    # Local copy
    XBatchLocal = np.zeros(XBatchShared.shape, dtype='float32')
    print("CHILD Process Spun")
    TOT_FEAT = 0
    while(True):
        print "bar"
        m = pipe.recv()
        print m
        if m == -1:
            break
        time1 = time.time()
        print("Receiving (ASYNC) Batch {0}".format(m))
        np.copyto(XBatchLocal, XBatchShared)
        time2 = time.time()
        print 'Receiving (ASYNC) took {0} seconds'.format((time2-time1))
        XBatchTrain = XBatchLocal[:numTrain,:]
        XBatchTest = XBatchLocal[numTrain:,:]
        print("XBATCH DTYPE " + str(XBatchTest.dtype))
        print("Accumulating (ASYNC) Gram")
        time1 = time.time()
        TOT_FEAT += XBatchTrain.shape[1]
        trainKernel += XBatchTrain.dot(XBatchTrain.T)
        testKernel += XBatchTest.dot(XBatchTrain.T)
        time2 = time.time()
        print "baz"
        print 'Accumulating (ASYNC) Batch {1} gram took {0} seconds'.format((time2-time1), m)

    finish_lock.release()
    return None


def patchify_all_imgs(X, patch_shape, pad=True, pad_mode='constant', cval=0):
    out = []
    print X.shape
    X = X.transpose(0,2,3,1)
    i = 0
    for x in X:
        dim = x.shape[0]
        patches = patchify(x, patch_shape, pad, pad_mode, cval)
        out_shape = patches.shape
        out.append(patches.reshape(out_shape[0]*out_shape[1], patch_shape[0], patch_shape[1], -1))
    return np.array(out)

def patchify(img, patch_shape, pad=True, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always

    if pad:
        pad_size= (patch_shape[0]/2, patch_shape[0]/2)
        img = np.pad(img, (pad_size, pad_size, (0,0)),  mode=pad_mode, constant_values=cval)

    img = np.ascontiguousarray(img)  # won't make a copy if not needed

    X, Y, Z = img.shape
    x, y= patch_shape
    shape = ((X-x+1), (Y-y+1), x, y, Z) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
#    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y*Z, Z, Y*Z, Z, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches

def make_empirical_filter_gen(patches, labels, MIN_VAR_TOL=0):
    patches = patches.reshape(patches.shape[0]*patches.shape[1],
                              *patches.shape[2:])
    all_idxs = np.random.choice(patches.shape[0], patches.shape[0],
                                replace=False)
    curr_idx = [0]
    def empirical_filter_gen(num_filters):
        idxs = all_idxs[curr_idx[0]:curr_idx[0]+num_filters]
        curr_idx[0] += num_filters
        unfiltered = patches[idxs].astype('float32').transpose(0,3,1,2)
        old_shape = unfiltered.shape
        unfiltered = unfiltered.reshape(unfiltered.shape[0], -1)
        unfiltered_vars = np.var(unfiltered, axis=1)
        filtered = unfiltered[np.where(unfiltered_vars > MIN_VAR_TOL)]
        out = filtered[:num_filters].reshape(num_filters, *old_shape[1:])
        return out
    return empirical_filter_gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train simple deep nets with robust optimization objective.')
    parser.add_argument('--dataset_dir', default="",
                        type=str, help="Path to folder in which test and train protos can be found")

    args = parser.parse_args()
    # Load CIFAR

    NUM_FEATURE_BATCHES=300
    DATA_BATCH_SIZE=(1280)
    FEATURE_BATCH_SIZE=(1024)
    NUM_CLASSES = 10
    POOL_TYPE ='avg'
    FILTER_GEN ='empirical'
    BANDWIDTH = 1.0
    LAMBDAS = [1e-1/FEATURE_BATCH_SIZE, 1e-2/FEATURE_BATCH_SIZE, 1e-3/FEATURE_BATCH_SIZE, 1e-4/FEATURE_BATCH_SIZE, 1e-5/FEATURE_BATCH_SIZE]
    CUDA_CONVNET = True
    SCALE = 55.0
    BIAS = 1.25
    MIN_VAR_TOL = 1e-4
    TOT_FEAT = FEATURE_BATCH_SIZE*NUM_FEATURE_BATCHES

    np.random.seed(10)
    (XTrain, labelsTrain), (XVal, labelsVal), (XTest, labelsTest) \
        = load_dataset(args.dataset_dir)

    # Jaded much?
    XTrain = np.vstack((XTrain, XVal))
    labelsTrain = np.concatenate((labelsTrain, labelsVal))

    (XTrain, XTest) = preprocess(XTrain, XTest)
    patches = patchify_all_imgs(XTrain, (6,6), pad=False)
    filter_gen = make_empirical_filter_gen(patches, labelsTrain)

    featurizeTrainAndEvaluateDualModelAsync(
	XTrain, XTest, labelsTrain, labelsTest, filter_gen,
	num_feature_batches=NUM_FEATURE_BATCHES,
        solve_every_iter=NUM_FEATURE_BATCHES/4, regs=LAMBDAS, solve=False)
