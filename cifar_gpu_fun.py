import time
import logging
import math
import os
import argparse

import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe

from theano import function, config, shared, sandbox
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import theano.tensor as T

# from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
# from pylearn2.sandbox.cuda_convnet.pool import MaxPool, AvgPool

import numpy as np
import scipy.linalg
import SharedArray as sa

from sklearn import metrics

#WARNING FOR AVERAGE POOLING THIS RELIES ON THIS FORK OF PYLEARN2:
# https://github.com/Vaishaal/pylearn2

#logging.getLogger('theano.gof.cmodule').setLevel(logging.DEBUG)

from proto import data_pb2

def load_dataset(base_dir):
    train_filename = os.path.join(base_dir, 'train.binaryproto')
    test_filename = os.path.join(base_dir, 'train.binaryproto')

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
    outX = int(math.ceil((data.shape[2] - ps + 1)/float(pool_size)))
    outY = int(math.ceil((data.shape[3] - ps + 1)/float(pool_size)))
    outFilters = feature_batch_size*num_feature_batches
    if (symmetric_relu):
        outFilters = 2*outFilters

    print "Out Shape ", outX, "x", outY, "x", outFilters
    XFinal = np.zeros((data.shape[0], outFilters, outX, outY), 'float32')
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
        FTheano = shared(F.astype('float32'))

        start_filters = j*feature_batch_size
        end_filters = (j+1)*feature_batch_size

        if symmetric_relu:
            start_filters *= 2
            end_filters *= 2

        for i in range(int(np.ceil(numImages/float(data_batch_size)))):
                start = i*data_batch_size
                end = min((i+1)*data_batch_size, numImages)
                print "FEATURE BATCH #", (j + start_feature_batch), "DATA BATCH #", i,  " SIZE IS ", end - start
                if (cuda_convnet):
                    XBlock = shared(data[:, :, :, start:end])
                else:
                    XBlock = shared(data[start:end, :, :, :])

                if (cuda_convnet):
                    XBlock_gpu = gpu_contiguous(XBlock)
                    FTheano_gpu = gpu_contiguous(FTheano)

                # CONV
                XBlock_conv_out = conv_op(XBlock_gpu, FTheano_gpu)

                # RELU
                XBlock0 = T.nnet.relu(XBlock_conv_out - bias, 0)
                if (symmetric_relu):
                    XBlock1 = T.nnet.relu(-1.0 * XBlock_conv_out - bias, 0)

                XBlock0 = pool_op(XBlock0)
                if (symmetric_relu):
                    XBlock1 = pool_op(XBlock1)
                    XBlockOut = np.concatenate((XBlock0.eval(), XBlock1.eval()), axis=CHANNEL_AXIS)
                else:
                    XBlockOut = np.array(XBlock0.eval())

                if (cuda_convnet):
                    XBlockOut = XBlockOut.transpose(3,0,1,2)
                    F = F.transpose(3,0,1,2)

                XBlock.set_value([[[[]]]])
                XFinal[start:end,start_filters:end_filters,:,:] = XBlockOut
        FTheano.set_value([[[[]]]])

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

def learnPrimal(trainData, labels, reg=0.1):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)

    X = np.ascontiguousarray(trainData, dtype=np.float32).reshape(trainData.shape[0], -1)
    print "X SHAPE ", trainData.shape
    print "Computing XTX"
    XTX = X.T.dot(X)
    print "Done Computing XTX"

    print "REG is " + str(reg)
    idxes = np.diag_indices(XTX.shape[0])
    XTX[idxes] += reg

    y = np.eye(max(labels) + 1)[labels]
    XTy = X.T.dot(y)

    print "Learning Primal Model"
    model = scipy.linalg.solve(XTX, XTy)
    return model

def learnDual(gramMatrix, labels, reg=0.1, TOT_FEAT=1, NUM_TRAIN=1):
    ''' Learn a model from K matrix -> labels '''
    print ("Learning Dual Model")
    y = np.eye(max(labels) + 1)[labels]
    idxes = np.diag_indices(gramMatrix.shape[0])
    gramMatrix /= float(TOT_FEAT)
    gramMatrix[idxes] += (NUM_TRAIN * reg)
    model = scipy.linalg.solve(gramMatrix + NUM_TRAIN * reg * np.eye(gramMatrix.shape[0]), y)
    gramMatrix[idxes] -= (NUM_TRAIN * reg)
    gramMatrix *= TOT_FEAT
    return model

def evaluatePrimalModel(data, model):
    data = data.reshape(data.shape[0],-1)
    yHat = np.argmax(data.dot(model), axis=1)
    return yHat


def evaluateDualModel(kMatrix, model, TOT_FEAT=1):
    print("MODEL SHAPE " + str(model.shape))
    print("KERNEL SHAPE " + str(kMatrix.shape))
    kMatrix *= TOT_FEAT
    y = kMatrix.dot(model)
    kMatrix /= TOT_FEAT
    print("pred SHAPE " + str(y.shape))
    yHat = np.argmax(y, axis=1)
    return yHat

def trainAndEvaluateDualModel(XTrain, XTest, labelsTrain, labelsTest, reg=0.1):
    K = XTrain.dot(XTrain.T)
    KTest = XTest.dot(XTrain)
    model = learnDual(K,labelsTrain, reg=reg)
    predTrainLabels = evaluateDualModel(K, model)
    predTestLabels = evaluateDualModel(KTest, model)
    train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
    test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
    return train_acc, test_acc


def trainAndEvaluatePrimalModel(XTrain, XTest, labelsTrain, labelsTest,
                                reg=0.1):
    model = learnPrimal(XTrain, labelsTrain, reg=reg)
    predTrainLabels = evaluatePrimalModel(XTrain, model)
    predTestLabels = evaluatePrimalModel(XTest, model)
    train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
    test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
    print "CONFUSION MATRIX"
    print metrics.confusion_matrix(labelsTest, predTestLabels)
    return train_acc, test_acc

def featurizeTrainAndEvaluateDualModel(XTrain, XTest, labelsTrain, labelsTest,
                                       filter_gen, num_feature_batches=1,
                                       solve_every_iter=1, reg=0.1):
    trainKernel = np.zeros((XTrain.shape[0], XTrain.shape[0]),dtype='float32')
    testKernel= np.zeros((XTest.shape[0], XTrain.shape[0]),dtype='float32')
    for i in range(1, (num_feature_batches + 1)):
        X = np.vstack((XTrain, XTest))
        print("Convolving features")
        time1 = time.time()
        (XBatch, filters) = conv(X, filter_gen, FEATURE_BATCH_SIZE, 1,
                                 DATA_BATCH_SIZE, CUDA_CONVNET,
                                 symmetric_relu=True, start_feature_batch=i-1,
                                 pool_type=POOL_TYPE)
        time2 = time.time()
        print 'Convolving features took {0} seconds'.format((time2-time1))
        XBatchTrain = XBatch[:50000,:,:,:].reshape(NUM_TRAIN,-1)
        XBatchTest = XBatch[50000:,:,:,:].reshape(NUM_TEST,-1)
        print("Accumulating Gram")
        time1 = time.time()
        trainKernel += XBatchTrain.dot(XBatchTrain.T)
        testKernel += XBatchTest.dot(XBatchTrain.T)
        time2 = time.time()
        print 'Accumulating gram took {0} seconds'.format((time2-time1))
        if ((i % solve_every_iter == 0) or i == num_feature_batches - 1):
            time1 = time.time()
            model = learnDual(trainKernel, labelsTrain, reg)
            time2 = time.time()
            print 'learningDual took {0} seconds'.format((time2-time1))
            predTrainLabels = evaluateDualModel(trainKernel, model)
            predTestLabels = evaluateDualModel(testKernel, model)
            print("true shape " + str(labelsTrain.shape))
            print("pred shape " + str(predTrainLabels.shape))
            train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
            test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
            print "(dual conv #{batchNo}) train: , {convTrainAcc}, (dual conv batch #{batchNo}) test: {convTestAcc}".format(batchNo=i, convTrainAcc=train_acc, convTestAcc=test_acc)
    return train_acc, test_acc

def featurizeTrainAndEvaluateDualModelAsync(
        XTrain, XTest, labelsTrain, labelsTest, filter_gen, solve=False,
        num_feature_batches=1, solve_every_iter=1, regs=[0.1], pool_size=14,
        FEATURE_BATCH_SIZE=1024, CUDA_CONVNET=True, DATA_BATCH_SIZE=1280):
    print("RELOADING MOTHER FUCKER 3")
    X = np.vstack((XTrain, XTest))
    parent, child = Pipe()
    XBatchShared = sa.create("shm://xbatch", (X.shape[0],FEATURE_BATCH_SIZE*8), dtype='float32')
    trainKernelShared = sa.create("shm://trainKernel", (XTrain.shape[0], XTrain.shape[0]), dtype='float32')
    testKernelShared = sa.create("shm://testKernel", (XTest.shape[0], XTrain.shape[0]), dtype='float32')

    trainKernelLocal = np.zeros((XTrain.shape[0], XTrain.shape[0]))
    testKernelLocal = np.zeros((XTest.shape[0], XTrain.shape[0]))

    p = Process(target=accumulateGramAndSolveAsync,
                args=(child,XTrain.shape[0], XTest.shape[0], regs, labelsTrain,
                      labelsTest, solve))
    p.start()
    try:
        for i in range(1, (num_feature_batches + 1)):
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
        parent.send(-1)
        child.recv()
        print("Receiving kernel from child")
        np.copyto(trainKernelLocal, trainKernelShared)
        np.copyto(testKernelLocal, testKernelShared)
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
        pipe, numTrain, numTest, regs, labelsTrain, labelsTest, solve=False):
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
        m = pipe.recv()
        if m == -1:
            break
        time1 = time.time()
        print("Receiving (ASYNC) Batch {0}".format(m))
        np.copyto(XBatchLocal, XBatchShared)
        time2 = time.time()
        print 'Receiving (ASYNC) took {0} seconds'.format((time2-time1))
        XBatchTrain = XBatchLocal[:50000,:]
        XBatchTest = XBatchLocal[50000:,:]
        print("XBATCH DTYPE " + str(XBatchTest.dtype))
        print("Accumulating (ASYNC) Gram")
        time1 = time.time()
        TOT_FEAT += XBatchTrain.shape[1]
        trainKernel += XBatchTrain.dot(XBatchTrain.T)
        testKernel += XBatchTest.dot(XBatchTrain.T)
        time2 = time.time()
        print 'Accumulating (ASYNC) Batch {1} gram took {0} seconds'.format((time2-time1), m)

    train_accs = []
    test_accs = []
    if (solve):
        for reg in regs:
            time1 = time.time()
            print 'learningDual (ASYNC) reg: {reg}'.format(reg=reg)
            model = learnDual(trainKernel, labelsTrain, reg, TOT_FEAT=TOT_FEAT,
                              NUM_TRAIN=labelsTrain.shape[0])
            time2 = time.time()
            print 'learningDual (ASYNC) reg: {reg} took {0} seconds'.format((time2-time1), reg=reg)
            predTrainLabels = evaluateDualModel(
                trainKernel, model, TOT_FEAT=TOT_FEAT)
            predTestLabels = evaluateDualModel(
                testKernel, model, TOT_FEAT=TOT_FEAT)
            print("true shape " + str(labelsTrain.shape))
            print("pred shape " + str(predTrainLabels.shape))
            train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
            test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
            print "(async dual conv reg: {reg}) train: , {convTrainAcc}, (dual conv batch) test: {convTestAcc}".format(convTrainAcc=train_acc, convTestAcc=test_acc, reg=reg)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
    return train_accs, test_accs


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

def make_balanced_empirical_filter_gen(patches, labels):
    ''' NUM_FILTERS MUST BE DIVISIBLE BY NUM_CLASSES '''
    def empirical_filter_gen(num_filters):
        filters = []
        for c in range(NUM_CLASSES):
            patch_ss = patches[np.where(labels == c)]
            patch_ss = patch_ss.reshape(patch_ss.shape[0]*patch_ss.shape[1],
                                        *patch_ss.shape[2:])
            idxs = np.random.choice(patch_ss.shape[0], num_filters/NUM_CLASSES,
                                    replace=False)
            unfiltered = patch_ss[idxs].astype('float32').transpose(0,3,1,2)
            old_shape = unfiltered.shape
            unfiltered = unfiltered.reshape(unfiltered.shape[0], -1)
            unfiltered_vars = np.var(unfiltered, axis=1)
            filtered = unfiltered[np.where(unfiltered_vars > MIN_VAR_TOL)]
            out = filtered[:num_filters].reshape(num_filters/NUM_CLASSES,
                                                 *old_shape[1:])
            filters.append(out)
        return np.concatenate(filters, axis=0)
    return empirical_filter_gen

def estimate_bandwidth(patches):
    patch_norms = np.linalg.norm(patches.reshape(patches.shape[0], -1), axis=1)
    return np.median(patch_norms)

def make_gaussian_filter_gen(bandwidth, patch_size=6, channels=3):
    ps = patch_size
    def gaussian_filter_gen(num_filters):
        out = np.random.randn(
            num_filters, channels, ps, ps).astype('float32') * bandwidth
        print out.shape
        return out
    return gaussian_filter_gen

def make_gaussian_cov_filter_gen(patches, sub_sample=100000):
    patches = patches.reshape(patches.shape[0]*patches.shape[1],
                              *patches.shape[2:])
    idxs = np.random.choice(patches.shape[0], sub_sample, replace=False)
    patches = patches[idxs, :, :, :]
    patches = patches.reshape(patches.shape[0], -1)
    means = patches.mean(axis=0)[:,np.newaxis]
    covMatrix = 1.0/(patches.shape[0]) \
                * patches.T.dot(patches) - means.dot(means.T)
    covMatrixRoot = np.linalg.cholesky(covMatrix).astype('float32')
    print(covMatrixRoot.shape)
    def gaussian_filter_gen(num_filters):
        out = np.random.randn(num_filters, 3*6*6).astype(
            'float32').dot(covMatrixRoot)
        return out.reshape(out.shape[0], 3, 6, 6)
    return gaussian_filter_gen

def make_gaussian_cc_cov_filter_gen(patches, labels, patch_size=6, channels=3,
                                    sub_samples=10000):
    ''' NUM_FILTERS MUST BE DIVISBLE BY NUM_CLASSES '''
    covMatrixRoots = []
    bws = []
    for c in range(NUM_CLASSES):
        patch_ss = patches[np.where(labels == c)]
        patch_ss = patch_ss.reshape(patch_ss.shape[0]*patch_ss.shape[1], *patch_ss.shape[2:])
        bw = estimate_bandwidth(patch_ss)
        bws.append(bw)
        idxs = np.random.choice(patch_ss.shape[0], sub_samples, replace=False)
        patch_ss = patch_ss.reshape(patch_ss.shape[0], -1)
        means = patch_ss.mean(axis=0)[:,np.newaxis]
        covMatrix = 1.0/(patch_ss.shape[0]) \
                    * (patch_ss.T.dot(patch_ss) - means.dot(means.T))
        #covMatrix =  1.0  * np.eye(patch_ss.shape[1]) * 10.0/bw
        covMatrixRoot = np.linalg.cholesky(covMatrix).astype('float32')
        covMatrixRoots.append(covMatrixRoot)

    def gaussian_filter_gen(num_filters):
        ps = patch_size
        filters = []
        for c in range(NUM_CLASSES):
            out = np.random.randn(
                num_filters/NUM_CLASSES, channels*ps*ps).astype(
                    'float32').dot(covMatrixRoots[c])
            filters.append(out.reshape(out.shape[0], channels, ps, ps))
        return np.concatenate(filters, axis=0)
    return gaussian_filter_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train simple deep nets with robust optimization objective.')
    parser.add_argument('--dataset_dir', default="",
                        type=str, help="Path to folder in which test and train protos can be found")

    args = parser.parse_args()
    # Load CIFAR

    NUM_FEATURE_BATCHES=512
    DATA_BATCH_SIZE=(1280)
    FEATURE_BATCH_SIZE=(1024)
    NUM_TRAIN = 50000
    NUM_TEST = 10000
    NUM_CLASSES = 10
    POOL_TYPE ='avg'
    FILTER_GEN ='empirical'
    BANDWIDTH = 1.0
    LAMBDAS = [1e-1/FEATURE_BATCH_SIZE, 1e-2/FEATURE_BATCH_SIZE, 1e-3/FEATURE_BATCH_SIZE, 1e-4/FEATURE_BATCH_SIZE, 1e-5/FEATURE_BATCH_SIZE]
    CUDA_CONVNET = False
    SCALE = 55.0
    BIAS = 1.25
    MIN_VAR_TOL = 1e-4
    TOT_FEAT = FEATURE_BATCH_SIZE*NUM_FEATURE_BATCHES

    np.random.seed(10)
    (XTrain, labelsTrain), (XTest, labelsTest), _ \
        = load_dataset(args.dataset_dir)
    patches = patchify_all_imgs(XTrain, (6,6), pad=False)
    if FILTER_GEN == 'gaussian':
        filter_gen = make_gaussian_filter_gen(1.0)
    elif FILTER_GEN == 'empirical':
        filter_gen = make_empirical_filter_gen(patches, labelsTrain)
    elif FILTER_GEN == 'empirical_balanced':
        filter_gen = make_balanced_empirical_filter_gen(patches, labelsTrain)
    elif FILTER_GEN == 'gaussian_cov':
        filter_gen = make_gaussian_cov_filter_gen(patches)
    elif FILTER_GEN == 'gaussian_cc_cov':
        filter_gen = make_gaussian_cc_cov_filter_gen(patches, labelsTrain)
    else:
        raise Exception('Unknown FILTER_GEN value')


    '''
    X = np.vstack((XTrain, XTest))
    time1 = time.time()
    (Xlevel1, filters) = conv(X, filter_gen, FEATURE_BATCH_SIZE, 1, DATA_BATCH_SIZE, CUDA_CONVNET, pool_size=2, symmetric_relu=False)
    time2 = time.time()
    print 'Convolutions with {0} filters took {1} seconds'.format(NUM_FEATURE_BATCHES*FEATURE_BATCH_SIZE, (time2-time1))
    Xlevel1Train = Xlevel1[:50000,:,:,:]
    Xlevel1Test = Xlevel1[50000:,:,:,:]
    patches = patchify_all_imgs(Xlevel1Train, (3,3), pad=False)
    patches = patches.reshape(patches.shape[0]*patches.shape[1],*patches.shape[2:])
    bw = estimate_bandwidth(patches)
    filter_gen = make_gaussian_filter_gen(10.0/bw, patch_size=3, channels=128)
    print ('level 1 shape ' + str(Xlevel1.shape))
    (XFinal, filters) = conv(Xlevel1, filter_gen, FEATURE_BATCH_SIZE*10, NUM_FEATURE_BATCHES, DATA_BATCH_SIZE, CUDA_CONVNET, pool_size=6, bias=1.0)
    print('level 2 shape ' + str(XFinal.shape))

    XFinalTrain = XFinal[:50000,:,:,:].reshape(NUM_TRAIN,-1)
    XFinalTest = XFinal[50000:,:,:,:].reshape(NUM_TEST,-1)
    print "Output train data shape ", XFinalTrain.shape
    print "Output test data shape ", XFinalTest.shape
    print "Output filters shape ", filters.shape
    convTrainAcc, convTestAcc = trainAndEvaluatePrimalModel(XFinalTrain, XFinalTest, labelsTrain, labelsTest, reg=LAMBDAS[0])
    print "(conv) train: ", convTrainAcc, "(conv) test: ", convTestAcc
    print("STARTING LEVEL 2")
    '''
    featurizeTrainAndEvaluateDualModelAsync(XTrain, XTest, labelsTrain, labelsTest, filter_gen, num_feature_batches=NUM_FEATURE_BATCHES, solve_every_iter=NUM_FEATURE_BATCHES/4, regs=LAMBDAS)
