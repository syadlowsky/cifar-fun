from theano import function, config, shared, sandbox
import theano.tensor as T
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from multiprocessing import Pipe
import multiprocessing as mp
from six.moves import cPickle
import logging
import numpy as np
import scipy.linalg
from sklearn import metrics
import time
from multiprocessing import Process, Queue
import numpy as np
import SharedArray as sa
from sklearn.metrics import accuracy_score

#logging.getLogger('theano.gof.cmodule').setLevel(logging.DEBUG)

def unpickle(infile):
    import cPickle
    fo = open(infile, 'rb')
    outdict = cPickle.load(fo)
    fo.close()
    return outdict

def load_cifar_processed():
    npzfile = np.load("./cifar_processed")
    return (npzfile['XTrain'], npzfile['yTrain']), (npzfile['XTest'], npzfile['yTest'])

def load_cifar(center=False):
    train_batches = []
    train_labels = []
    for i in range(1,6):
        cifar_out = unpickle("../cifar/data_batch_{0}".format(i))
        train_batches.append(cifar_out["data"])
        train_labels.extend(cifar_out["labels"])

    # Stupid bull shit to get pixels in correct order
    X_train= np.vstack(tuple(train_batches)).reshape(-1, 32*32, 3)
    X_train = X_train.reshape(-1,3,32,32)
    mean_image = np.mean(X_train, axis=0)[np.newaxis, :, :]
    y_train = np.array(train_labels)
    cifar_out = unpickle("../cifar/test_batch")
    X_test = cifar_out["data"].reshape(-1, 32*32, 3)
    X_test = X_test.reshape(-1,3,32,32)
    X_train = X_train
    X_test = X_test
    y_test = cifar_out["labels"]
    return (X_train, np.array(y_train)), (X_test, np.array(y_test))


def conv(data, feature_batch_size, num_feature_batches, data_batch_size, cuda_convnet=True, bandwidth=0.01, symmetric_relu=True, start_feature_batch=0):
    outX = []
    filters = []
    numImages = data.shape[0]
    data = data.astype('float32')

    # Convert to cuda-convnet order
    if (cuda_convnet):
        data = data.transpose(1,2,3,0)

    #XTheano = shared(data.astype('float32'))
    for j in range(num_feature_batches):
        F = np.random.randn(feature_batch_size, 3, 6, 6).astype('float32') * bandwidth
        if (cuda_convnet):
            F = F.transpose(1,2,3,0)

        filters.append(F)
        FTheano = shared(F.astype('float32'))
        out = []
        for i in range(int(np.ceil(numImages/float(data_batch_size)))):
                start = i*data_batch_size
                end = min((i+1)*data_batch_size, numImages)
                print "FEATURE BATCH #", (j + start_feature_batch), "DATA BATCH #", i,  " SIZE IS ", end - start
                if (cuda_convnet):
                    XBlock = shared(data[:, :, :, start:end])
                else:
                    XBlock = shared(data[start:end, :, :, :])

                if (cuda_convnet):
                    conv_op = FilterActs()
                    # Turn into continigious chunk for theano
                    XBlock_gpu = gpu_contiguous(XBlock)
                    FTheano_gpu = gpu_contiguous(FTheano)
                else:
                    conv_op = lambda X, F: T.nnet.conv2d(X, F)

                # CONV
                XBlock_conv_out = conv_op(XBlock_gpu, FTheano_gpu)

                # RELU
                XBlock0 = T.nnet.relu(XBlock_conv_out, 0)
                if (symmetric_relu):
                    XBlock1 = T.nnet.relu(-1.0 * XBlock_conv_out, 0)

                # MAX POOL
                if (cuda_convnet):
                    pool_op = MaxPool(ds=14, stride=14)
                else:
                    pool_op = lambda X: T.signal.pool.pool_2d(X, (14, 14), ignore_border=False, mode='max')

                XBlock0 = pool_op(XBlock0)
                if (symmetric_relu):
                    XBlock1 = pool_op(XBlock1)
                    XBlockOut = np.concatenate((XBlock0.eval(), XBlock1.eval()), axis=1)
                else:
                    XBlockOut = np.array(XBlock0.eval())

                if (cuda_convnet):
                    XBlockOut = XBlockOut.transpose(3,0,1,2)
                    F = F.transpose(3,0,1,2)

                XBlock.set_value([[[[]]]])
                out.append(XBlockOut)

        FTheano.set_value([[[[]]]])
        outX.append(np.concatenate(out, axis=0))

    XFinal = np.concatenate(outX, axis=1)
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
    train_norms = np.linalg.norm(train, axis=1)
    test_norms = np.linalg.norm(test, axis=1)

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

    XTX += reg * np.eye(XTX.shape[0])


    y = np.eye(max(labels) + 1)[labels]
    XTy = X.T.dot(y)

    print "Learning Primal Model"
    model = scipy.linalg.solve(XTX, XTy)
    return model

def learnDual(gramMatrix, labels, reg=0.1):
    ''' Learn a model from K matrix -> labels '''
    print ("Learning Dual Model")
    y = np.eye(max(labels) + 1)[labels]
    model = scipy.linalg.solve(gramMatrix + reg * np.eye(gramMatrix.shape[0]), y)
    print model.shape
    return model

def evaluatePrimalModel(data, model):
    data = data.reshape(data.shape[0],-1)
    yHat = np.argmax(data.dot(model), axis=1)
    return yHat


def evaluateDualModel(kMatrix, model):
    print("MODEL SHAPE " + str(model.shape))
    print("KERNEL SHAPE " + str(kMatrix.shape))
    y = kMatrix.dot(model)
    print("pred SHAPE " + str(y.shape))
    yHat = np.argmax(y, axis=1)
    return yHat


def trainAndEvaluatePrimalModel(XTrain, XTest, labelsTrain, labelsTest):
    model = learnPrimal(XTrain, labelsTrain)
    predTrainLabels = evaluatePrimalModel(XTrain, model)
    predTestLabels = evaluatePrimalModel(XTest, model)
    train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
    test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
    return train_acc, test_acc

def featurizeTrainAndEvaluateDualModel(XTrain, XTest, labelsTrain, labelsTest, num_feature_batches=1, solve_every_iter=1, reg=0.1):
    trainKernel = np.zeros((XTrain.shape[0], XTrain.shape[0]),dtype='float32')
    testKernel= np.zeros((XTest.shape[0], XTrain.shape[0]),dtype='float32')
    for i in range(1, (num_feature_batches + 1)):
        X = np.vstack((XTrain, XTest))
        print("Convolving features")
        time1 = time.time()
        (XBatch, filters) = conv(X, FEATURE_BATCH_SIZE, 1, DATA_BATCH_SIZE, CUDA_CONVNET, symmetric_relu=True, start_feature_batch=i-1)
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

def featurizeTrainAndEvaluateDualModelAsync(XTrain, XTest, labelsTrain, labelsTest, num_feature_batches=1, solve_every_iter=1, reg=0.1):
    X = np.vstack((XTrain, XTest))
    parent, child = Pipe()
    sa.delete("shm://xbatch")
    XBatchShared = sa.create("shm://xbatch", (X.shape[0],FEATURE_BATCH_SIZE*8), dtype='float32')
    p = Process(target=accumulateGramAndSolveAsync, args=(child,XTrain.shape[0], XTest.shape[0], reg))
    p.start()
    child.close()

    for i in range(1, (num_feature_batches + 1)):
        print("Convolving features")
        time1 = time.time()
        (XBatch, filters) = conv(X, FEATURE_BATCH_SIZE, 1, DATA_BATCH_SIZE, CUDA_CONVNET, symmetric_relu=True, start_feature_batch=i-1)
        time2 = time.time()
        print 'Convolving features took {0} seconds'.format((time2-time1))
        print("Sending features")
        time1 = time.time()
        np.copyto(XBatchShared, XBatch.reshape(XBatch.shape[0], -1))
        parent.send(0)
        print 'Sending features took {0} seconds'.format((time2-time1))
        time2 = time.time()
    print("CLOSING CONNECTION")
    parent.send(-1)
    parent.close()

def accumulateGramAndSolveAsync(pipe, numTrain, numTest, reg):
    trainKernel = np.zeros((numTrain, numTrain), dtype='float32')
    testKernel= np.zeros((numTest, numTrain), dtype='float32')
    XBatchShared = sa.attach("shm://xbatch")
    # Local copy
    XBatchLocal = np.zeros(XBatchShared.shape, dtype='float32')
    print("CHILD Process Spun")
    while(True):
        m = pipe.recv()
        if m == -1:
            break
        time1 = time.time()
        print("Receiving (ASYNC) Gram")
        np.copyto(XBatchLocal, XBatchShared)
        time2 = time.time()
        print 'Receiving (ASYNC) took {0} seconds'.format((time2-time1))
        XBatchTrain = XBatchLocal[:50000,:]
        XBatchTest = XBatchLocal[50000:,:]
        print("XBATCH DTYPE " + str(XBatchTest.dtype))
        print("Accumulating (ASYNC) Gram")
        time1 = time.time()
        trainKernel += XBatchTrain.dot(XBatchTrain.T)
        testKernel += XBatchTest.dot(XBatchTrain.T)
        time2 = time.time()
        print 'Accumulating (ASYNC) gram took {0} seconds'.format((time2-time1))
    time1 = time.time()
    model = learnDual(trainKernel, labelsTrain, reg)
    time2 = time.time()
    print 'learningDual (ASYNC) took {0} seconds'.format((time2-time1))
    predTrainLabels = evaluateDualModel(trainKernel, model)
    predTestLabels = evaluateDualModel(testKernel, model)
    print("true shape " + str(labelsTrain.shape))
    print("pred shape " + str(predTrainLabels.shape))
    train_acc = metrics.accuracy_score(labelsTrain, predTrainLabels)
    test_acc = metrics.accuracy_score(labelsTest, predTestLabels)
    print "(async dual conv) train: , {convTrainAcc}, (dual conv batch) test: {convTestAcc}".format(convTrainAcc=train_acc, convTestAcc=test_acc)
    return train_acc, test_acc

if __name__ == "__main__":
    # Load CIFAR

    NUM_FEATURE_BATCHES=1
    DATA_BATCH_SIZE=(1280+256)
    FEATURE_BATCH_SIZE=(1024)
    NUM_TRAIN = 50000
    NUM_TEST = 10000
    CUDA_CONVNET = True

    np.random.seed(0)
    (XTrain, labelsTrain), (XTest, labelsTest) = load_cifar_processed()

    '''
    X = np.vstack((XTrain, XTest))
    time1 = time.time()
    (XFinal, filters) = conv(X, FEATURE_BATCH_SIZE, NUM_FEATURE_BATCHES, DATA_BATCH_SIZE, CUDA_CONVNET, symmetric_relu=True)
    time2 = time.time()
    print 'Convolutions with {0} filters took {1} seconds'.format(NUM_FEATURE_BATCHES*FEATURE_BATCH_SIZE, (time2-time1))
    XFinalTrain = XFinal[:50000,:,:,:].reshape(NUM_TRAIN,-1)
    XFinalTest = XFinal[50000:,:,:,:].reshape(NUM_TEST,-1)
    print "Output train data shape ", XFinalTrain.shape
    print "Output test data shape ", XFinalTest.shape
    print "Output filters shape ", filters.shape
    convTrainAcc, convTestAcc = trainAndEvaluatePrimalModel(XFinalTrain, XFinalTest, labelsTrain, labelsTest)
    print "(conv) train: ", convTrainAcc, "(conv) test: ", convTestAcc
    '''

    featurizeTrainAndEvaluateDualModelAsync(XTrain, XTest, labelsTrain, labelsTest, num_feature_batches=NUM_FEATURE_BATCHES, solve_every_iter=NUM_FEATURE_BATCHES)

    '''


    '''

