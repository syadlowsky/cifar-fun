from theano import function, config, shared, sandbox
import theano.tensor as T
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import logging
import numpy as np
import time

#logging.getLogger('theano.gof.cmodule').setLevel(logging.DEBUG)

def unpickle(infile):
    import cPickle
    fo = open(infile, 'rb')
    outdict = cPickle.load(fo)
    fo.close()
    return outdict

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


def conv(data, feature_batch_size, num_feature_batches, data_batch_size, cuda_convnet=True):
    outX = []
    filters = []
    numImages = data.shape[0]

    # Convert to cuda-convnet order
    if (cuda_convnet):
        data = data.transpose(1,2,3,0)

    XTheano = shared(data.astype('float32'), borrow=True)
    for j in range(num_feature_batches):
        F = np.random.randn(feature_batch_size, 3, 6, 6).astype('float32')
        if (cuda_convnet):
            F = F.transpose(1,2,3,0)

        filters.append(F)
        FTheano = shared(F.astype('float32'), borrow=True)
        out = []
        for i in range(int(np.ceil(numImages/float(data_batch_size)))):
                start = i*data_batch_size
                end = min((i+1)*data_batch_size, numImages)

                print "FEATURE BATCH #", j, "DATA BATCH #", i,  " SIZE IS ", end - start
                if (cuda_convnet):
                    XBlock = XTheano[:, :, :, start:end]
                else:
                    XBlock = XTheano[start:end, :, :, :]

                if (cuda_convnet):
                    conv_op = FilterActs()
                    # Turn into continigious chunk for theano
                    XBlock = gpu_contiguous(XBlock)
                    FTheano = gpu_contiguous(FTheano)
                else:
                    conv_op = lambda X, F: T.nnet.conv2d(X, F)

                # CONV
                XBlock = conv_op(XBlock, FTheano)

                # RELU
                XBlock0 = T.nnet.relu(XBlock, 0)
                XBlock1 = T.nnet.relu(-1.0 * XBlock, 0)

                # MAX POOL
                if (cuda_convnet):
                    pool_op = MaxPool(ds=14, stride=14)
                else:
                    pool_op = lambda X: T.signal.pool.pool_2d(X, (14, 14), ignore_border=False, mode='max')

                XBlock0 = pool_op(XBlock0)
                XBlock1 = pool_op(XBlock1)

                XBlockOut = np.concatenate((XBlock0.eval(), XBlock1.eval()), axis=1)
                if (cuda_convnet):
                    XBlockOut = XBlockOut.transpose(3,0,1,2)
                    F = F.transpose(3,0,1,2)

                out.append(XBlockOut)
        outX.append(np.concatenate(out, axis=0))

    XFinal = np.concatenate(outX, axis=1)
    filters = np.concatenate(filters,axis=0)

    return (XFinal, filters)

if __name__ == "__main__":
    # Load CIFAR

    NUM_FEATURE_BATCHES=1
    DATA_BATCH_SIZE=(1280)
    FEATURE_BATCH_SIZE=(1024)
    CUDA_CONVNET = True

    (XTrain, yTrain), (XTest, yTest) = load_cifar()
    X = np.vstack((XTrain, XTest))



    (XFinal, filters) = conv(X, FEATURE_BATCH_SIZE, NUM_FEATURE_BATCHES, DATA_BATCH_SIZE, CUDA_CONVNET)
    print "Output data shape ", XFinal.shape
    print "Output filters shape ", filters.shape

