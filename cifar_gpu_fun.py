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
        cifar_out = unpickle("./cifar/data_batch_{0}".format(i))
        train_batches.append(cifar_out["data"])
        train_labels.extend(cifar_out["labels"])

    # Stupid bull shit to get pixels in correct order
    X_train= np.vstack(tuple(train_batches)).reshape(-1, 32*32, 3)
    X_train = X_train.reshape(-1,3,32,32)
    mean_image = np.mean(X_train, axis=0)[np.newaxis, :, :]
    y_train = np.array(train_labels)
    cifar_out = unpickle("./cifar/test_batch")
    X_test = cifar_out["data"].reshape(-1, 32*32, 3)
    X_test = X_test.reshape(-1,3,32,32)
    X_train = X_train
    X_test = X_test
    y_test = cifar_out["labels"]
    return (X_train, np.array(y_train)), (X_test, np.array(y_test))


def conv(data, feature_batch_size, num_feature_batches, data_batch_size):
    XTheano = shared(X.astype('float32'), borrow=True)
    outX = []
    filters = []
    for j in range(num_feature_batches):
        F = np.random.randn(3, 6, 6, feature_batch_size).astype('float32')
        filters.append(F)
        FTheano = shared(F.astype('float32'), borrow=True)
        out = []
        for i in range(int(np.ceil(X.shape[-1]/float(data_batch_size)))):
                start = i*data_batch_size
                end = min((i+1)*data_batch_size, X.shape[-1])

                print "FEATURE BATCH #", j, "DATA BATCH #", i,  " SIZE IS ", end - start
                XBlock = XTheano[:, :, :, start:end]
                conv_op = FilterActs()

                # Turn into continigious chunk for theano
                XBlock = gpu_contiguous(XBlock)
                FTheano = gpu_contiguous(FTheano)

                # CONV
                XBlock = conv_op(XBlock, FTheano)

                # RELU
                XBlock1 = T.nnet.relu(XBlock, 0)
                XBlock1 = T.nnet.relu(-1.0 * XBlock, 0)


                XBlock = XBlock1

                # MAX POOL
                pool_op = MaxPool(ds=13, stride=13)

                XBlock = pool_op(XBlock)
                XBlock1 = pool_op(XBlock1)

                # evaluation
                XBlockOut = np.concatenate((XBlock.eval(), XBlock1.eval()), axis=0)
                out.append(XBlockOut)
        outX.append(np.concatenate(out, axis=3))
    XFinal = np.concatenate(outX, axis=0)
    return (XFinal, filters)

if __name__ == "__main__":
    # Load CIFAR
    (XTrain, yTrain), (XTest, yTest) = load_cifar()
    X = np.vstack((XTrain, XTest))
    
    # Convert to cuda-convnet order
    X = X.transpose(1,2,3,0)

    NUM_FEATURE_BATCHES=1
    DATA_BATCH_SIZE=1280
    FEATURE_BATCH_SIZE=1024

    conv(X, FEATURE_BATCH_SIZE, NUM_FEATURE_BATCHES, DATA_BATCH_SIZE)

