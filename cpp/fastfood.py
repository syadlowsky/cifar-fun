# coding: utf-8
import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

ff = ctypes.cdll.LoadLibrary('./fastfood.so').fastfood
ff.argtypes = [ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]

def _is_power_2(num):
    return (num != 0 and ((num & (num - 1)) == 0))

def fastfood(gaussian, samples, outsize, seed=0, scale=1):
    ''' gaussian Input should be IID  N(0,1) gaussian
        variance is captured in scale (which is (1/sigma*sqrt(d)))
    '''

    assert(_is_power_2(outsize))
    gaussian = gaussian.astype('float32')
    np.random.seed(seed)
    radamacher = np.random.binomial(1, 0.5, outsize).astype('float32')
    radamacher[np.where(radamacher== 0)] = -1
    chisquared = np.sqrt(np.random.chisquare(outsize, outsize)).astype('float32') * 1.0/np.linalg.norm(gaussian).astype('float32')
    out = np.zeros((samples.shape[0], outsize), 'float32')
    ff(gaussian, radamacher, chisquared, samples, out, outsize, samples.shape[1], samples.shape[0])
    out /= scale
    return out


def __fastfood(gaussian, radamacher, chisquared, samples, output, outsize, insize, numsamples):
    ''' LOW LEVEL WRAPPER
    This function will mutate output '''
    ff(gaussian, radamacher, chisquared, samples, output, outsize, insize, numsamples)

