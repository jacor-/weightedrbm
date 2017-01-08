
import numpy
from pylab import figure, plot, show

def fromShapeToNumber(shape):
    number = 0
    for i in range(len(shape)):
        if type(shape[i]) == list or type(shape[i]) == numpy.ndarray :
            for j in range(len(shape[i])):
                number += shape[i][j]*2**(i+j*len(shape[0]))
        else:
            number += shape[i]*2**i
    return number

def plotaProbsAndData(probs, valid_indexs_train, valid_indexs_test):
    figure()
    plot(probs, 'kx')
    plot(valid_indexs_train, probs[valid_indexs_train], 'ob')
    plot(valid_indexs_test, probs[valid_indexs_test], 'or')

