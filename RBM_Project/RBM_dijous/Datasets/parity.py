import numpy
from itertools import combinations
from pylab import *
from Utils.CommonUtils import fromShapeToNumber

def generatePossibleCorrectStates(desired_visible_states):
    shape_dim = desired_visible_states
    result = []
    for i in range(0,shape_dim+1,2):
        for comb in combinations(''.join(map(str,range(shape_dim))),i):
            result.append(numpy.zeros(shape_dim))
            for ind in comb:
                result[-1][int(ind)] = 1.
    #return result
    return numpy.array(map(int,map(fromShapeToNumber,result)))

