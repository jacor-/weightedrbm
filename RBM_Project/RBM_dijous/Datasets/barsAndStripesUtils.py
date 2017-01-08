
import numpy
from itertools import combinations
from Utils.CommonUtils import fromShapeToNumber

def fromNumberToShape(number, VISIBLES):
    v = numpy.zeros(VISIBLES)
    i = 0
    while number > 0:
        v[-1-i] = numpy.mod(number, 2)
        number /= 2
        i += 1
    return v

def generatePossibleRows(shape_dim):
    result = []
    for i in range(shape_dim+1):
        for comb in combinations(''.join(map(str,range(shape_dim))),i):
            result.append(numpy.zeros((shape_dim, shape_dim)))
            for ind in comb:
                result[-1][int(ind),:] = 1.
    return map(int,map(fromShapeToNumber,result))

def generatePossibleColumns(shape_dim):
    result = []
    for i in range(shape_dim+1):
        for comb in combinations(''.join(map(str,range(shape_dim))),i):
            result.append(numpy.zeros((shape_dim, shape_dim)))
            for ind in comb:
                result[-1][:,int(ind)] = 1.
    return map(int,map(fromShapeToNumber,result))

def generatePossibleCorrectStates(desired_visible_states):
    return numpy.array(list(set(generatePossibleRows(int(numpy.sqrt(desired_visible_states)))+generatePossibleColumns(int(numpy.sqrt(desired_visible_states))))))
