import numpy
from itertools import combinations
from pylab import *
from Utils.CommonUtils import fromShapeToNumber

def generatePossibleCorrectStates(desired_visible_states):
    shape_dim = (desired_visible_states - 3) / 2 
    result = []
    for i in range(shape_dim):
        for comb in combinations(''.join(map(str,range(shape_dim))),i):
            aux = numpy.zeros(shape_dim*2+3)
            for ind in comb:
                aux[int(ind)] = 1.
            aux0 = numpy.array(aux)
            aux0[shape_dim] = 1.
            aux0[shape_dim+3] = aux[shape_dim-1]
            aux0[shape_dim+4:] = aux[:shape_dim-1]

            aux2 = numpy.array(aux)
            aux2[shape_dim+2] = 1.
            aux2[-1] = aux[0]
            aux2[shape_dim+3:-1] = aux[1:shape_dim]
            
            aux1 = numpy.array(aux)
            aux1[shape_dim+1] = 1.
            aux1[shape_dim+3:] = aux[:shape_dim]
            
            result.append(aux0)
            result.append(aux1)
            result.append(aux2)                
    #return result
    return numpy.array(map(int,map(fromShapeToNumber,result)))
