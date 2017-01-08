import time
import numpy
from Datasets import getDataset
from Machines.RBM import RBM
from pylab import *
from Datasets import AleixDatasetGeneration as dg
from bzrlib.config import StartingPathMatcher

NUM_EPOCHS = 6000
HIDDEN = 16

learning_rate = 0.01
momentum_rate = 0.9
sigma = 1.

'''
VISIBLES = 9
PROBLEM = "BarsAndStripes"
dataset, VISIBLES = dg.load_data("bars_stripes", {'num_rows':int(numpy.sqrt(VISIBLES)), 'num_cols':int(numpy.sqrt(VISIBLES))}, 1., 0)
'''

VISIBLES = 9
dataset, VISIBLES = dg.load_data("parity", {'num_vis':int(VISIBLES), 'w_size':6}, 1., 0)
print VISIBLES
print HIDDEN

def execute(dataset, probabilities, machine_type, k, learning_rate, momentum_rate, sigma):
    t1 = time.time()
    machine = RBM(HIDDEN,VISIBLES, dataset, probabilities, learning_rate, momentum_rate, sigma, k = k, type = machine_type, sample_last = True, computeProbs = True)
    learning_algorithm = machine.trainStep    
    iters = 0
    monit_likelihood = []
    while iters < NUM_EPOCHS:
        iters += 1
        learning_algorithm()
        if iters % 100 == 0:
            print iters
            monit_likelihood.append(machine.compute_KL())
    print "TEST_FINISHED in %.1f seconds" % (time.time()-t1)
    return monit_likelihood

figure()
for i in range(1):
    likelihood_proposal_1 = execute(dataset[0], dataset[1], machine_type = "equal_weights", k = 1, learning_rate=learning_rate, momentum_rate = momentum_rate, sigma = sigma);   plot(likelihood_proposal_1, 'r', label = "CD-1 Equal weights")
#likelihood_proposal_3 = execute(dataset[0], dataset[1], machine_type = "exact", k = 1, learning_rate=learning_rate, momentum_rate = momentum_rate, sigma = sigma);           plot(likelihood_proposal_3, label = "Exact")
for i in range(1):
    likelihood_proposal_4 = execute(dataset[0], dataset[1], machine_type = "model_weights", k = 1, learning_rate=learning_rate, momentum_rate = momentum_rate, sigma = sigma);   plot(likelihood_proposal_4, 'g', label = "CD-1 Model weights")
legend()
show()

