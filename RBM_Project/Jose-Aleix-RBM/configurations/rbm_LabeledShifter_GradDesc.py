import time

import numpy as np

import dataset_generation as dg

from rbm import RBM

########################################################
################# Start Parameters #####################
########################################################

# -------- Problem independent parameters -------- #
TrainRout = "GradDesc"  # Can be "GradDesc" for gradient descent or "ContDiv"
                        #   for contrastive divergence
Persistent = None       # True for persistent CD
Weighted   = None       # True for Weighted CD

NumHid = 16             # Number of hidden neurons
NeuronType = 1          # It is 0 or 1 for {0, 1} or {-1, 1} neurons.
CDk = None              # Iterations of contrastive divergence

Sigma = 1.0             # Variance of initial weights
LRate = 0.01            # Learning rate: it usually changes from 0.1 to 0.001
Momentum = 0.9          # Momentum of the training usually 0.9
WeightDecay = 0         # L2 regularization
MinEpochs = 200         # Minimum number of epochs
MaxEpochs = 5000        # Maximum number of epochs 
nPrint = 50             # Print probabilities and KL every nPrint iterations
MiniBatchSize = 32      # If it is None, the gradient is computed for all 
                        #   elements in the training set. Otherwise, it is the
                        #   size of the mini-batches.
TrainPercent = 1.00     # Percentage of training examples (in [0,1])

File_w = None           # File to save the weights after the last iteration

# --------- Parity problem parameters ---------------- #
#NumVis = 8              # Number of visible units
#WSize = 5               # The HOBM has the form:
#                        #   exp[WeightSize * S1 * S2 * ... * Sm]
#HOBMwOrder = 8          # Order of the HOBM used to generate the training set
#AllCombInOrder = False  # If it is False, then we generate the training set 
#                        #   with just one high-order weight. Otherwise, we use
#                        #   all the high order weights. IT SHOULD BE FALSE!
#DatasetName = "parity"

# -------------- Bars and stripes parameters --------- #
#NumRows = 3
#NumCols = 3
#DatasetName = "bars_stripes"

# -------------- Labeled shifter parameters ---------- #
NumElems = 4
DatasetName = "labeled_shifter"


########################################################
################## End Parameters ######################
########################################################

