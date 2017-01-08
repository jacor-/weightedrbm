import theano

'''
from sklearn.cross_validation import KFold
from Datasets import barsAndStripesUtils, bitSwap, parity
import numpy

def getDataset(dataset_keyword, DESIRED_VISIBLES, kfolds = 10):
    if dataset_keyword == "BarsAndStripes":
        valid_states_indexs = numpy.array(barsAndStripesUtils.generatePossibleCorrectStates(DESIRED_VISIBLES))
    elif dataset_keyword == "Parity":
        valid_states_indexs = numpy.array(parity.generatePossibleCorrectStates(DESIRED_VISIBLES))
    elif dataset_keyword == "BitSwap":
        valid_states_indexs = numpy.array(bitSwap.generatePossibleCorrectStates(DESIRED_VISIBLES))
    else:
        raise "UNKNOWN DATASET"
        
    index_quantity = numpy.arange(len(valid_states_indexs))
    
    kfold = KFold(n=index_quantity.shape[0], n_folds=kfolds)
    
    for train_indexs, test_indexs in kfold:
        data_train_indexs = valid_states_indexs[train_indexs]
        data_test_indexs = valid_states_indexs[test_indexs]
        yield (data_train_indexs, data_test_indexs)
'''



#from sklearn.cross_validation import KFold
from Datasets import barsAndStripesUtils, bitSwap, parity
import numpy

def getDataset(dataset_keyword, DESIRED_VISIBLES, kfolds = 10):
    if dataset_keyword == "BarsAndStripes":
        valid_states_indexs = numpy.array(barsAndStripesUtils.generatePossibleCorrectStates(DESIRED_VISIBLES))
    elif dataset_keyword == "Parity":
        valid_states_indexs = numpy.array(parity.generatePossibleCorrectStates(DESIRED_VISIBLES))
    elif dataset_keyword == "BitSwap":
        valid_states_indexs = numpy.array(bitSwap.generatePossibleCorrectStates(DESIRED_VISIBLES))
    else:
        raise "UNKNOWN DATASET"

    if kfolds != 0:
        index_quantity = numpy.arange(len(valid_states_indexs))
    
        #kfold = KFold(n=index_quantity.shape[0], n_folds=kfolds)
        #
        #for train_indexs, test_indexs in kfold:
        #    data_train_indexs = valid_states_indexs[train_indexs]
        #    data_test_indexs = valid_states_indexs[test_indexs]
        #    yield (data_train_indexs, data_test_indexs)
    
        for _ in range(kfolds):
            numpy.random.shuffle(index_quantity)
            data_train_indexs = numpy.asarray(valid_states_indexs[index_quantity[:len(index_quantity)/kfolds]], dtype=theano.config.floatX)  # @UndefinedVariable
            data_test_indexs = numpy.asarray(valid_states_indexs[index_quantity[len(index_quantity)/kfolds:]], dtype=theano.config.floatX)  # @UndefinedVariable
            yield (data_train_indexs, data_test_indexs)
    else:
        yield valid_states_indexs
    
    
    
