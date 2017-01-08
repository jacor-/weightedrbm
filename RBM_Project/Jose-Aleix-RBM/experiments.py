from rbm import RBM
import JoseRBM

def execAleixRBM(hyperparameters, train_type):
    rbm = RBM(num_vis, NumHid, NeuronType)
    rbm.rand_w(Sigma)
    rbm.train(train_type, dataset, hyperparameters, nPrint, File_w)

def execJoseEBM(hyperparameters, train_type):
    TrainRout, Persistent, Weighted = train_type
    LRate, Momentum, WeightDecay, MinEpochs, MaxEpochs, MiniBatchSize, CDk = hyperparameters
    
    if TrainRout == "GradDesc":
        machine_type = "exact"
    elif TrainRout == "ContDiv":
        if Weighted:
            machine_type = "model_weights"
        else:
            machine_type = "equal_weights"
    
    print machine_type
    jose_rbm = JoseRBM.RBM(num_vis, NumHid, dataset, neuronType = NeuronType, sigma = Sigma, k = CDk, type = machine_type, require_probs = True)
    aux = jose_rbm.train(hyperparameters, nPrint, File_w)


from settings import paramsDataSet, DatasetName, NeuronType, TrainPercent, hyperparameters, train_type, nPrint, File_w, NumHid, Sigma
import dataset_generation as dg 



# --------- Obtain data and select train/test -------- #
dataset, num_vis = dg.load_data(DatasetName, paramsDataSet, TrainPercent, NeuronType)





#def execAleixRBM(hyperparameters, train_type):
#    rbm = RBM(num_vis, NumHid, NeuronType)
#    rbm.rand_w(Sigma)
#    rbm.train(train_type, dataset, hyperparameters, nPrint, File_w)

#execJoseEBM(hyperparameters, train_type)
execAleixRBM(hyperparameters, train_type)