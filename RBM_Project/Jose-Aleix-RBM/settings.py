from configurations.rbm_BarsStripes_ContDiv_k01 import *

# ---------------- Select dataset -------------------- #
if DatasetName == "parity":
    paramsDataSet = {'w_size'  : WSize,  \
                     'num_vis' : NumVis, \
                     'hobm_w_order' : HOBMwOrder, \
                     'all_in_order' : AllCombInOrder}
elif DatasetName == "bars_stripes":
    paramsDataSet = {'num_rows' : NumRows, \
                     'num_cols' : NumCols}
elif DatasetName == "labeled_shifter":
    paramsDataSet = {'num_elems' : NumElems}
else:
    print("Invalid kind of dataset")


# --------------------- Train ------------------------ #
train_type = (TrainRout, Persistent, Weighted)
hyperparameters = (LRate, Momentum, WeightDecay, MinEpochs, MaxEpochs, \
                   MiniBatchSize, CDk)

