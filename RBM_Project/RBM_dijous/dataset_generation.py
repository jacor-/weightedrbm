from __future__ import division, print_function

import itertools
import struct
import array

import numpy as np


def load_data(dataset_name, dataset_params, train_percent, neuron_type):

    # Load entire dataset
    if dataset_name == "parity":

        num_vis = dataset_params['num_vis']
        w_size = dataset_params['w_size']

        if 'hobm_w_order' in dataset_params and \
                dataset_params['hobm_w_order'] is not None:
            hobm_order = dataset_params['hobm_w_order']
        else:
            hobm_order = num_vis

        if 'all_in_order' in dataset_params and \
                dataset_params['all_in_order'] is not None:
            all_in_order = dataset_params['all_in_order']
        else:
            all_in_order = False

        dataset, probs = generate_parity(neuron_type, num_vis, hobm_order, \
                                         w_size, all_in_order)

    elif dataset_name == "bars_stripes":
        num_rows = dataset_params['num_rows']
        num_cols = dataset_params['num_cols']
        num_vis = num_rows * num_cols
        dataset, probs = generate_bars_stripes(neuron_type, num_rows, num_cols)

    elif dataset_name == "labeled_shifter":
        num_elems = dataset_params['num_elems']
        num_vis = 2 * num_elems + 3
        dataset, probs = generate_labeled_shifter(neuron_type, num_elems)

    else:
        print("Invalid name of dataset %s" % dataset_name)

    len_dataset = len(dataset)
    idx = np.arange(len_dataset)
    np.random.shuffle(idx)

    len_train = int(len_dataset * train_percent)

    train_set = dataset[idx[:len_train]]
    train_probs = probs[idx[:len_train]]

    test_set = dataset[idx[len_train:]]
    test_probs = probs[idx[len_train:]]

    return (train_set, train_probs, test_set, test_probs), num_vis


def generate_parity(neuron_type, num_vis, hobm_order, w_size, all_in_order):
    """Generate training sets from high order weights
    
    Parameters
    ----------
        num_vis : int
            number of visible neurons of the training set.
        hobm_order : int
            order of the weights that are used to generate the training set.
        w_size : float
            size of the weights.
        all_in_order : bool
            True : all weights of that order are used to generate the set.
            False : only a single high order weight is used.

    Returns
    -------
        tset : 2d numpy.array
            Array with training elements as rows.
        probs : 1d numpy.array
            Probabilities of the states in tset.
    """
    # Generate all posible states
    tset = np.array(list(itertools.product([-1, 1], repeat=num_vis)))
    probs = np.zeros(len(tset))

    if all_in_order is False:
        # If hobm_order = m, then only consider the high-order weight relating
        # the first m visible neurons. Then, the energy is simply
        #   E = -w_size * S1 * S2 * ... * Sm
        for i in range(len(tset)):
            E = - w_size * np.prod(tset[i, :hobm_order])
            probs[i] = np.exp(-E)
    else:
        # Consider all combinations of weights of order m, all having the 
        # same contribution.
        for i in range(len(tset)):
            E = 0
            for idx in itertools.combinations(np.arange(num_vis), hobm_order):
                E -= w_size * np.prod(tset[i, list(idx)])
            probs[i] = np.exp(-E)

    # Correct it for {0, 1} neurons 
    if neuron_type == 0:
        tset[tset==-1] = 0.0

    # Normalize probability
    probs /= sum(probs)

    return tset, probs


def generate_bars_stripes(neuron_type, num_rows, num_cols):
    dataset = []

    # Add stripes
    for n in range(1, num_rows):
        for idx in itertools.combinations(range(num_rows), n):
            s = np.zeros((num_rows, num_cols))
            s[idx, :] = 1
            dataset.append(s.flatten())

    # Add bars
    for n in range(1, num_cols):
        for idx in itertools.combinations(range(num_cols), n):
            s = np.zeros((num_rows, num_cols))
            s[:, idx] = 1
            dataset.append(s.flatten())

    # Add full and empty cases
    dataset.append(np.zeros(num_rows * num_cols))
    dataset.append(np.ones (num_rows * num_cols))

    # Convert to array
    tset = np.array(dataset, dtype=int)

    # Correct it for {-1, 1} neurons
    if neuron_type == 1:
        tset[tset==0] = -1

    # Equal probabilities for all data
    probs = np.ones(len(tset)) / len(tset)

    return tset, probs


def generate_labeled_shifter(neuron_type, num_elems):

    def left_shift(x):
        n = len(x)
        return x[n-1:] + x[:n-1]

    def right_shift(x):
        n = len(x)
        return x[1:n] + x[:1]

    dataset = []
    for s in itertools.product([0, 1], repeat=num_elems):
        x1 = np.concatenate([s,  left_shift(s), (1, 0, 0)])
        x2 = np.concatenate([s,              s, (0, 1, 0)])
        x3 = np.concatenate([s, right_shift(s), (0, 0, 1)])
        dataset += [x1, x2, x3]

    # Convert to array
    tset = np.array(dataset)

    # Correct it for {-1, 1} neurons
    if neuron_type == 1:
        tset[tset==0] = -1.0

    # Equal probabilities for all data
    probs = np.ones(len(tset)) / len(tset)

    return tset, probs


def load_mnist(path, kind):
    """Based on:
    https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
    """

    if kind == 'train':
        path_img = path + 'train-images.idx3-ubyte'
        path_lbl = path + 'train-labels.idx1-ubyte'
    elif kind == 'test':
        path_img = path + 't10k-images.idx3-ubyte'
        path_lbl = path + 't10k-labels.idx1-ubyte'
    else:
        print("Invalid kind of MNIST dataset : %s" % kind)

    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)
        labels = array.array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)
        image_data = array.array("B", file.read())

    # This is more efficient than np.array(x, dtype=np.int8). See:
    #   http://stackoverflow.com/questions/5674960/efficient-python-array-
    #   to-numpy-array-conversion
    labels = np.frombuffer(labels, dtype=np.int8)
    images = np.frombuffer(image_data, dtype=np.uint8)
    images = images.reshape(size, rows*cols)

    return images, labels


def display(img):
    """Based on:
    https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
    """
    width = 28
    render = ''
    for i in range(len(img)):
        if i % width == 0: 
            render += '\n'
        if img[i] > 100:
            render += '1'
        else:
            render += '0'
    return render


if __name__ == '__main__':

    test_parity = True
    test_bars_stripes = True
    test_labeled_shifter = True
    test_mnist = False

    if test_parity:
        num_vis = 6
        params = {'w_size' : 5.0, 'num_vis' : num_vis}
        train_percent = 0.5
        type_rbm = 0
        dataset = load_data("parity", params, train_percent, type_rbm)
        train_set, train_probs, test_set, test_probs, num_vis = dataset

        print("Train set")
        for i in range(len(train_set)):
            print("%s %f" % (train_set[i], train_probs[i]))

        print("\nTest set")
        for i in range(len(test_set)):
            print("%s %f" % (test_set[i], test_probs[i]))

    if test_bars_stripes:
        params = {'num_rows' : 3, 'num_cols' : 3}
        train_percent = 0.8
        type_rbm = 0
        dataset = load_data("bars_stripes", params, train_percent, type_rbm)
        train_set, train_probs, test_set, test_probs, num_vis = dataset

        print("Train set")
        for i in range(len(train_set)):
            print("%s %f" % (train_set[i], train_probs[i]))

        print("\nTest set")
        for i in range(len(test_set)):
            print("%s %f" % (test_set[i], test_probs[i]))

    if test_labeled_shifter:
        params = {'num_elems' : 4}
        train_percent = 0.5
        type_rbm = 0
        dataset = load_data("labeled_shifter", params, train_percent, type_rbm)
        train_set, train_probs, test_set, test_probs, num_vis = dataset

        print("Train set")
        for i in range(len(train_set)):
            print("%s %f" % (train_set[i], train_probs[i]))

        print("\nTest set")
        for i in range(len(test_set)):
            print("%s %f" % (test_set[i], test_probs[i]))

    if test_mnist:
        img_train, lbl_train = load_mnist('../MNIST/', 'train')
        img_test, lbl_test = load_mnist('../MNIST/', 'test')
        print(lbl_train[10], display(img_train[10, :]), '\n')
        print(lbl_test[10], display(img_test[10, :]))
