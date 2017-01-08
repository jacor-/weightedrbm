from __future__ import division, print_function

import itertools
import time

import numpy as np


class RBM:

    # ----------------------------------------------------------------------- #
    #                           General routines                              #
    # ----------------------------------------------------------------------- #

    def __init__(self, nv, nh, type_rbm):
        """Initialize Restricted Boltzmann Machine

        Parameters
        ----------
            nv : integer
                number of visible units
            nh : integer
                number of hidden units
            type_rbm : integer
                it can take values 0 or 1:
                0 -> neurons take values in { 0, 1}
                1 -> neurons take values in {-1, 1}.

        Example
        -------
        Initialize a RBM with 10 visible units and 4 hidden units using {0, 1}
            rbm = RBM(10, 4, 0)
        """

        # Dimensions of RBM
        self.nv = nv
        self.nh = nh

        # Initialize variables for the momentum in the training
        self.incw = 0
        self.incb = 0
        self.incc = 0

        # Initialize persistent chain to None
        self.ph_old = None

        # Initial reconstruction error to zero
        self.rec_err = 0.0

        # Choose whether routines are for {0, 1} or {-1, 1} neurons
        self.type_rbm = type_rbm
        if type_rbm == 0:       # Case {0, 1}
            self.unnormalized_prob_vis = self.unnormalized_prob_vis_01
            self.part_fun = self.part_fun_01
        elif type_rbm == 1:
            self.unnormalized_prob_vis = self.unnormalized_prob_vis_11
            self.part_fun = self.part_fun_11
        else:
            print("Invalid type of RBM: %s" % type_rbm)

        return

    def set_w(self, w, b, c):
        """Set the weights of the RBM
        
        Parameters
        ----------
            w : 2d np.array
                Matrix with two body weights, with dimensions (nh, nv)
            b : 1d np.array
                Matrix with visible biases
            c : 1d np.array
                Matrix with hidden biases
        """
        self.w = w
        self.b = b
        self.c = c
        self.Z = None       # Z has not been computed yet
        return

    def rand_w(self, sigma, SEED=1234):
        """Initialize random weights
        
        The distribution is a zero mean normal distribution with variance sigma
        for w, b, c.
        """
        np.random.seed(SEED)
        self.w = sigma * np.random.randn(self.nh, self.nv)
        self.b = sigma * np.random.randn(self.nv)
        self.c = sigma * np.random.randn(self.nh)
        self.Z = None       # Z has not been computed yet
        return

    def save_w(self, filename):
        """Save weights and biases to file"""
        # append biases to weight matrix
        wmod = np.hstack([self.w, self.c[:, np.newaxis]])
        bmod = np.hstack([self.b, 0])[np.newaxis, :]
        wmod = np.vstack([wmod, bmod])
        # save file
        np.savetxt(filename, wmod)
        return

    def load_w(self, filename):
        """Load weights and biases from file"""
        # load data from file
        wmod = np.loadtxt(filename)
        # split into weights and biases
        self.w = wmod[:-1, :-1]
        self.b = wmod[-1, :-1]
        self.c = wmod[:-1, -1]
        return

    # ----------------------------------------------------------------------- #
    #                           Operations of RBM                             #
    # ----------------------------------------------------------------------- #

    def propup(self, v):
        return self.prob_con(np.dot(self.w, v) + self.c[:, np.newaxis]).T

    def propdown(self, h):
        return self.prob_con(np.dot(h, self.w) + self.b[np.newaxis, :]).T

    def mean_value(self, prob):
        return prob - self.type_rbm * (1 - prob)

    def samp(self, p):
        """Sample an array of floating point elements
        
        Parameters
        ----------
            p : n dimensional numpy.array
                contains floating point numbers between 0 and 1

        Returns
        -------
            s : n dimensional numpy.array
                sampled values that are 1 with probability p and {0, -1} 
                otherwise
        """
        # Check if probability to be up is bigger than a random number
        rand = np.random.rand(*p.shape)
        idx_up = p - rand > 0
        # Put up neurons to 1 and the rest to 0 or -1
        if self.type_rbm == 0:
            s = np.zeros(p.shape)
            s[idx_up] = 1
        else:
            s = -np.ones(p.shape)
            s[idx_up] = 1
        return s

    def prob_con(self, x):
        """Function to compute conditional probabilities for {0, 1} neurons
        
        It can be shown that
            p( h | v ) = prob_con( w * v + c )
            p( v | h ) = prob_con( h * w + b )
        """
        if self.type_rbm == 0:
            return 1 / (1 + np.exp(-x))
        else:
            return 1 / (1 + np.exp(-2*x))

    def prob_vis(self, v):
        """Probability of visible states

        Parameters
        ----------
            v : 2d np.array 
                The visible neurons are in columns

        Returns
        -------
            P : 1d np.array 
                Probabilities of input states
        """
        return self.unnormalized_prob_vis(v) / self.part_fun()

    def log_likelihood(self, v):
        """Log-likelihood of a dataset"""
        P = self.prob_vis(v)
        return np.sum(np.log(P))

    def reconstruct_err(self, batch):
        v = batch.T
        ph = self.propup(v)
        h = self.samp(ph)
        pv = self.propdown(h)
        v_recon = self.mean_value(pv)
        return np.sum((v-v_recon)**2)

    # ----------------------------------------------------------------------- #
    #                           Training routines                             #
    # ----------------------------------------------------------------------- #

    def train(self, train_type, dataset, hyperparameters, num_print, file_w):

        train_set, train_probs, test_set, test_probs = dataset
        lrate, moment, l2_reg, min_epochs, max_epochs, minibatch_size, k = \
                                                                hyperparameters

        if test_set is not None and test_set.size != 0:
            print_test = True
        else:
            print_test = False

        def print_values(set_, probs, train=True):
            model_probs = self.prob_vis(set_.T)
            KL = np.sum(probs * np.log(probs / model_probs))
            log_likelihood = np.sum(np.log(model_probs)) / len(set_)
            sum_probs = np.sum(model_probs)
            rec_err = self.reconstruct_err(set_) / len(set_)

            if train: 
                print("Train", end="")
            else:
                print("Test ", end="")
            print("  KL  %f  LogLKH  %.3f  SumProb  %.3f  RecErr(CD1)  %.3f" % \
                    (KL, log_likelihood, sum_probs, rec_err), end="")
            if train: 
                print("  RecErr(Real)  %.3f" % (self.rec_err / len(train_set)))
            else:
                print("\n", end="")

            return model_probs, KL

        def print_probs(set_, probs, model_probs):
            print("\tProbabilities: ")
            for i in range(len(set_)):
                print("\t%-16s  %f   %f" % (list(set_[i]), model_probs[i], \
                      probs[i]))
            sum_probs = sum(probs)
            log_lkh = sum(np.log(probs)) / len(set_)
            print("\tLogLKH_Data  %.3f  SumProb_Data:  %.3f" % (log_lkh, sum_probs))
            return

        # Print training set and the initial probabilities
        print("Initial values")
        model_probs, train_KL = print_values(train_set, train_probs, \
                                             train=True)
        print_probs(train_set, train_probs, model_probs)
        if print_test:
            model_probs, test_KL = print_values(test_set, test_probs, \
                                                train=False)
            print_probs(test_set, test_probs, model_probs)

        #--------- Do the training ---------#

        init_KL = train_KL
        itime = time.time()
        for epoch in range(max_epochs):
            self.train_epoch(train_type, train_set, lrate, moment, l2_reg, k, 
                       minibatch_size, train_probs)

            # Print results every num_print epochs
            if (epoch + 1) % num_print == 0:
                print("Epoch %5d" % (epoch + 1))
                _, train_KL = print_values(train_set, train_probs, train=True)
                if print_test:
                    print_values(test_set, test_probs, train=False)
                print("\n", end="")

                # Check the training progress is correct 
                if np.isnan(train_KL) or np.isinf(train_KL):
                    print("Bad parameters")
                    break
                if epoch > min_epochs:
                    if train_KL > 2*init_KL:
                        print("Bad parameters")
                        break

        #-------- Print final values ---------#

        print("Final values")
        model_probs, train_KL = print_values(train_set, train_probs, \
                                             train=True)
        print_probs(train_set, train_probs, model_probs)
        if print_test:
            model_probs, test_KL = print_values(test_set, test_probs, \
                                                train=False)
            print_probs(test_set, test_probs, model_probs)

        #-------- Save weights --------#
        if file_w is not None:
            self.save_w(file_w)

        #-------- Elapsed time ---------#
        print("Total elapsed time: %fs" % (time.time() - itime))
        return

    def train_epoch(self, train_type, tset, lrate, moment, l2_reg, k,
            minibatch_size=None, probs=None):
        """Step of training algorithm

        Parameters
        ----------
            train_routine : string
                it can be "GradDesc" for gradient descent or "ContDiv" for 
                contrastive divergence.
            tset : 2d np.array 
                the training states are rows
            lrate : float
                learning rate of the algorithm
            moment : float
                momentum of the algorithm
            minibatch_size : integer
                length of the batchs, if not provided a single batch is used.
            probs : 1d np.array 
                Probabilities assigned to training states. If not provided, 
                the training set is considered equiprobable.
        """

        train_routine, persistent, weighted = train_type

        # Normalize probabilities of states and make them equiprobable if not
        # provided
        if probs is None:
            probs = np.ones(len(tset)) / len(tset)

        if minibatch_size is None:      # Use the whole training set
            minibatch_size = len(tset)

        # Shuffle indices of the training set
        len_tset = len(tset)
        num_batch = int(np.ceil(len_tset / minibatch_size))
        idx = np.arange(len_tset)
        np.random.shuffle(idx)

        # Iterate over batches and perform training
        self.rec_err = 0.0
        for i in range(num_batch):
            # Build current batch
            i1 = i * minibatch_size
            i2 = min((i+1)*minibatch_size, len_tset)
            batch_state = tset[idx[i1:i2], :]
            batch_probs = probs[idx[i1:i2]]

            # Select training routine
            if train_routine == "GradDesc":
                dW = self.exact_gradient_descent(batch_state, batch_probs)
            elif train_routine == "ContDiv":
                dW = self.contrastive_divergence(batch_state, batch_probs,
                                                 k, persistent, weighted)

            self.apply_gradient(dW, lrate, moment, l2_reg, sum(batch_probs))

        return

    def exact_gradient_descent(self, batch, probs):
        """Step of the exact gradient descent method

        Parameters
        ----------
            batch : 2d np.array 
                Array with the batch where training states are rows
            lrate : float
                learning rate of the algorithm
            moment : float
                momentum of the algorithm
            probs : 1d numpy.array
                probabilities assigned to each state of the batch
        """

        # Positive phase
        ph = self.propup(batch.T)
        h_avg = self.mean_value(ph)
        pdw = np.dot(np.dot(h_avg.T, np.diag(probs)), batch)
        pdb = np.dot(probs, batch)
        pdc = np.dot(probs, h_avg)

        # Negative phase
        all_vis = itertools.product([-self.type_rbm, 1], repeat=self.nv)
        all_vis = np.array(list(all_vis))
        neg_probs = self.prob_vis(all_vis.T) * sum(probs)

        ph = self.propup(all_vis.T)
        h_avg = self.mean_value(ph)

        ndw = np.dot(np.dot(h_avg.T, np.diag(neg_probs)), all_vis)
        ndb = np.dot(neg_probs, all_vis)
        ndc = np.dot(neg_probs, h_avg)

        return pdw, pdb, pdc, ndw, ndb, ndc

    def contrastive_divergence(self, batch, probs, k, persistent=False, \
                               weighted=False):
        """Step of Contrastive divergence

        Parameters
        ----------
            batch : 2d numpy.array  
                The training are in the rows
            lrate : float
                Learning rate.
            k : integer
                Number of steps of contrastive divergence. If not provided is 
                assumed to be CD-1.
            moment : float
                Momentum, that can vary between 0 (no momentum) and 1.
            probs : 1d numpy.array
                probabilities assigned to each state of the batch
        """

        # Positive phase
        ph = self.propup(batch.T)
        h_avg = self.mean_value(ph)

        pdw = np.dot(np.dot(h_avg.T, np.diag(probs)), batch)
        pdb = np.dot(probs, batch)
        pdc = np.dot(probs, h_avg)

        # Negative phase

        # load persistent state
        if persistent:
            if self.ph_old is not None:
                ph = self.ph_old

        for i in range(k-1):        # Iterations to reach equilibrium
            h  = self.samp(ph)
            pv = self.propdown(h)
            v  = self.samp(pv)
            ph = self.propup(v)

        # Last iteration without sampling to remove noise
        h = self.samp(ph)
        pv = self.propdown(h)
        v_avg = self.mean_value(pv)
        ph = self.propup(v_avg)
        h_avg = self.mean_value(ph)

        # Compute reconstruction error
        self.rec_err += np.sum((batch.T - v_avg)**2)

        # save persistent state
        if persistent:
            self.ph_old = ph

        if weighted:
            # compute unnormalized probabilities
            neg_probs = self.unnormalized_prob_vis(v_avg, scale=False)
            neg_probs /= sum(neg_probs)
            neg_probs *= sum(probs)
        else:
            neg_probs = probs

        # increment of weights and biases for negative phase
        ndw = np.dot(np.dot(h_avg.T, np.diag(neg_probs)), v_avg.T)
        ndb = np.dot(neg_probs, v_avg.T)
        ndc = np.dot(neg_probs, h_avg)

        return pdw, pdb, pdc, ndw, ndb, ndc

    def apply_gradient(self, dW, lr, m, l2_reg, sum_probs):

        pdw, pdb, pdc, ndw, ndb, ndc = dW

        self.incw = m * self.incw + lr * (pdw - ndw - l2_reg * self.w * sum_probs)
        self.incb = m * self.incb + lr * (pdb - ndb)
        self.incc = m * self.incc + lr * (pdc - ndc)

        self.w += self.incw
        self.b += self.incb
        self.c += self.incc

        self.Z = None   # The partition function changes for the new weights
        return

    # ----------------------------------------------------------------------- #
    #                     Routines for {0, 1} neurons                         #
    # ----------------------------------------------------------------------- #

    def unnormalized_prob_vis_01(self, v, scale=False):
        """Non-normalized probability of visible states

        Parameters
        ----------
            v : 2d np.array 
                The visible neurons are in columns

        Returns
        -------
            P : 1d np.array 
                Non-normalized probabilities of input states
        """
        aux1 = np.exp(np.dot(self.b, v))
        aux2 = 1 + np.exp(np.dot(self.w, v) + self.c[:, np.newaxis])
        if scale:
            aux2 /= np.mean(aux2)
        aux2 = np.prod(aux2, axis=0)
        return aux1 * aux2

    def part_fun_01(self):
        """Partition function of the RBM
        
        Returns
        -------
            Z : float
                partition function of the system with the current weights.

        The value of Z is stored, so it is only computed once for every set of
        weights.
        """
        
        if self.Z is None:          # If Z has not been computed yet
            if self.nv <= self.nh:
                v = np.array(list(itertools.product([0, 1], repeat=self.nv))).T
                aux1 = np.exp(np.dot(self.b, v))
                aux2 = 1 + np.exp(np.dot(self.w, v) + self.c[:, np.newaxis])
                aux2 = np.prod(aux2, axis=0)
                self.Z = np.dot(aux1, aux2)
            else:
                h = np.array(list(itertools.product([0, 1], repeat=self.nh)))
                aux1 = np.exp(np.dot(h, self.c))
                aux2 = 1 + np.exp(np.dot(h, self.w) + self.b[np.newaxis, :])
                aux2 = np.prod(aux2, axis=1)
                self.Z = np.dot(aux1, aux2)

        return self.Z

    # ----------------------------------------------------------------------- #
    #                     Routines for {-1, 1} neurons                        #
    # ----------------------------------------------------------------------- #

    def unnormalized_prob_vis_11(self, v, scale=False):
        aux1 = np.exp(np.dot(self.b, v))
        aux2 = 2 * np.cosh(np.dot(self.w, v) + self.c[:, np.newaxis])
        if scale:
            aux2 /= np.mean(aux2)
        aux2 = np.prod(aux2, axis=0)
        return aux1 * aux2

    def part_fun_11(self):
        if self.Z is None:
            if self.nv <= self.nh:
                viter = itertools.product([-1, 1], repeat=self.nv)
                v = np.array(list(viter)).T
                aux1 = np.exp(np.dot(self.b, v))
                aux2 = 2 * np.cosh(np.dot(self.w, v) + self.c[:, np.newaxis])
                aux2 = np.prod(aux2, axis=0)
                self.Z = np.dot(aux1, aux2)
            else:
                h = np.array(list(itertools.product([-1, 1], repeat=self.nh)))
                aux1 = np.exp(np.dot(h, self.c))
                aux2 = 2 * np.cosh(np.dot(h, self.w) + self.b[np.newaxis, :])
                aux2 = np.prod(aux2, axis=1)
                self.Z = np.dot(aux1, aux2)

        return self.Z

if __name__ == '__main__':
    print("Testing")
