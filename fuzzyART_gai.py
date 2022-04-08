from functools import partial

import numpy as np


l1_norm = partial(np.linalg.norm, ord=1, axis=-1)


class FuzzyART:
    """
    Fuzzy ART

    An unsupervised clustering algorithm
    """

    def __init__(self, alpha=1.0, gamma=0.01, rho=0.5, complement_coding=True):
        """
        :param alpha: learning rate [0,1]
        :param gamma: regularization term >0
        :param rho: vigilance [0,1]
        :param complement_coding: use complement coding scheme for inputs
        """
        self.alpha = alpha  # default learning rate
        self.beta = 1 - alpha
        self.gamma = gamma  # default choice parameter
        self.rho = rho  # default vigilance
        self.complement_coding = complement_coding

        self.match_num = 2 # number of best matching categories

        self.w = None

    def _init_weights(self, x):
        self.w = np.atleast_2d(x)

    def _complement_code(self, x):
        if self.complement_coding:
            _x = np.hstack((x, 1 - x))
        else:
            _x = x
        return _x

    def _add_category(self, x):
        self.w = np.vstack((self.w, x))

    def _score_category(self, x, rho):
        if rho is None:
            _rho = self.rho
        else:
            _rho = rho

        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        threshold = fuzzy_norm / l1_norm(x) >= _rho
        return scores * threshold.astype(int)

    def _match_category(self, x, rho):
    #    scores = self._score_category(x, rho)
     #   best_idx = np.flip(np.argsort(scores))
      #  print(scores)
       # print(best_idx)
        #print('----')

        if rho is None:
            _rho = self.rho
        else:
            _rho = rho
    
        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        threshold = fuzzy_norm / l1_norm(x) >= _rho

        if np.any(threshold):
            best_cat = np.argmax(scores * threshold.astype(int))
        else:
            best_cat = -1
        return best_cat

#        best_cats = np.full(self.match_num, -1)
 #       for j, idx in enumerate(best_idx):
  #          best_cats[j] = idx
   #     return best_cats

    def _update_weight(self, category, sample, alpha):
        if alpha is None:
            _alpha = self.alpha
        else:
            _alpha = alpha
        _beta = 1 - _alpha
        w = self.w[category]
        self.w[category] = _alpha * np.minimum(sample, w) + _beta * w

    def train(self, x, epochs=1, rho=None, alpha=None):
        """
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1]
        :param epochs: number of training epochs, the training samples are
        shuffled after each epoch
        :return: self
        """

        samples = self._complement_code(np.atleast_2d(x))

        if self.w is None:
            self._init_weights(samples[0])

        for epoch in range(epochs):
            for sample in np.random.permutation(samples):
                category = self._match_category(sample, rho)#[0]
                if category == -1:
                    self._add_category(sample)
                else:
                    self._update_weight(category, sample, alpha)
        return self

    def test(self, x, rho=None):
        """
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1]
        :return: category IDs for each provided sample
        """

        samples = self._complement_code(np.atleast_2d(x))

        categories = np.zeros((samples.shape[0], self.match_num))
        for i, sample in enumerate(samples):
            categories[i] = self._match_category(sample, rho)
        return categories

    # to be deprecated
    def _set_properties(self, alpha=None, gamma=None, rho=None):
        # log old properties
        old_prop = self.alpha, self.beta, self.gamma, self.rho
        # update properties
        ispropvaried = False
        for propname, new_val in zip(('alpha', 'gamma', 'rho'), (alpha, gamma, rho)):
            if self.__dict__[propname] != new_val and new_val is not None:
                self.__dict__[propname] = new_val
                ispropvaried = True
        self.beta = 1 - self.alpha
        # log new properties
        new_prop = self.alpha, self.beta, self.gamma, self.rho
        # message about update
        if ispropvaried:
            print("""
                (alpha, beta, gamma, rho) - changed from\n
                {} to\n
                {}
                """.format(old_prop, new_prop))
        return ispropvaried

    def getcat_bipole(self):
        self.featnum = self.w.shape[1] // 2
        return self.w[:, :self.featnum], 1 - self.w[:, self.featnum:]

    def getcat_centre(self):
        return np.add(*self.getcat_bipole()) / 2

    def getcat_vertex(self):
        u, v = self.getcat_bipole()
        vertnum = 2 ** self.featnum
        vertices = np.zeros((self.w.shape[0], vertnum, self.featnum))
        for featidx in range(self.featnum):
            stride = 2 ** featidx
            uvidx = np.reshape(np.arange(vertnum), (stride, -1))
            uidx, vidx = np.split(uvidx, 2, axis=1)
            vertices[:, uidx.flatten(), featidx] = np.tile(u[:, featidx], (vertnum // 2, 1)).T
            vertices[:, vidx.flatten(), featidx] = np.tile(v[:, featidx], (vertnum // 2, 1)).T

        return vertices


