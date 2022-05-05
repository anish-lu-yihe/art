from kit_basic import *


class FuzzyART:
    """
    Fuzzy ART

    An unsupervised clustering algorithm
    """

    def __init__(self, feature_num, alpha=1.0, beta=1.0, gamma=1e-5, rho=0.5, best_match_num=1):
        """
        :param feature_num: number of features
        :param alpha: learning rate [0,1]
        :param beta: unlearning rate [0,1]
        :param gamma: regularization term >0
        :param rho: vigilance [0,1]

        complement coding scheme for inputs is always used
        """
        self.featnum = feature_num
        self.alpha = alpha  # default learning rate
        self.beta = beta  # default unlearning rate
        self.gamma = gamma  # default choice parameter
        self.rho = rho  # default vigilance

        self.match_num = best_match_num
        self.multi_match = self.match_num > 1

        self.w = None

    def _init_weights(self, x):
        self.w = np.atleast_2d(x)

    def _complement_code(self, x):
        _x = np.hstack((x, 1 - x))
        return _x

    def _scale_weight(self, s):
        if s is None:
            _w = self.w
        elif s <= 0.5:
            u, vc = np.hsplit(self.w, 2)  # w = (u, v^c) where u and v are bipoles of hyperbox
            sDvu = s * (1 - vc - u)
            _w = np.minimum(np.maximum(self.w + np.tile(sDvu, 2), 0), 1)
        else:
            raise ValueError("Wrong scaling parameter!")
        return _w

    def _getuv(self, s):
        _w = self._scale_weight(s)
        _u, _vc = np.split(_w, 2, axis=1)
        return _u, 1 - _vc

    def _getct(self):
        return np.mean(self._getuv(None), axis=0)

    def _getvx(self, s):
        u, v = self._getuv(s)
        vertnum = 2 ** self.featnum
        vertices = np.zeros((self.w.shape[0], vertnum, self.featnum))
        for featidx in range(self.featnum):
            stride = 2 ** featidx
            uvidx = np.reshape(np.arange(vertnum), (stride, -1))
            uidx, vidx = np.split(uvidx, 2, axis=1)
            vertices[:, uidx.flatten(), featidx] = np.tile(u[:, featidx], (vertnum // 2, 1)).T
            vertices[:, vidx.flatten(), featidx] = np.tile(v[:, featidx], (vertnum // 2, 1)).T

        return vertices

    def _add_category(self, x):
        self.w = np.vstack((self.w, x))

    def _score_category(self, x, rho, s):
        if rho is None:
            _rho = self.rho
        else:
            _rho = rho

        fuzzy_weights = np.minimum(x, self._scale_weight(s))
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        threshold = fuzzy_norm / l1_norm(x) >= _rho
        return scores, threshold

    def _match_category(self, x, rho, s):
        scores, threshold = self._score_category(x, rho, s)
        if self.multi_match:
            sort_idx = np.argsort(scores)
            best_idx = np.where(threshold[sort_idx], sort_idx, -1)
            best_cat = np.full(self.match_num, -1)
            cat_num = np.minimum(self.match_num, threshold.size)
            best_cat[:cat_num] = np.flip(best_idx[-cat_num:])
        else:
            best_cat = np.argmax(scores),
            if not threshold[best_cat]:
                best_cat = -1,
        return best_cat

    def _expansion(self, category, sample, alpha):
        if alpha is None:
            _alpha = self.alpha
        else:
            _alpha = alpha
        w = self.w[category]
        self.w[category] = _alpha * np.minimum(sample, w) + (1 - _alpha) * w

    def _contraction(self, category, sample, beta, whichidx='least-loss'):
        w = self.w[category]
        if whichidx == 'least-loss':  # index leading to least loss of categorical hyperbox area
            featidx = np.argmin(sample - w)
        elif whichidx == 'random':
            featidx = np.random.randint(w.size)
        self.w[category, featidx] = beta * sample[featidx] + (1 - beta) * w[featidx]

    def _resample_fromuv(self, u, v, number):
        rand_init = np.random.rand(number, self.featnum)
        return u + (v - u) * rand_init

    def _resample_category(self, category, number, s):
        u, v = [uv[category] for uv in self._getuv(s)]
        return self._resample_fromuv(u, v, number)

    def _resample_vertex(self, category, number, s):
        vertices = self._getvx(s)[category]
        np.random.shuffle(vertices)
        return vertices[:number]

    def train(self, x, epochs=1, rho=None, alpha=None, s=None):
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
                category = self._match_category(sample, rho, s)[0]
                if category == -1:
                    self._add_category(sample)
                else:
                    self._expansion(category, sample, alpha)
        return self

    def test(self, x, rho=None, s=None):
        """
        :param x: 2d array of size (samples, features), where all features are
         in [0, 1]
        :return: category IDs for each provided sample
        """

        samples = self._complement_code(np.atleast_2d(x))

        categories = np.zeros((samples.shape[0], self.match_num))
        for i, sample in enumerate(samples):
            categories[i] = self._match_category(sample, rho, s)
        return categories

    def replay_null(self, total_number, s):
        u, vc = np.split(np.min(self._scale_weight(s), axis=0), 2)
        replay = self._resample_fromuv(u, 1 - vc, total_number)
        label_exp = np.full(total_number, -1)
        return replay, label_exp

    def replay_1cat(self, category, total_number, s, scheme='in-box'):
        allidx = np.atleast_1d(category)
        replay_percat = total_number // allidx.size
        replay = np.empty((0, self.featnum))
        label_exp = np.empty((0, 0), dtype=int)
        if scheme == 'in-box':
            _resample = self._resample_category
        elif scheme == 'vertex':
            _resample = self._resample_vertex
        else:
            raise TypeError("Replay scheme does NOT exist!")
        for catidx in allidx:
            catreplay = _resample(catidx, replay_percat, s)
            replay = np.vstack((replay, catreplay))
            label_exp = np.append(label_exp, np.full((catreplay.shape[0], 1), catidx))

        randidx = np.random.permutation(replay.shape[0])
        return replay[randidx], label_exp[randidx]

    # for external uses, maybe deprecated later
    def getcat_bipole(self, s=None):
        return self._getuv(s)

    def getcat_centre(self):
        return self._getct()

    def getcat_vertex(self, s=None):
        return self._getvx(s)


