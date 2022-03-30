import numpy as np

from fuzzy_art import *

class FuzzyARTgai(FuzzyART):
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


