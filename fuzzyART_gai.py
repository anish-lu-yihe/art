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

    def getcat_bipolars(self):
        featnum = self.w.shape[1] // 2
        return self.w[:, :featnum], 1 - self.w[:, featnum:]

    def getcat_centre(self):
        return np.add(*self.getcat_bipolars()) / 2

    def getcat_vertices(self):
        pass