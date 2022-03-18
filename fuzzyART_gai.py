from fuzzy_art import *

class FuzzyARTgai(FuzzyART):
    def _set_properties(self, alpha=None, gamma=None, rho=None):
        old_prop = self.alpha, self.beta, self.gamma, self.rho
        ispropvaried = False
        if alpha != self.alpha and alpha is not None:
            self.alpha = alpha
            self.beta = 1 - alpha
            ispropvaried = True
        if gamma != self.gamma and gamma is not None:
            self.gamma = gamma
            ispropvaried = True
        if rho != self.rho and rho is not None:
            self.rho = rho
            ispropvaried = True

        new_prop = self.alpha, self.beta, self.gamma, self.rho
        if ispropvaried:
            print("""
                (alpha, beta, gamma, rho) - changed from\n
                {} to\n
                {}
                """.format(old_prop, new_prop))
