import numpy as np


class observation_bound:
    def __init__(self, n=2, bound=None):
        # There are n bound values
        self.state_space = n
        # (n,2), n variable, 2 status
        self.bound = bound

    def set(self, up, low):
        bound = np.concatenate([[up], [low]])
        self.bound = bound

    def check(self, state):
        for i in range(len(state)):
            upper = self.bound[0][i]
            lower = self.bound[1][i]
            check = state[i]
            # It out of range
            if check < lower or check > upper:
                return False

        # pass the test
        return True
    def sample(self):
        temp = self.bound.T
        #random.uniform(c[1], c[0])
        result = [random.uniform(c[1],c[0]) for c in temp]
        return result