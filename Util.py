import random
from itertools import product
from sklearn.utils.validation import (check_is_fitted, check_array, FLOAT_DTYPES)
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import pickle


# ============= Function of Basis =================
def get_n_features(n_features, degree=3, bias=True):
    combinations = _combinations(n_features, degree)
    if not bias:
        n_output_features = sum(1 for _ in combinations) - 1
    else:
        n_output_features = sum(1 for _ in combinations)
    return n_output_features


def Fourier_Kernel(X,degree=3, bias=True):
    """

    :param X: array -> (n_samples, n_features)
    :return:
    """
    X = np.array(X)
    n_samples, n_features = X.shape

    combinations = _combinations(n_features, degree)
    n_output_features = sum(1 for _ in combinations)
    if (not bias):
        n_output_features -= 1

    #n_output_features = sum(1 for _ in combinations) - 1
    XP = np.empty((n_samples, n_output_features), dtype=X.dtype)
    i = 0

    combinations = _combinations(n_features, degree)
    for e in combinations:
        if e == (0, 0, 0, 0) and not bias:
            continue

        e = np.array(e)
        temp = X.dot(e)
        temp = np.cos(temp * np.pi)
        XP[:, i] = temp
        i = i + 1
    return np.array(XP)


def _combinations(n_features, degree, interaction_only=False):
    comb = product(range(degree+1), repeat=n_features)
    #comb = combinations(range(degree + 1), n_features)
    return comb



# ===================================================

# ==================== Plotting function ===========================================





#=================================== Class ==========================================

class Linear_Approximator():
    def __init__(self, normalized, featurized, n_feature=10, alpha=0.001, action_n = 1,
                 lambda2=0, discount_factor=1):
        self.lambda2=lambda2
        self.discount_factor = discount_factor

        self.normalized = normalized
        self.featurized = featurized
        self.n_feature =n_feature
        self.alpha= alpha
        self.action_n = action_n
        self.weight = np.zeros((self.action_n, self.n_feature))

        self.v_w = lambda x: self.weight.dot(x)
        self.e = np.zeros((self.action_n, self.n_feature))
        self.dv_w = lambda x: x

    def alpha(self,alpha):
        self.alpha = alpha

    def initial(self):
        self.weight = np.zeros((self.action_n, self.n_feature))
        self.e = np.zeros((self.action_n, self.n_feature))
        self.v_w = lambda x: self.weight.dot(x)
        self.dv_w = lambda x: x

    def perdict(self, state, a=-1, featurized=True):
        if not featurized:
            state = self.normalized(state)
            state = self.featurized(state)

        if self.action_n == 1:
            a = 0
        result = self.weight[a].dot(state)
        if a == -1:
            result = self.weight.dot(state)
        return result

    def update(self, state, target, a=0, featurized=True):
        #todo: check correctness for e-trace
        if not featurized:
            state = self.normalized(state)
            state = self.featurized(state)
        if self.action_n == 1:
            a=0

        td_error = target - self.v_w(state)[a]
        temp1= self.e[a]
        temp2 = self.dv_w(state)
        temp3 = self.weight[a]
        self.e[a] = self.discount_factor * self.lambda2 * self.e[a] + self.dv_w(state)

        temp = self.alpha * td_error * self.e[a]
        self.weight[a] += temp
        self.v_w = lambda x: self.weight.dot(x)
        self.dv_w = lambda x: x

class Softmax_Approximator():
    def __init__(self, normalized, featurized, n_feature=10, alpha=0.001, action_n = 1, discount_factor= 1, lambda2=0 ):
        self.discount_facotr= discount_factor
        self.lambda2= lambda2
        self.normalized = normalized
        self.featurized = featurized
        self.n_feature = n_feature
        self.action_n = action_n
        self.alpha= alpha
        #self.weight = np.zeros((action_n, n_feature))
        self.weight = np.zeros((self.n_feature, self.action_n))
        self.one_hot = np.eye(action_n)

    def initial(self):
        self.weight = np.zeros((self.n_feature, self.action_n))
        #self.weight=np.random.rand(self.n_feature, self.action_n)
        self.e = np.zeros((self.n_feature, self.action_n))


    def _softmax_grad(self,prob):
        s = prob.reshape(-1, 1)
        a = np.diagflat(s)
        b = np.dot(s, s.T)

        return np.diagflat(s) - np.dot(s, s.T)

    def _softmax(self,state):
        z = state.dot(self.weight)
        exp = np.exp(z)
        return exp / np.sum(exp)

    def to_phi(self,state):
        state = self.normalized(state)
        state = self.featurized(state)
        return state
    def sample(self,state,featurized=True):
        if not featurized:
            state = self.normalized(state)
            state = self.featurized(state)
        p = self._softmax(state)
        action = np.random.choice(range(self.action_n), p=p)
        return action

    def update(self,state, a, td_error, featurized=True):
        state = np.array(state)
        if not featurized:
            state = self.normalized(state)
            state = self.featurized(state)
        p = self._softmax(state)
        dsoftmax = self._softmax_grad(p)[a, :]
        dlog = dsoftmax / p[a]
        temp1 = state[None,:].T
        temp2 = dlog[None,:]
        grad = temp1.dot(dlog[None, :])
        self.e = self.discount_facotr * self.lambda2 * self.e + grad
        self.weight += self.alpha * self.e * td_error

        #self.weight += self.alpha * grad * qvalue



class featurization:
    def __init__(self, bound, normalize, name = "rbf", degree = 3, n_feature = 100):
        """""
        bound: 2d array [[upper],[lower]]
        """""

        self.bound = bound
        self.n_feature = n_feature*4
        #self.normalize = normalize
        if name == "rbf":
            state_examples = [normalize(bound.sample()) for i in range(50000)]
            state_examples.append((3,12,1.8,8.0))
            state_examples.append((-3, -12, -1.8, -8.0))
            state_examples = pickle.load(open("sample_state", "rb"))

            n = n_feature

            self.featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=n)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n))
            ])

            self.featurizer.fit(state_examples)


    #State is in 1xD [] shape
    def transform(self, state):
        featurized = self.featurizer.transform([state])
        return featurized[0]



#use softMax technique to solve the q_vales
def softMax_Index(q_values):
    e_x = np.exp(q_values - np.max(q_values))
    p = e_x / e_x.sum(axis=0)

    r = random.random()

    #print("p is ")
    #print(p)
    #print("r is ")
    #print(r)

    b = 0
    n = 0
    for i in p:
        b = b+i
        if r <= i:
            return n
        n=n+1

    return n-1



#Test:

