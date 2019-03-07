import numpy as np
import random




def random_action(actions, state, state_table):
    n = len(actions)
    action = random.randint(0, n-1)
    return actions[action], action


def epsilon_greedy(Q_table, epsilon, nA, ifcontinue = False):
    def policy_fn(state):
        A = np.ones(nA, dtype=float) * epsilon / nA
        if ifcontinue:
            temp = Q_table(state)
            temp = list(temp)
            index = [i for i, x in enumerate(temp) if x == max(temp)]
            best_action = np.argmax(Q_table(state))
            A[best_action] += 1.0 - epsilon
            #for action in index:
            #    A[action] += (1.0-epsilon)/len(index)
        else:
            try:
                temp = Q_table[state]
                temp=list(temp)
                index = [i for i, x in enumerate(temp) if x == max(temp)]
                #best_action = np.argmax(temp)
                #A[best_action] += 1.0 - epsilon
            except:
                print("error")
            for action in index:
                A[action] += (1.0-epsilon)/len(index)
        if sum(A) > 1.00001 or sum(A) < 0.999999999:
            print("Wrong!: ", A, "sum", sum(A),"Q_table", Q_table[state], "epsion: ", epsilon, "index", index)


        return A
    return policy_fn

def random_search(Q_table,ifcontinue = False, alpha=0.5):
    def policy_fn():
        new_policy = None
        if ifcontinue:
            pass
        else:
            temp = np.array(Q_table)

            mu, sigma = 0, alpha
            s = np.random.normal(mu,sigma, temp.shape)

            change = s*temp
            new_policy = temp + change

        return new_policy
    return policy_fn


def extract_policy(Q_table):
    def policy_fn(state):
        return Q_table[state]

    return policy_fn

def softmax(Q_table, epsilon, nA, ifcontinue = False):
    def policy_fn(state):
        if ifcontinue:
            temp = Q_table(state)

        else:
            temp = Q_table[state]

        x=temp*epsilon
        try:
            result = np.exp(x) / np.sum(np.exp(x), axis=0)
        except:
            print("??")
            print(np.exp(x))
            print(np.sum(np.exp(x)))
            A = np.zeros(nA, dtype=float)
            best_action = np.argmax(temp)
            A[best_action] += 1.0
            result = A

        while sum(result) > 1.00001 or sum(result) < 0.999999999:
            #Too large, greedy select directly
            A = np.zeros(nA, dtype=float)
            best_action = np.argmax(temp)
            A[best_action] += 1.0

            result = A


        if epsilon < 0.001:
            A = np.zeros(nA, dtype=float)
            best_action = np.argmax(temp)
            A[best_action] += 1.0

            result = A


        if sum(result) > 1.00001 or sum(result) < 0.999999999:
            print("wrong")
        return result
    return policy_fn


class linear_function_estimator:

    def __init__(self, size):
        self.weight = np.zeros(size)

    def value(self, state):
        return self.weight.dot(state)

    def update(self, weight):
        self.weight = weight

