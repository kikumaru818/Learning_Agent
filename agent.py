import numpy as np
from collections import defaultdict as dict
import Util
import policy
import pickle
import matplotlib.pyplot as plt
from plot_3_pics_with_ylim import plot_trails
import itertools

class evaluate_agent:
    def __init__(self, env, policy_name="random_action", gym=False, state_type = "Tabular",state_n=23, degree = 3, expansion='f',trail_n=1, num_e=100):
        self.num_e = num_e
        self.trail_n = trail_n
        self.degree=degree


        self.env = env
        self.env_name = env.name
        self.policy_type = policy_name
        self.policy =getattr(policy, policy_name)
        self.state_type = state_type


        if gym:
            self.action_n = env.action_space.n
            self.action_space = [i for i in range(self.action_space)]
        else:
            self.action_n = len(env.action_space)
            self.action_space = env.action_space

        if state_type == "Tabular":
            self.state_n = state_n


        self.expansion = expansion
        if state_type == "continue":
            self.normalized = env.normalize

            if expansion == "rbf":
                self.degree = degree
                self.e = 2.0/(self.degree-1)
                prod = itertools.product(list(range(0, degree + 1)), repeat=env.n_state_space)
                temp = []
                for p in prod:
                    temp.append(p)
                c = np.array(temp)
                c = c/degree
                a0 = 1.0 / (2 * np.pi * self.e ** 2) ** 0.5
                self.featurized = lambda x: a0 * (np.exp((-(np.linalg.norm(c - x, axis=1))**2) / 2 / (self.e ** 2)))
                self.n_featurized = len(temp)

            if expansion == "f":
                self.degree = degree

                prod = itertools.product(list(range(0, degree + 1)), repeat=env.n_state_space)
                temp = []
                for p in prod:
                    temp.append(p)
                c = np.array(temp)
                self.featurized = lambda x: np.cos(np.pi*np.dot(c,x))
                self.n_featurized = len(temp)

    def grid_search_continue(self, learning_agent, agent_name = "",alphas=[0.001], epsilons=[0.2], degrees=[3],path = "Result/",decay=True,more=""):
        a=dict()
        trails=[]

        for alpha in alphas:
            for epsilon in epsilons:
                for degree in degrees:
                    for i in range(self.trail_n):
                        n_out_features = self.n_featurized
                        weight = np.zeros((self.action_n, n_out_features))
                        result = learning_agent(weight, num_e=self.num_e, alpha=alpha, epsilon=epsilon,decay=decay)
                        trails.append(result)
                        if i % 20 == 0:
                            print("Step: ", i)
                    print("Done with ",(alpha, epsilon, self.degree))
                    name = path + self.env_name + "/" + self.policy_type + "/" + self.expansion + "/" + agent_name + "/" + str(
                        alpha) + "_" + str(epsilon)+"_"+str(self.degree)+more
                    name = name.replace(".", "")
                    pickle.dump(trails, open(name, "wb"))
                    plot_trails(trails, name)
                    a.update({(alpha, epsilon, self.degree): result})
        return a