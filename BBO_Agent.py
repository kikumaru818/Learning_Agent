import numpy as np
import operator
import Gridworld.mdp_Gridworld as grid
import Gridworld.cart_pole as cartpole
import matplotlib.pyplot as plt
import time


class cross_entropy:
    n_state = 23
    n_action = 4

    def __init__(self, env=0, n_state=23, n_action=4,sigma=0.9,K=20,K_e=4,
                 n_samples=10,epsilon=0.45, world_name="grid"):

        self.n = n_state * n_action
        n=self.n
        self.n_action=n_action
        self.n_state=n_state
        self.env = env

        if world_name == "cartpole":
            # if not featulized, the observing state space is just 4 for cartpole
            self.n = 4
        self.mean=[0]*self.n
        self.cov = np.identity(self.n)
        self.signma = sigma
        self.K=K
        self.K_e=K_e
        self.n_samples=n_samples
        self.epsilon=epsilon
        self.world_name = world_name

    def transfer(self, vector, n_state, n_action):

        shape = (n_state, n_action)
        vector.reshape(shape)
        return vector

    #for first choice hill climbing
    def run_trail_fchc(self,limit=150):

        max_iteration=limit
        not_done = True
        mean = self.mean
        cov = self.signma*np.identity(self.n)
        K = self.K
        N = self.n_samples
        K_e = self.K_e

        result = []
        result_theta = []

        index = 0
        theta_max = np.array(mean)



        if self.world_name == "grid":
            shape = (self.n_state, self.n_action)
            theta_max = theta_max.reshape(shape)
        self.theta = theta_max
        J_max, J_all = self.evaluate(theta_max, N)
        J_max_all = J_all


        while not_done:
            theta_f = np.random.multivariate_normal(mean, cov, 1)
            theta = theta_f[0]
            if self.world_name == "grid":
                shape = (self.n_state, self.n_action)
                theta = theta.reshape(shape)

            self.theta = theta
            J, J_all = self.evaluate(theta, N)

            if J > J_max:
                mean = theta_f[0]
                print(mean)
                theta_max = theta
                J_max = J
                J_max_all = J_all

            result = result + J_max_all
            result_theta.append(theta_max)

            index +=1

            if index % 20 == 0:
                print("index:", index)
            if index == max_iteration:
                not_done = False

        return result, J_max, theta_max


    def run_trail_ce(self,limit= 150):

        J_max = 0
        theta_max = 0

        max_iteration=limit
        not_done = True
        mean = self.mean
        cov = self.cov
        K = self.K
        N = self.n_samples
        K_e = self.K_e

        result = []
        result_theta = []

        index = 0
        while not_done:

            dic = {}

            theta_all = np.random.multivariate_normal(mean, cov, K)
            for i in range(self.K):
                theta = theta_all[i]
                # update current theta

                if self.world_name == "grid":
                    shape = (self.n_state, self.n_action)
                    theta = theta.reshape(shape)

                self.theta = theta
                # self.prepare_for_current_policy()
                J_i, J_i_all = self.evaluate(theta,N)



                if (J_max < J_i):
                    theta_max = theta
                    J_max = J_i

                #since cartpole is determinist transitiion funciton
                #todo: hardcod, delete later
                #if (J_max > 1009):
                #    not_done = False


                """""
                if (J_max > 3):
                    self.K = 10
                    self.K_e= 2
                    self.epsilon = 0.2
                if (J_max > 4):
                    self.K_e = 1
                    self.epsilon = 0.05
                """

                dic.update({i: J_i})


                #J_i_all.sort()

                result = result + J_i_all
                result_theta.append(mean)


            sorted_J = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
            sorted_J = sorted_J[:K_e]
            theta_e = [theta_all[i[0]] for i in sorted_J]
            theta_e = np.array(theta_e)

            # theta = sum(theta_e)/K_e


            mean = np.mean(theta_e, axis=0)
            cov = np.cov(theta_e.T)+self.epsilon*np.eye(self.n,self.n)

            ave_J = np.average([i[1] for i in sorted_J])  # Todo sum over the keys
            #result.append(ave_J)
            #result_theta.append(mean)

            index +=1

            if index % 20 == 0:
                print("index:", index)
            if index == max_iteration:
                not_done = False

        return result, J_max, theta_max

    def evaluate(self, theta, N):

        if self.world_name == "grid":
            self.prepare_policy()
            self.theta = theta
            J_mean = grid.runSimulation(num_e=N, p='function', p_function=self.policy)
            return J_mean
        elif self.world_name == "cartpole":
            # todo: for test
            J_mean = cartpole.runSimulation(num_e=N, p='action', function=self.side_policy)
            return J_mean

    def prepare_policy(self):
        theta=self.theta
        exp_form = np.exp(theta * self.signma)
        sum_over_action = np.sum(exp_form, axis=1)
        value = exp_form / sum_over_action[:, None]
        self.policy_all = value
        return value

    # return a probablity for all action
    def policy(self,state):
        # 1.get percentage for each action
        actions = self.policy_all[state]
        return actions

    # directly return a action index
    def side_policy(self, state):
        theta = self.theta

        # TODO : delete later
        #theta = [-0.15104537, 2   , -1.84203697, 2.01880024]
        temp = np.matmul(theta,state)
        action = 0 if np.matmul(theta, state) < 0 else 1
        #FOR test, modify later
        #return 0
        return action


def homework2_q1():
    agent = cross_entropy(K=20, K_e=4, epsilon=0.5)
    result, J_max, theta_max = agent.run_trail_ce()
    """""
    x = [i for i in range(len(result))]
    plt.plot(x, result)
    plt.title("grid world cross_entropy")
    plt.show()
    """
    return result, J_max, theta_max


def homework2_q2():
    agent = cross_entropy(world_name="cartpole",K_e=4, epsilon=0.5,n_samples=1)
    result, J_max, theta_max = agent.run_trail_ce(limit=100)
    #x = [i for i in range(len(result))]
    #plt.plot(x, result)
    #plt.title("cartpole cross_entropy")
    #plt.show()
    return result, J_max, theta_max

def homework2_q3():
    agent = cross_entropy(K=20,K_e = 4, sigma=0.45, n_samples=50)
    result, J_max, theta_max= agent.run_trail_fchc(limit=300)
    #x = [i for i in range(len(result))]
    #print("J_max is : ", result[-1])
    #plt.plot(x, result)
    #plt.title("grid world fchc")
    #plt.show()
    return result, J_max, theta_max

def homework2_q4():
    agent = cross_entropy(world_name="cartpole", K_e=2, sigma=1.2,n_samples=1)
    result, J_max, theta_max= agent.run_trail_fchc(limit=100)
    #x = [i for i in range(len(result))]
    #print("J_max is : ", result[-1])
    #plt.plot(x, result)
    #plt.title("cartpole fchc")
    #plt.show()
    return result, J_max, theta_max

def homework_trails(homework_to_run, num_trails = 500, limit = 100, output_file = "trails",
                    differentlength=False):
    import pickle
    results = []
    J_max = 0
    J_max_trail=[]
    theta_max = 0
    for i in range(num_trails):
        print(i)
        result, J, theta = homework_to_run()
        results.append(result)
        if J_max < J:
            J_max=J
            theta_max = theta
        J_max_trail.append(J)

        if i % 50 == 0:
            pickle.dump(results, open(output_file + "_result.pkl", "wb"))
            pickle.dump((J_max,theta_max,J_max_trail), open(output_file + "_theta.pkl", "wb"))


    if (differentlength):
        list_length = [len(i) for i in results]
        list_max = max(list_length)
        new_result = [results[i] + [J_max[i]] * (list_max - len(results[i])) for i in range(len(results))]

    pickle.dump(results,open(output_file+"_result.pkl", "wb"))
    pickle.dump((J_max,theta_max,J_max_trail),open(output_file+"_theta.pkl", "wb"))
    print("Done\nJ_Max", J_max, "theta_max", theta_max)


#time.time()
homework2_q3()