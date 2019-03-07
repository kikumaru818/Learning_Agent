import agent
import itertools
import Util
import pickle
import numpy as np
from collections import defaultdict as dict
import policy
import matplotlib.pyplot as plt
from plot_3_pics_with_ylim import plot_trails
import operator
import math
from scipy import stats
import matplotlib.pyplot as plt

#currently, only have tubalur version
#todo: update to continue version
#todo: returns normalization
class seldonian_agent(agent.evaluate_agent):

    def __init__(self, env, state_type="continue",policy_name = "extract_policy",degree=3,expansion="f",trail=100,num_e=2000, delta= 0.05):
        agent.evaluate_agent.__init__(self,env,state_type=state_type,policy_name=policy_name,degree=degree,
                                      expansion=expansion,trail_n=trail,num_e=num_e)
        self.agent_name = "seldonian"
        self.new_policy = getattr(policy, "random_search")
        self.delta = delta
        self.RATIO = 5

        if state_type=="Tabular":
            #self.policy_table = s=np.random.dirichlet(np.ones(self.action_n),size=self.state_n)
            self.initial_policy()
            self.policy_table = self.prepare_policy()



    def run_optimizae(self, iteration=2000):
        returns = []
        for i in range(iteration):
            new_policy, online_return = self.run_one_tabular()
            print(new_policy, "online return: ", online_return)
            if new_policy =="NSF":
                pass
            else:
                self.policy_table = new_policy
                self.mean = self.theta
                returns.append(online_return)
        plt.plot(range(len(returns)),returns)
        plt.show()


    def run_one_tabular(self,discount_factor = 0.9):
        # regular q_learning update
        env = self.env
        policy = self.policy
        D = []
        Return = []
        Trajectory = []

        q_table = self.policy_table
        policy = policy(q_table)

        for i_episode in range(self.num_e):
            returns = 0.000
            state = env.reset()
            trajectory = []
            for turn in itertools.count():

                # Get the action properties for each one

                x, y = state
                state2 = env.state_transform(x, y)

                action_p = policy(state2)
                # Choose the action
                action_index = 1
                try:
                    action_index = np.random.choice(range(self.action_n), p=action_p)

                except:
                    print("error")

                action = self.action_space[action_index]
                trajectory.append([state2, action_index])

                # Take one step based on the action
                next_state, reward, done = env.step(action)

                returns = returns + discount_factor ** turn * reward
                x, y = next_state
                next_state2 = env.state_transform(x, y)

                if done:
                    Return.append(returns)
                    Trajectory.append(trajectory)
                    #result.append([trajectory, returns])
                    break

                state = next_state

        #split result into training and testing
        n_D = len(Trajectory)

        testing_index = int(n_D/self.RATIO)
        training_r = Return[:testing_index]
        testing_r = Return[testing_index:]
        training = Trajectory[:testing_index]
        testing = Trajectory[testing_index:]


        D_training = [training,training_r]
        D_testing = [testing,testing_r]


        best_safty_policy = self.quasi_seldonian(0.5, D_training, D_testing, len(D_testing[0]), self.delta)
        return best_safty_policy, np.mean(Return)


    def daedalus(self, pi_0, delta, beta):
        pass


    #Just support one delta with one g so far
    #pi_b: current behvaior policy vector
    #f estimation of return function
    #g: unbiased estimate function of a behavior constraint
    def quasi_seldonian(self, alpha, D1, D2, m, delta):
        old_policy = self.policy_table

        #Debug
        #old_policy = self.test_policy()


        safty_new_policy= None
        generator = self.new_policy(old_policy, alpha = alpha)
        _, returns = D1
        max_f = - math.inf
        old_performance = np.mean(returns)
        for i in range(300):
            new_policy = generator()
            new_policy = self.prepare_policy()
            #new_policy = self.test_policy()


            performances = self._f(new_policy, old_policy,D1)

            performance = np.array(list(map(lambda x: np.mean(x), performances)))

            returns = np.array(returns)

            g_std = np.std(returns - performance)
            g_mean = np.mean(returns - performance)
            f_mean = np.mean(performance)
            old_mean = np.mean(returns)

            #g: behavior constraint
            temp = stats.t.ppf(1-delta, m-1)
            ttest = g_mean + 2*g_std/math.sqrt(m) * stats.t.ppf(1-delta, m-1)
            if ttest <= 0 and max_f < f_mean:
                max_f = f_mean
                safty_new_policy = new_policy
        if safty_new_policy is None:
            return "NSF"
        else:
            _, returns = D2
            performances = self._f(safty_new_policy, old_policy, D2)
            performance = np.array(list(map(lambda x: np.mean(x), performances)))

            old_performance = np.mean(returns)

            g_std = np.std(returns - performance)
            g_mean = np.mean(returns - performance)
            f_mean = np.mean(performance)
            old_mean = np.mean(returns)

            ttest = g_mean + g_std / math.sqrt(m) * stats.t.ppf(1 - delta, m - 1)
            if ttest <= 0:
                return safty_new_policy

            return "NSF"


    def _f(self, new_policy, old_policy, D):
        trajectories, returns = D
        estimate_f = np.array(list(map(lambda x: self._importance_weights(new_policy,old_policy,x), trajectories)))
        estimate_f = np.array(list(map(lambda x: np.prod(x), estimate_f)))

        estimate_f = np.array(returns) * estimate_f
        return estimate_f


    #95% chance increase the performance of policy
    def _g_1 (self, p_new_policy, p_old_policy):
        return p_old_policy -p_new_policy


    def initial_policy(self, env=0, n_state=23, n_action=4,sigma=1.2,K=20,K_e=4,
                 n_samples=10):

        self.n = n_state * n_action
        self.n_state = n_state
        n=self.n
        self.n_action=n_action
        self.mean = [0] * self.n
        self.cov = np.identity(self.n)
        self.signma = sigma
        self.K = K
        self.K_e = K_e
        self.n_samples = n_samples



    def prepare_policy(self):
        mean = self.mean
        cov = self.signma * np.identity(self.n)
        K = self.K
        N = self.n_samples
        K_e = self.K_e
        theta_f = np.random.multivariate_normal(mean, cov, 1)
        theta = theta_f[0]
        self.theta = theta
        shape = (self.n_state, self.n_action)
        theta = theta.reshape(shape)
        exp_form = np.exp(theta * self.signma)
        sum_over_action = np.sum(exp_form, axis=1)
        value = exp_form / sum_over_action[:, None]

        return value


    def _importance_weights(self, new_policy, old_policy, trajectory):
        new = np.array(list(map(lambda x: new_policy[x[0]][x[1]], trajectory)))
        old = np.array(list(map(lambda x: old_policy[x[0]][x[1]], trajectory)))

        temp = new/old
        return new/old

    def test_policy(self):
        theta= [1.68250619, - 0.17782381,  0.58296881 , 2.8253172 ,  1.11681143 , 1.21129844,
         -2.09606219, -1.42363356,  3.72976143, - 4.70337323,  1.27216942,  2.36557276,
         0.50897341, - 2.30729548 ,- 1.0812306 ,  1.81008366 ,- 0.85741563 , 2.31909211,
         - 0.91467391 , 2.67646596 , 0.89221976, - 0.04062639 , 0.37559366 , 4.56160681,
         2.41998529, - 1.65650924 ,- 1.60941672 , 4.23234849 ,- 2.66287122 ,- 3.48256955,
         - 1.83369577, - 1.33109421, - 2.14707122,  3.21361446, - 4.10074454 , 0.55456706,
         - 3.38428997 ,- 0.21479968, - 5.52735515 ,- 0.33872522 ,- 1.68687079, - 2.74029164,
         3.34053128 ,- 0.96526429 ,- 1.72383424, - 3.86394183, - 2.2521756 ,  1.55762722,
         - 2.51769639 ,- 3.61926907 , 1.93024343 ,- 1.26328015 , 1.00928965 , 2.40611751,
         - 3.56613514 , 3.70564599, - 0.34255832, - 3.22030482,  0.66936123 , 0.96819657,
         8.01323608,  0.11897321 , 4.02026626, - 3.14133514, - 4.98964058 ,- 1.83670731,
         0.91995877 , 1.08620432,  0.25224948,  3.46322219, - 2.55766335 , 0.56529435,
         - 2.04480628 , 2.94587789 , 0.22917704 ,- 2.11408426 ,- 4.22347794 , 1.86729422,
         2.64993353 ,- 0.77659118,  4.18933582, - 2.39546996, - 2.4956906 ,- 1.45259717,
         - 2.57629958 ,- 0.46044999,  1.28441261,  0.46274241, - 2.9566883 ,- 0.9261662,
         - 1.86816641, - 1.62950904]
        shape = (self.n_state, self.n_action)
        theta = np.array(theta)
        theta = theta.reshape(shape)
        exp_form = np.exp(theta * self.signma)
        sum_over_action = np.sum(exp_form, axis=1)
        value = exp_form / sum_over_action[:, None]
        return value

def run_grid_world():
    from World import mdp_Gridworld as world
    env = world.grid()
    env = world.mdp_Gridworld(env)
    agent1 = seldonian_agent(env, state_type="Tabular")
    agent1.initial_policy()
    agent1.run_optimizae()

run_grid_world()