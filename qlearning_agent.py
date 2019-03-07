import agent
import itertools
import Util
import pickle
import numpy as np
import operator
from collections import defaultdict as dict
import policy
import matplotlib.pyplot as plt
from plot_3_pics_with_ylim import plot_trails



class qlearning_agent(agent.evaluate_agent):
    def __init__(self, env, state_type="continue",policy_name = "epsilon_greedy",degree=3,expansion="f",trail=50,num_e=100):
        agent.evaluate_agent.__init__(self,env,state_type=state_type,policy_name=policy_name,degree=degree,
                                      expansion=expansion,trail_n=trail,num_e=num_e)
        self.agent_name = "q"

    def run(self, alpha=0.005, discount_factor = 1, lambda2 = 0, epsilon= 0.05, decay = False):

        results = []
        normalized = self.normalized
        featurized = self.featurized

        #todo: didn't write "decay" case
        env = self.env
        policy = self.policy

        critic = Util.Linear_Approximator(normalized, featurized,lambda2=lambda2,
                                          n_feature=self.n_featurized, alpha=alpha, action_n=self.action_n)
        critic.initial()


        policy = policy(critic.perdict, epsilon, self.action_n, ifcontinue=True)
        for i_e in range(self.num_e):
            state = env.reset()
            state2 = normalized(state)
            phi_state = featurized(state2)
            returns = 0

            for turn in itertools.count():
                action_p = policy(phi_state)
                # Choose the action
                action_index = np.random.choice(range(self.action_n), p=action_p)
                action = self.action_space[action_index]

                next_state, reward, done = env.step(action)

                #Caculate Return
                returns = returns + discount_factor ** turn * reward
                next_state2 = normalized(next_state)
                phi_next_state = featurized(next_state2)

                if done:
                    target = reward + discount_factor * 0
                    critic.update(phi_state, target, action_index)
                    results.append(returns)
                    #print("done with ", i_e, "epsiode: ", turn)
                    break

                if turn == 1000:
                    results.append(returns)
                    break


                # Function approximation TD_update
                q_next_state = critic.perdict(phi_next_state, a=-1)
                target = reward + discount_factor * max(q_next_state)
                q_phi_sate =  critic.perdict(phi_state,action_index)
                td_error = target - q_phi_sate
                critic.update(phi_state, target, action_index)

                state = next_state
                phi_state = phi_next_state

        return results

    def runTrails(self, alphas=[0.001,0.002,0.003,0.008],epsilons=[0.008,0.01, 0.05], lambda2s=[0, 0.3,0.5,0.8] ,path="Result/"):
        log = "alpha    epsilon     lambuda       degree    mean_return    max_return     last_return\n"
        dict = {}

        for alpha in alphas:
            for epsilon in epsilons:
                for lambda2 in lambda2s:
                    trails = []
                    for i in range(self.trail_n):
                        if self.state_type == "continue":
                            result = self.run(alpha=alpha, epsilon=epsilon, lambda2=lambda2)
                        else:
                            result = self.run_tabular(alpha=alpha, epsilon=epsilon, lambda2=lambda2)
                            print("done one",i)
                        trails.append(result)
                    path_t = path + self.env_name + "/" + self.agent_name + "/"
                    name = path_t + str(alpha) + "_" + str(epsilon) + "_" +str(lambda2) +"_"+str(
                        self.degree)
                    name = name.replace(".", "")
                    pickle.dump(trails, open(name, "wb"))
                    plot_trails(trails, name)
                    trails = np.array(trails)
                    trails = trails[:, 90:]
                    dict.update({str(alpha) + "_" + str(epsilon) + "_" + str(lambda2) + "_" + str(self.degree): np.mean(trails)})
                    print("done one trails", np.mean(trails))
        sorted_J = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_J[0])
        f = open(path_t+'log.txt', 'a')
        f.write(str(sorted_J))
        f.write("\n")
        f.close()

    def run_tabular(self, discount_factor=0.9, alpha=0.05, lambda2=0.5, epsilon=0.05, decay=True):
        # regular q_learning update
        env = self.env
        policy = self.policy
        result = []

        q_table = np.zeros((self.state_n, self.action_n))
        e_table = np.zeros((self.state_n, self.action_n))
        policy = policy(q_table, epsilon, self.action_n, ifcontinue=False)

        for i_episode in range(self.num_e):
            returns = 0.000
            turn = 0
            state = env.reset()
            not_done = True

            if decay and i_episode >= 80:
                epsilon = 1.0 / (i_episode + 1) ** 2
                epsilon = 0
                policy = self.policy(q_table, epsilon, self.action_n, ifcontinue=False)
                decay = False
            for turn in itertools.count():

                # Get the action properties for each one

                x, y = state
                state2 = env.state_transform(x, y)

                action_p = policy(state2)
                # Choose the action
                action_index = np.random.choice(range(self.action_n), p=action_p)
                action = self.action_space[action_index]

                # Take one step based on the action
                next_state, reward, done = env.step(action)

                returns = returns + discount_factor ** turn * reward
                x, y = next_state
                next_state2 = env.state_transform(x, y)
                # Update the q_function
                target = reward + discount_factor * np.max(q_table[next_state2])
                td_error = target - q_table[state2][action_index]
                #q_table[state2][action_index] += alpha * td_error

                #Update e
                e_table[state2][action_index] = e_table[state2][action_index] + 1
                q_table += alpha * e_table * td_error
                e_table *= discount_factor * lambda2

                if done:
                    result.append(returns)
                    break

                state = next_state
        return result


def run_moutain_car():
    from World import moutain_car as world
    env = world.moutain_car()
    agent1 = qlearning_agent(env)
    agent1.runTrails(alphas=[0.008], epsilons=[0.01, 0.05, 0.2], lambda2s=[0.05,0.1,0.2])
def run_grid_world():
    from World import mdp_Gridworld as world
    env = world.grid()
    env = world.mdp_Gridworld(env)
    agent1 = qlearning_agent(env, state_type="Tabular")
    agent1.runTrails(alphas=[0.16], epsilons=[0.6], lambda2s=[0.47, 0.1])


if __name__ == '__main__':
    run_grid_world()
    #run_moutain_car()

