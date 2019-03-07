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



class REINFORCE_agent(agent.evaluate_agent):
    def __init__(self, env, state_type="continue",degree=5,expansion="f",trail=50,num_e=100,policy_name="softmax"):
        agent.evaluate_agent.__init__(self,env,state_type=state_type,degree=degree,
                                      expansion=expansion,trail_n=trail,num_e=num_e,policy_name=policy_name)
        self.agent_name = "REINFORCE"

    def run(self, alpha=0.005,beta=0.001, discount_factor = 1, lambda2 = 0.5):
        #Note: normally, alpha should be larger than beta

        env = self.env
        num_e = self.num_e
        action_space = self.action_space



        results = []
        normalized = self.normalized
        featurized = self.featurized

        #Use q-esitmate instead of value function
        critic = Util.Linear_Approximator(normalized, featurized,
                                               n_feature=self.n_featurized, alpha=alpha, action_n = self.action_n)
        #use value esitmate
        critic = Util.Linear_Approximator(normalized, featurized, lambda2=lambda2,
                                          n_feature=self.n_featurized, alpha=alpha, action_n=1)

        actor = Util.Softmax_Approximator(normalized, featurized, lambda2= lambda2,
                                                     n_feature=self.n_featurized, alpha=beta, action_n = self.action_n)

        critic.initial()
        actor.initial()
        for i_e in range(num_e):
            state = env.reset()
            returns = 0
            episodes = []
            #todo: a ignore gamma version
            for turn in itertools.count():
                phi_state = actor.to_phi(state)
                action_index = actor.sample(phi_state)
                action = action_space[action_index]
                next_state, reward, done = env.step(action)
                phi_next_state = actor.to_phi(next_state)
                next_action_index = actor.sample(phi_next_state)
                next_action = action_space[next_action_index]
                returns = returns + discount_factor ** turn * reward
                episodes.append((phi_state,action_index,reward))
                action = next_action
                action_index = next_action_index
                state = next_state
                phi_state = phi_next_state

                if turn == 5000:
                    results.append(returns)
                    break
                if done:
                    results.append(returns)
                    break


            #print("fist action choose: ", action_index)
            last_reward = 0
            for i in range(len(episodes)):
                phi_state,action_index,reward = episodes[i]
                q_state = critic.perdict(phi_state, action_index)


                if i+1 < len(episodes):
                    phi_next_state,next_action_index,_= episodes[i+1]
                    q_next_state = critic.perdict(phi_next_state, next_action_index)
                else:
                    q_next_state = 0
                #Update critic
                target = reward + discount_factor * q_next_state
                td_error = target - q_state

                if i == 0:
                    update = returns
                else:
                    update = (update - last_reward)/discount_factor
                last_reward = reward
                #critic.update(phi_state, target, action_index)
                actor.update(phi_state, action_index, update)

                last_reward = reward



        name="reinforce_moutaincar"
        return results

    def run_Tabular(self, alpha=0.005, beta=0.0001, discount_factor=0.9, decay=True):
        # regular q_learning update
        env = self.env
        policy = self.policy
        result = []

        v_table = np.zeros((self.state_n, 1))  # critic
        p_table = np.zeros((self.state_n, self.action_n))  # actor
        policy = policy(p_table, 1, self.action_n, ifcontinue=False)



        for i_episode in range(self.num_e):
            returns = 0.000
            state = env.reset()
            not_done = True

            if decay and i_episode >= 80:
                epsilon = 1.0 / (i_episode + 1) ** 2
                epsilon = 0
                policy = self.policy(p_table, 0, self.action_n, ifcontinue=False)
                decay = False

            episodes = []
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

                episodes.append((state2, action_index, reward))

                if done:
                    #print("one episode", returns)
                    result.append(returns)
                    break

                state = next_state

            last_reward = 0
            update = 0

            for i in range(len(episodes)):

                # Get the action properties for each one
                state2, action_index, reward = episodes[i]

                if i + 1 < len(episodes):
                    next_state2, next_action_index, _ = episodes[i + 1]
                    v_next_state = v_table[next_state2]

                else:
                    v_next_state = 0
                # Update the q_function
                #target = reward + discount_factor * v_next_state
                #td_error = target - v_table[state2]
                if i == 0:
                    update = returns
                else:
                    update = (update - last_reward)/discount_factor - v_table[state2]
                    #update = (update - discount_factor**(turn-1)*last_reward) - v_table[state2]
                # Update
                p_table[state2][action_index] += alpha * update
                #v_table[state2] += beta * td_error
                last_reward = reward


        return result

    def runTrails(self, alphas=[0.001,0.005,0.00592], betas=[0.0003, 0.0005,0.00798], lambdas=[0,0.2,0.4,0.6], degree=[3], path="Result/"):
        print("actor_critic(lambda)\n")
        log = "alpha    beta     lambda2     degree    mean_return    max_return     last_return\n"
        dict = {}

        for alpha in alphas:
            for beta in betas:
                for lambda2 in lambdas:
                    trails = []
                    for i in range(self.trail_n):
                        if self.state_type == "continue":
                            result = self.run(alpha=alpha, beta=beta, lambda2=lambda2)
                        else:
                            result = self.run_Tabular(alpha=alpha, beta=beta)
                        trails.append(result)
                    path_temp = path + self.env_name + "/" + self.agent_name + "/"
                    name = path_temp + str(alpha) + "_" + str(
                        beta) + "_" + str(lambda2)+"_"+\
                           str(self.degree)
                    name = name.replace(".", "")
                    pickle.dump(trails, open(name, "wb"))
                    plot_trails(trails, name)
                    trails = np.array(trails)
                    trails = trails[:, 90:]
                    dict.update({str(alpha) + "_" + str(beta) + "_" + str(lambda2)+"_"+str(self.degree):np.mean(trails)})
                    print("done one trails", np.mean(trails))

        sorted_J = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
        print(str(sorted_J[0]))
        f = open(path_temp+'log.txt', 'a')
        f.write(str(sorted_J))
        f.write("\n")
        f.close()


def run_moutain_car():
    from World import moutain_car as world
    from World import cart_pole as world
    env = world.cart_pole()
    #env = world.moutain_car()
    agent1 = REINFORCE_agent(env,num_e=1000)
    agent1.runTrails(alphas=[5e-07], betas=[4.0e-07,5.0e-07,6.0e-07], lambdas=[0])


def run_grid_world():
    from World import mdp_Gridworld as world
    env = world.grid()
    env = world.mdp_Gridworld(env)
    agent1 = REINFORCE_agent(env, state_type="Tabular", policy_name="softmax")
    agent1.runTrails(alphas=[0.01,0.02], betas=[0.001], lambdas=[0])



if __name__ == '__main__':
    #run_grid_world()
    run_moutain_car()