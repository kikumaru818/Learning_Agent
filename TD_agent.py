import numpy as np
from collections import defaultdict as dict
import Util
import policy
import pickle
import matplotlib.pyplot as plt
from plot_3_pics_with_ylim import plot_trails
import itertools


#path = "Result/Homework4/"


class evaluate_agent:

    def __init__(self, env, policy_name="random_action", gym=False, state_type = "Tabular",state_n=23, degree = 3, expansion='f',trail_n=1, num_e=100):
        self.num_e = num_e
        self.trail_n = trail_n
        self.degree=degree


        self.env = env
        self.env_name = env.name
        #self.policy = policy
        self.policy_type = policy_name
        self.policy =getattr(policy, policy_name)


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
            if expansion == "rbf":
                #bound = env.bound
                #f = Util.featurization(bound, env.normalize)
                #self.featurized = f.transform
                #self.n_featurized = f.n_feature
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


    # tabular TD
    def td_update_tabular(self, v_table, num_e=100, discount_factor=0.9, alpha=0.05, mode="training", state_trans = False):
        env = self.env
        policy = self.policy
        #v_table = np.zeros(self.state_n)
        td_errors = []
        for i_episode in range(num_e):
            state = env.reset()
            not_done = True

            while not_done:

                action = policy(self.action_space, state, v_table)

                # Take one step based on the action choosing:
                next_state, reward, done = env.step(action)

                if not state_trans:
                    x, y = next_state
                    next_state2 = env.state_transform(x, y)
                    x, y = state
                    state2 = env.state_transform(x, y)

                # TD update
                target = reward + discount_factor * v_table[next_state2]

                if reward >1 or reward < 0:
                    nihao = 1

                #print ("reward", reward, "state: ",  state, "next state", next_state)
                td_error = target - v_table[state2]

                if mode == "training":
                     v_table[state2] += alpha * td_error
                     #print("state v", v_table[state2])
                elif mode == "testing":
                    td_errors.append(td_error)
                state = next_state


                if done:
                    break
        result = 0
        if mode == "training":
            result = v_table
        elif mode == "testing":
            result = td_errors

        return result

    # continue-space TD
    def td_update_continue(self, weight, num_e=100, discount_factor =1, alpha=0.05, mode="training", degree=3):
        env = self.env
        policy = self.policy
        a_n = self.action_n

        #n_out_features = Util.get_n_features(degree)
        # weight = np.zeros(n_out_features)

        v_w = lambda x: weight.dot(x)
        dv_w = lambda x: x
        #featurized = lambda x: Util.Fourier_Kernel(x, degree)
        featurized = self.featurized


        td_errors = []

        for i_episode in range(num_e):
            state = env.reset()
            state2 = env.normalize(state)
            phi_state = featurized([state2])
            phi_state = phi_state[0]
            not_done = True

            while not_done:
                action = policy(self.action_space, phi_state, v_w)
                next_state, reward, done = env.step(action)


                next_state2 = env.normalize(next_state)
                #phi_next_state = featurized([next_state2])
                phi_next_state = featurized(next_state2)
                phi_next_state = phi_next_state[0]


                # Function approximation TD_update
                try:
                    v_phi_next_state = v_w(phi_next_state)
                    target = reward + discount_factor * v_w(phi_next_state)
                except:
                    print("error")
                v_phi_sate = v_w(phi_state)
                td_error = target - v_w(phi_state)

                if mode == "training":
                    weight += alpha * td_error * dv_w(phi_state)
                    try:
                        a = weight.dot(phi_state)
                    except:
                        print  ("error")

                    v_w = lambda x: weight.dot(x)
                    dv_w = lambda x: x

                elif mode == "testing":
                    td_errors.append(td_error)

                state = next_state
                phi_state = phi_next_state
                if done:
                    break

        result = 0

        if mode == "training":
            result = weight
        elif mode == "testing":
            result = td_errors
        return result

    #  Tubular q(lambda=0)
    def q_learning_tabular(self, num_e=100, discount_factor=0.9, alpha=0.05, lambda2=0, epsilon=0.05, state_trans= False,degree=3,decay=True):
        # regular q_learning update
        env = self.env
        policy = self.policy
        result = []



        if lambda2 <= 0:
            q_table = np.zeros((self.state_n, self.action_n))
            policy = policy(q_table,epsilon, self.action_n, ifcontinue=False)

            for i_episode in range(num_e):
                returns = 0.000
                turn = 0
                state = env.reset()
                not_done = True

                if decay and i_episode >= 80:
                    epsilon = 1.0 / (i_episode + 1)**2
                    epsilon = 0
                    policy = self.policy(q_table, epsilon, self.action_n, ifcontinue=False)
                    decay = False
                while not_done:

                    # Get the action properties for each one

                    x,y=state
                    state2 = env.state_transform(x,y)



                    action_p = policy(state2)
                    # Choose the action
                    action_index = np.random.choice(range(self.action_n), p=action_p)
                    action = self.action_space[action_index]

                    #action = policy(self.action_space, state, q_table)


                    # Take one step based on the action
                    next_state, reward, done = env.step(action)

                    returns = returns + discount_factor ** turn * reward


                    if not state_trans:
                        x, y = next_state
                        next_state2 = env.state_transform(x, y)

                    # Update the q_function
                    target = reward + discount_factor*np.max(q_table[next_state2])
                    td_error = target - q_table[state2][action_index]
                    q_table[state2][action_index] += alpha*td_error

                    if done:
                        result.append(returns)
                        break

                    state = next_state
                    turn += 1

        elif lambda2 > 0:
            q_table = np.zeros((self.state_n, self.action_space.n))
            e_table = np.zeros((self.state_n, self.action_space.n))
            policy = policy(q_table, epsilon, self.action_space.n)
            # returns = np.zeros(num_e)

            for i_episode in range(num_e):
                state = env.reset()
                not_done = True
                while not_done:

                    # Get the action properties for each one
                    action_p = policy(state)
                    # Choose the action
                    action_index = np.random.choice(self.action_space, p=action_p)
                    action = self.action_space[action_index]

                    # Take one step based on the action
                    next_state, reward, done = env.step(action)


                    if not state_trans:
                        x, y = next_state
                        next_state2 = env.state_transform(x, y)
                        x, y = state
                        state2 = env.state_transform(x, y)

                    # Update the q_function
                    target = reward + discount_factor*np.max(q_table[next_state2])
                    td_error = target - q_table[state2][action_index]
                    q_table[state2][action_index] += alpha*td_error

                    # update e:
                    e_table[state2][action_index] = e_table[state2][action_index]+1
                    q_table += alpha*e_table*td_error
                    e_table *= discount_factor * lambda2 * e_table

                    if done:
                        result.append(returns)

                    state = next_state
        return result

    def sarsa_tabular(self, num_e=100, discount_factor=0.9, alpha=0.05, lambda2=0, epsilon=0.05,
                           state_trans=False,decay=True):
        # regular q_learning update
        env = self.env
        policy = self.policy
        result = []

        if lambda2 <= 0:
            q_table = np.zeros((self.state_n, self.action_n))
            policy = policy(q_table, epsilon, self.action_n, ifcontinue=False)

            for i_episode in range(num_e):
                returns = 0.000
                turn = 0
                state = env.reset()
                not_done = True

                if decay and i_episode >= 80:
                    epsilon = 0
                    policy = self.policy(q_table, epsilon, self.action_n, ifcontinue=False)
                    decay = False

                x, y = state
                state2 = env.state_transform(x, y)
                action_p = policy(state2)
                action_index = np.random.choice(range(self.action_n), p=action_p)
                action = self.action_space[action_index]


                while not_done:
                    next_state, reward, done = env.step(action)
                    returns = returns + discount_factor ** turn * reward


                    if not state_trans:
                        x, y = next_state
                        next_state2 = env.state_transform(x, y)
                        x, y = state
                        state2 = env.state_transform(x, y)

                    #choose next action first
                    action_p = policy(state2)
                    next_action_index = np.random.choice(range(self.action_n), p=action_p)
                    next_action = self.action_space[next_action_index]

                    if done:
                        target = reward + discount_factor * 0
                        td_error = target - q_table[state2][action_index]
                        q_table[state2][action_index] += alpha * td_error
                        result.append(returns)
                        break

                    # Update the q_function
                    target = reward + discount_factor * q_table[next_state2][next_action_index]
                    td_error = target - q_table[state2][action_index]
                    q_table[state2][action_index] += alpha * td_error



                    state = next_state
                    action_index = next_action_index
                    action = next_action
                    turn += 1
        return result

    def q_learning_continue(self,weight, num_e=100, discount_factor=1, alpha=0.05, lambda2=0, epsilon=0.05, mode="training", step_limit = 1008, decay=False,more=""):
        env = self.env
        policy = self.policy
        a_n = self.action_n
        td_errors=[]
        results = []
        sample_states=[]
        q_w = lambda x: weight.dot(x)
        dq_w = lambda x: x
        featurized = self.featurized



        if lambda2 <= 0:
            policy = policy(q_w, epsilon, self.action_n, ifcontinue=True)
            for i_episode in range(num_e):
                state = env.reset()
                state2 = env.normalize(state)
                phi_state = featurized(state2)
                not_done = True

                turn = 0
                returns = 0

                if decay and i_episode == 80:
                    # stop exploration

                   policy = self.policy
                   epsilon = 0
                   policy = policy(q_w, epsilon, self.action_n, ifcontinue=True)
                   decay=False



                while not_done:
                    action_p = policy(phi_state)
                    # Choose the action
                    try:
                        action_index = np.random.choice(range(self.action_n), p=action_p)
                    except:
                        print("error")
                    action = self.action_space[action_index]

                    next_state, reward, done = env.step(action)
                    sample_states.append(np.array(next_state))


                    returns = returns + discount_factor ** turn * reward
                    next_state2 = env.normalize(next_state)
                    phi_next_state = featurized(next_state2)



                    if decay and turn > step_limit:
                        #stop exploration

                        policy = self.policy
                        epsilon = 0
                        policy = policy(q_w, epsilon, self.action_n, ifcontinue=True)
                        decay = False

                    if done:
                        if turn > step_limit:
                            target = reward + discount_factor * max(q_w(phi_next_state))
                        else:
                            target = reward + discount_factor * 0
                        q_phi_sate = q_w(phi_state)[action_index]
                        td_error = target - q_phi_sate
                        temp = alpha * td_error * dq_w(phi_state)
                        weight[action_index] += temp
                        q_w = lambda x: weight.dot(x)
                        dq_w = lambda x: x
                        results.append(returns)
                        break
                    # Function approximation TD_update
                    target = 0
                    try:
                        q_phi_next_state = q_w(phi_next_state)
                        target = reward + discount_factor * max(q_phi_next_state)
                    except:
                        print("error")
                    q_phi_sate = q_w(phi_state)[action_index]
                    td_error = target - q_phi_sate

                    if mode == "training":
                        temp = alpha * td_error * dq_w(phi_state)
                        b = weight.dot(phi_state)
                        weight[action_index] += temp
                        try:
                            a = weight.dot(phi_state)
                        except:
                            print  ("error")

                        q_w = lambda x: weight.dot(x)
                        dq_w = lambda x: x

                    elif mode == "testing":
                        td_errors.append(td_error)


                    state = next_state

                    phi_state = phi_next_state
                    turn += 1

                    if done:
                        results.append(returns)
                        break


        sample_states = np.array(sample_states)
        print("q")
        print("min: ", np.min(sample_states,axis=0))
        print("max: ", np.max(sample_states, axis=0))
        print("mean: ", np.mean(sample_states, axis=0))
        print(np.mean(results),"max",np.max(results))
        return results

    def sarsa_continue(self, weight, num_e=100, discount_factor=1, alpha=0.05, lambda2=0, epsilon=0.05, mode="training", step_limit=1008, decay=False,more=""):
        env = self.env
        policy = self.policy
        a_n = self.action_n
        td_errors=[]
        results = []
        q_w = lambda x: weight.dot(x)
        dq_w = lambda x: x
        featurized = self.featurized


        if lambda2 <= 0:
            policy = policy(q_w, epsilon, self.action_n, ifcontinue=True)
            for i_episode in range(num_e):
                state = env.reset()
                state2 = env.normalize(state)
                phi_state = featurized(state2)

                not_done = True
                turn = 0
                returns = 0

                #choose the first action
                action_p = policy(phi_state)
                # Choose the action
                action_index = np.random.choice(range(self.action_n), p=action_p)
                action = self.action_space[action_index]
                if decay and i_episode == 80:
                    # stop exploration
                    policy = self.policy
                    epsilon = 0
                    policy = policy(q_w, epsilon, self.action_n, ifcontinue=True)
                    decay=False




                while not_done:

                    next_state, reward, done = env.step(action)
                    returns = returns + discount_factor ** turn * reward

                    next_state2 = env.normalize(next_state)
                    phi_next_state = featurized(next_state2)

                    # choose the first action
                    action_p = policy(phi_next_state)
                    # Choose the action

                    next_action_index = np.random.choice(range(self.action_n), p=action_p)
                    next_action = self.action_space[next_action_index]


                    if done:
                        q_phi_sate = q_w(phi_state)[action_index]
                        q_phi_next_state = q_w(phi_next_state)
                        if turn > step_limit:
                            target = reward + discount_factor * q_phi_next_state[next_action_index]
                        else:
                            target = reward + discount_factor * 0
                        td_error = target - q_phi_sate
                        temp = alpha * td_error * dq_w(phi_state)
                        weight[action_index] += temp
                        q_w = lambda x: weight.dot(x)
                        dq_w = lambda x: x

                        results.append(returns)
                        break

                    # Function approximation TD_update
                    q_phi_next_state = q_w(phi_next_state)
                    target = reward + discount_factor * q_phi_next_state[next_action_index]
                    q_phi_sate = q_w(phi_state)[action_index]
                    td_error = target - q_phi_sate

                    if mode == "training":
                        temp = alpha * td_error * dq_w(phi_state)

                        a = weight.dot(phi_state)
                        weight[action_index] += temp

                        q_w = lambda x: weight.dot(x)
                        b = q_w(phi_state)
                        dq_w = lambda x: x

                    elif mode == "testing":
                        td_errors.append(td_error)

                    state = next_state
                    phi_state = phi_next_state
                    action = next_action
                    action_index = next_action_index
                    turn += 1

        print("sarsa")
        print(np.mean(results), "max", np.max(results))


        return results






    # ======================== Method for some special tasks ============================================
    def run_trails_over_stepsize_tabular(self, alphas, training_e = 100, testing_e = 100):
        v_table = np.zeros(self.state_n)
        td_errors_all = []
        for a in alphas:
            v_table = np.zeros(self.state_n)
            v_table = self.td_update_tabular(v_table, alpha=a, num_e=training_e)
            td_errors = self.td_update_tabular(v_table, alpha=a, mode="testing",num_e=testing_e)
            td_errors_all.append(td_errors)
            print("done with alphas: ", a)
            print ("Expectation of TD_erros: ", np.mean(td_errors))
        return td_errors_all

    def run_trails_over_stepsize_continue(self, alphas, degree=3):
        v_table = np.zeros(self.state_n)
        td_errors_all = []
        n_out_features = Util.get_n_features(self.env.n_state_space, degree)
        for a in alphas:
            weight = np.zeros(n_out_features)
            weight = self.td_update_continue(weight, num_e=100, alpha=a, degree=degree)
            td_errors = self.td_update_continue(weight, alpha=a, mode="testing", degree=degree)
            td_errors_all.append(td_errors)
            print("done with alphas: ", a)
            print ("Expectation of TD_erros: ", np.mean(td_errors))
        return td_errors_all


    def grid_search_continue(self, learning_agent, agent_name = "",alphas=[0.001,0.005, 0.01, 0.05], epsilons=[0.01, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8,1], degrees=[5], num_e=100, path = "Result/",decay=True,more=""):
        a=dict()
        trails=[]

        for alpha in alphas:
            for epsilon in epsilons:
                for degree in degrees:
                    for i in range(self.trail_n):

                        #n_out_features = Util.get_n_features(self.env.n_state_space, degree)
                        n_out_features = self.n_featurized
                        weight = np.zeros((self.action_n, n_out_features))
                        #weight = np.random.rand(self.action_n,n_out_features)
                        result = learning_agent(weight, num_e=self.num_e, alpha=alpha, epsilon=epsilon,decay=decay)
                        trails.append(result)
                        if i % 20 == 0:
                            print("Step: ", i)
                    print("Done with ",(alpha, epsilon, self.degree))
                    name = path + self.env_name + "/" + self.policy_type + "/" + self.expansion + "/" + agent_name + "/" + str(
                        alpha) + "_" + str(epsilon)+"_"+str(self.degree)+more
                    name = name.replace(".", "")
                    #name = name.replace("/","")
                    pickle.dump(trails, open(name, "wb"))
                    plot_trails(trails, name)
                    a.update({(alpha, epsilon, self.degree): result})
        return a

    def grid_search_tabular(self, learning_agent, agent_name="", alphas=[0.1,0.2,0.3,0.5,0.8], epsilons=[20], num_e=100, trail = 1, path = "Result/",decay=True,more=""):
        a=dict()
        trails = []

        for alpha in alphas:
            for epsilon in epsilons:
                trails = []
                for i in range(trail):
                    q_table = np.zeros((self.action_n, self.state_n))
                    if epsilon == 0:
                        result = learning_agent(num_e=num_e, alpha=alpha, epsilon=epsilon,decay=decay)
                    else:
                        result = learning_agent(num_e=num_e, alpha=alpha, epsilon=epsilon,decay=decay)
                    trails.append(result)
                    if i % 20 == 0:
                        print("Now at: ", i)

                print("Done with ", (alpha, epsilon))
                name = path+self.env_name+"/"+self.policy_type+"/"+"/"+agent_name+"/"+str(alpha)+"_"+str(epsilon)+more
                name = name.replace(".","")
                #name = name.replace("/", "_")
                pickle.dump(trails,open(name,"wb"))
                plot_trails(trails,name)

                a.update({(alpha, epsilon): result})
        return a



    # ============= Setting Method ======================================
    def set_policy(self, policy):
        self.policy = policy

    def set_env(self, env):
        self.env = env

# ================================ Homework 3 ==========================================================================
def td_mse(td_errors):
    td_errors = np.array(td_errors)
    square = td_errors * td_errors
    mse = np.mean(square)
    return mse



def homework3_gridworld():
    from World import mdp_Gridworld as world

    alphas = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
              0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
              0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #alphas = [1.5]
    env = world.grid()
    env = world.mdp_Gridworld(env)
    #p = policy.random_action
    agent = evaluate_agent(env)
    td_errors = agent.run_trails_over_stepsize_tabular(alphas)
    y = [td_mse(i) for i in td_errors]
    z = [np.mean(i) for i in td_errors]
    x = alphas
    print(z)
    return x, y

def homework3_cartpole(degree=3):
    from World import cart_pole as world
    alphas = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,0.0009,
              0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
              0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #alphas = [0.01, 0.02]
    env = world.cart_pole()
    #p = policy.random_action
    agent = evaluate_agent(env)
    td_errors = agent.run_trails_over_stepsize_continue(alphas, degree=degree)
    y = [td_mse(i) for i in td_errors]
    z = [np.mean(i) for i in td_errors]
    x = alphas
    print(z)
    return x, y
def run_for_homeowrk3():
    x, y = homework3_gridworld()
    pickle.dump((x, y), open("grid_world_result_100", "wb"))
    # print(x,y)
    x, y = homework3_cartpole(degree=3)
    pickle.dump((x, y), open("cart_pole_result_3", "wb"))
    x, y = homework3_cartpole(degree=5)
    pickle.dump((x, y), open("cart_pole_result_5", "wb"))

#========================== End of Homework 3 ==========================================================================

#========================== Homework 4 =================================================================================
def homework4_gridworld_q_learning(policy = "epsilon_greedy", alphas=[],epsilons=[],decay=False,more=""):
    from World import mdp_Gridworld as world

    env = world.grid()
    env = world.mdp_Gridworld(env)
    agent = evaluate_agent(env, policy_name=policy)
    learning_agent = agent.q_learning_tabular
    result = agent.grid_search_tabular(learning_agent, agent_name="q",alphas=alphas,epsilons=epsilons,more=more,decay=decay)
    learning_agent = agent.sarsa_tabular
    result = agent.grid_search_tabular(learning_agent,agent_name="sarsa",alphas=alphas,epsilons=epsilons,more=more,decay=decay)


    return result




def homework4_cart_pole_q_learning(policy="epsilon_greedy",degree =5, expansion="f", alphas=[], epsilons=[],decay=False, trail_n=1,num_e=100,more="",name_e=""):
    print("============> "+name_e)
    from World import cart_pole as world
    env= world.cart_pole()
    agent = evaluate_agent(env, policy_name=policy, state_type="continue",degree=degree, expansion=expansion, trail_n=trail_n,num_e=num_e)


    learning_agent = agent.q_learning_continue
    result = agent.grid_search_continue(learning_agent, agent_name="q",alphas=alphas,epsilons=epsilons,decay=decay,more=more)


    learning_agent = agent.sarsa_continue
    result = agent.grid_search_continue(learning_agent,agent_name="sarsa",alphas=alphas,epsilons=epsilons,decay=decay,more=more)
    return result

def homeork4_moutaincart_q_learning(policy="epsilon_greedy", degree=4, expansion="f", alphas=[],epsilons=[], decay=False, trail_n=1):
    from World import moutain_car as world
    env=world.moutain_car()
    agent = evaluate_agent(env, policy_name=policy,state_type="continue",degree=degree, expansion=expansion)

    learning_agent = agent.q_learning_continue
    result = agent.grid_search_continue(learning_agent, agent_name="q", alphas=alphas, epsilons=epsilons)

    learning_agent = agent.sarsa_continue
    result = agent.grid_search_continue(learning_agent, agent_name="sarsa", alphas=alphas, epsilons=epsilons)

    return result


def plot(output_file):
	(x,y) = pickle.load(open(output_file,"rb"))
	fig, ax = plt.subplots(figsize=(10,5))
	ax.set_ylabel("Square mean td error")
	ax.set_xlabel("Step size")
	ax.set_xscale("log", nonposx='clip') # log (x)
	plt.tight_layout()
	plt.plot(x,y,"r+",linestyle="-")
	fig.savefig(output_file+".pdf")
#=========================== End of Homework 4 =========

#Test area

#Question 1
#homework4_gridworld_q_learning(alphas=[0.08, 0.1],epsilons=[0.05, 0.2], decay=True,more="")
#homework4_cart_pole_q_learning(alphas=[0.0008, 0.001, 0.005, 0.008], epsilons=[0.01, 0.05, 0.1,0.2])
#homework4_cart_pole_q_learning(alphas=[0.001], epsilons=[0.05, 0.2],decay=True,trail_n=100, degree=4)
homework4_cart_pole_q_learning(alphas=[0.001], epsilons=[0.01,0.02,0.05,0.1],decay=False,trail_n=20, degree=5, num_e=1000,more="1000",name_e = "")

#Question 2 Try RBF basis
#homework4_cart_pole_q_learning(expansion="rbf",alphas=[0.001], epsilons=[0.05],decay=False, trail_n = 60, num_e=200,degree=5,more="nonodecay")

#Question 4 Try Softmax
#homework4_gridworld_q_learning(policy="softmax")
#homework4_cart_pole_q_learning(policy="softmax", alphas=[0.001], epsilons=[1,5,20], trail_n=50, decay=True, num_e=200)

#Question 5 Moutaincar
#homeork4_moutaincart_q_learning(expansion="f",alphas=[0.0008, 0.001, 0.005, 0.008, 0.01], epsilons=[0.01, 0.05, 0.1, 0.2],degree=5,trail_n=100)
#homeork4_moutaincart_q_learning(expanson="rbf", alphas=[0.0008, 0.001, 0.005, 0.008, 0.01], epsilons=[0.01, 0.05, 0.1, 0.2],degree=3,trail_n=100)