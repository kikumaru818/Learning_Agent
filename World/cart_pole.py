import gym
import math
import numpy as np
import random


class cart_pole:
    """
    Observation:
    Num	    Observation                Min            Max
     0	    Cart Position             -3.0           3.0
     1      Cart velocity             -10            10
     2	    Pole Angle                -pi/2          pi/2
     3	    Pole angel Velocity       -2.5pi         2.5pi
    """

    def __init__(self, start = (0,0,0,0)):
        #Initial state
        self.current_time = 0
        self.start = start
        self.state = self.start

        self.mc = 1.0  # mass of cart
        self.mp = 0.1  # mas of pole
        self.total_mass = (self.mc + self.mp)  # total mass
        self.length = 0.5  # length of pole
        self.gravity = 9.8
        self.delta_t = 0.02  # time different#
        self.F = 10.0

        #Time
        self.max_time = 20.0
        self.max_step = self.max_time / self.delta_t + 8

        # Bound information
        # Angle at which to fail the episode
        self.theta_threshold = math.pi / 2.0
        self.x_threshold = 3.0
        self.v_threshold = 10.0
        self.theta_dot_threshold = math.pi

        # Some basic information for running the trail

        self.n_state_space = 4
        upper = np.array([
            self.x_threshold,
            10,
            self.theta_threshold,
            2.5 * np.pi])

        self.bound = observation_bound(self.n_state_space)
        self.bound.set(upper, -upper)
        self.action_space = ["AL", "AR"]
        self.action_n = len(self.action_space)
        self.policy = "random"
        self.function = None

        self.name = "cartpole"
        self.done_time = 0
        self.done_time_limit = 10


    # ============================== Important =====================================
    def reset(self):
        self.state = self.start
        #self.state = self.bound.sample()
        self.current_time = 0
        self.done_time = 0
        return self.state

    def step(self, action):
        next_state, reward, done = self.takeAction(self.state, action)
        self.state = next_state
        self.current_time += 1
        return next_state, reward, done
    def normalize(self,state):
        bound = [[-3, 3], [-10, 10], [-np.pi/2, np.pi/2], [-np.pi, np.pi]]
        a = []
        for i in range(len(state)):
            temp = state[i]
            bottom = bound[i][1]-bound[i][0]
            up = state[i] - bound[i][0]
            a.append(up/bottom)


        return tuple(a)



    # =============================== IMPORTANT FUNCTION ===========================
    def chooseAction(self, state):
        if self.policy == "random":
            action = random.randint(0, 1)
            action = self.action_space[action]
            return action
        elif self.policy == "function":
            action_p = self.function(state)
            action = np.random.choice(self.action_space, p=action_p)
            return action
        elif self.policy == "action":
            #state = self.state_normolized(state)
            action = self.function(state)
            action = self.action_space[action]

            return action

        return self.action_space[0]



    def takeAction(self, state, action):
        x,v,theta,theta_dot=state
        force = self.F if action=="AR" else -self.F
        sintheta=math.sin(theta)
        costheta=math.cos(theta)



        temp1 = self.gravity*sintheta+costheta*(-force-self.mp*self.length*theta_dot*theta_dot*sintheta)/self.total_mass
        temp2 = self.length*(4.0/3.0-self.mp*costheta*costheta/self.total_mass)

        theta_two_dot = temp1/temp2

        temp1 = force+self.mp*self.length*(theta_dot**2*sintheta-theta_two_dot*costheta)
        temp2 = self.total_mass

        x_two_dot = temp1/temp2



        # TODO:For test delete later
        polemass_length = (self.mp * self.length)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.mp * costheta * costheta / self.total_mass))
        xacc = temp - polemass_length * thetaacc * costheta / self.total_mass
        tau = 0.02
        x2 = x + tau * v
        x_dot = v + tau * xacc
        theta2 = theta + tau * theta_dot
        theta_dot2 = theta_dot + tau * thetaacc
        # using euler approximation:

        x = x+self.delta_t*v
        v = v+self.delta_t*x_two_dot
        theta = theta + self.delta_t*theta_dot
        theta_dot = theta_dot+self.delta_t*theta_two_dot

        # update the state
        next_state = (x, v, theta, theta_dot)
        done = self.ifTerminalState(next_state)


        if not done:
            reward = 1
        else:
            #if self.done_time < self.done_time_limit:
             #   done=False
             #   self.done_time += 1
             #   next_state = state
            reward = 1

        if self.current_time > self.max_step:
            done = True
            reward = 1





        return next_state, reward, done




    def ifTerminalState(self, state):
        x, v, theta, theta_dot = state
        done = x < -self.x_threshold or x > self.x_threshold \
               or theta < -self.theta_threshold or theta > self.theta_threshold

        return done

    # =============================== SET FUNCTION ==================================
    # A serires of set method
    def setMyPolicy(self, policy):
        self.policy = policy

    def setPolicyFunction(self, function):
        self.function = function

    # ============================= Support Function ==================================
    def state_normolized(self, state):
        b=np.array([1,4.17,-1.68,-7.93])
        state = np.array(state)

        state = state/b
        return state


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



def runSimulation(start=(0,0,0,0),num_e=20,p="random", function=None,max_time=20):
    world = cart_pole()
    world.setMyPolicy(p)
    world.setPolicyFunction(function)
    gamma = 1

    discounted_returns = []
    turns = []

    for i in range(num_e):
        state = start
        notDone = True
        returns = 0.0000
        turn = 0

        while notDone:
            action = world.chooseAction(state)
            state, reward, done = world.takeAction(state,action)

            returns += gamma**turn*reward


            temp = max_time/world.delta_t+10
            if turn > max_time/world.delta_t+8:
                #print(state)
                #print("break 20 s ")
                break
            if done:
                #print(state)
                #print("break failed")
                break
            turn +=1

        discounted_returns.append(returns)
        turns.append(turn)

    mean = sum(discounted_returns) / float(len(discounted_returns))
    var = np.var(discounted_returns)
    max = np.max(discounted_returns)
    min = np.min(discounted_returns)

    #print("one simulation done:","mean",mean,"turn",turn)

    return mean, discounted_returns



def runSimulation2(start=(0,0,0,0), num_e=20, p="random", function=None,max_time=20):
    env = gym.make('CartPole-v0')
    env.reset()

    world = cart_pole()
    world.setMyPolicy(p)
    world.setPolicyFunction(function)




    gamma = 1

    discounted_returns = []
    turns = []

    for i in range(num_e):
        state = env.reset()
        state = start
        notDone = True
        returns = 0.0000
        turn = 0

        while notDone:
            action = world.chooseAction(state)

            if action == "AR": action = 1
            else: action = 0

            state,reward,done,_ = env.step(action)
            #a,b,c,d =temp

            returns += gamma**turn*reward

            if turn > max_time/world.delta_t:
                #print(state)
                #print("break 20 s ")

                break
            if done:
                #print(state)
                #print("break failed")
                break
            turn +=1

        discounted_returns.append(returns)
        turns.append(turn)

    mean = sum(discounted_returns) / float(len(discounted_returns))
    var = np.var(discounted_returns)
    max = np.max(discounted_returns)
    min = np.min(discounted_returns)

    print("one simulation done:","mean",mean,"turn",turn)
    return mean


#print ('done')
