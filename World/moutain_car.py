import math
import numpy as np
import Util
import random

class moutain_car:
    def __init__(self):
        self.min_x = -1.2
        self.max_x = 0.6
        self.max_speed = 0.07
        self.goal_x = 0.5
        self.Time_Out = 10000

        self.lower = np.array([self.min_x, -self.max_speed])
        self.upper = np.array([self.max_x, self.max_speed])

        #Engine
        self.speed = 0.001
        self.angle = 0.0025

        #action
        #[lower down, stay, speed up ]
        self.action_space = [0,1,2]
        self.action_n =len(self.action_space)

        #State
        self.state_n = 2
        self.state_space = ["position","velocity"]
        self.n_state_space = len(self.state_space)
        self.bound = observation_bound(self.state_n)
        self.bound.set(self.upper, self.lower)

        #Game
        self.start = (-0.6,0)
        self.state = self.start
        self.turn = 0

        self.name = "moutaincar"

    # =================   IMPORTANT ===========================================
    def reset(self):
        self.state = self.start
        self.state = (-0.5,0)
        self.turn = 0
        return self.state

    def step(self, action):
        self.turn += 1

        x, v = self.state
        v += (action - 1) * self.speed + math.cos(3*x) * (-self.angle)
        v = np.clip(v, -self.max_speed, self.max_speed)

        x += v
        x = np.clip(x, self.min_x, self.max_x)

        if x == self.min_x and v < 0:
            v = 0
        reward = -1

        self.state = (x, v)
        done = self._terminal(self.state)


        return self.state, reward, done


    def normalize(self,state):
        bound = [[-1.2, 0.5], [-0.07, 0.07]]
        a = []
        for i in range(len(state)):
            temp = state[i]
            bottom = bound[i][1]-bound[i][0]
            up = state[i] - bound[i][0]
            a.append(up/bottom)


        return tuple(a)

    # ======================================================================

    def _terminal(self,state):
        x, v = state
        #done = bool(x >= self.goal_x or self.turn > self.Time_Out)
        done = bool(x >= self.goal_x)
        return done

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