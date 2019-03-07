def flipCoin( p ):
    r = random.random()
    return r < p


import numpy as np
import random


class mdp_Gridworld:
    #In this world we have Three type of states

    #1.general sate: entering such state get reward 0
    #2.obstacle state: it's like a wall that agent can never enter
    #3.water state: entering such state get reward -10
    #4.Goal state: entering such state get reward 10 and exit

    #grid_word: a 2d matrix store the entering reward of the grid. [x][y]

    def __init__(self, grid_world, goal=(4,4), start=(0, 0)):

        #Get the descriotion of the world
        self.grid_word = grid_world
        self.height = len(grid_world)
        self.width = len(grid_world[0])
        self.policy = "random"
        self.start = start
        self.turn = 0
        self.max_turn = 1000
        self.state = self.start



        self.goal = (self.height-1,self.width-1)

        self.action_space = ["AU","AD","AL","AR"]
        self.n_state = len(self.action_space)

        self.name = "gridworld"

    #return all of the action this state could be excuted.
    def _getActions(self,state):

        if state == self.grid_word.goal_state:
            return ('exit')

        return ("AU","AD","AL","AR")
        # up, down, left, right

    def getInitialState(self):
        return (0,0)

    #each action has same probablity to be chossen
    def random_policy(self):
        random.randint(0, 3)


    def setMyPolicy(self,policy):
        self.policy = policy

    def setPolicyFunction(self,function):
        self.function = function

    def chooseAction(self,state):
        if self.policy == "random":
            action = random.randint(0,3)
            action = self.action_space[action]
            return action

        if self.policy == "function":
            x,y=state
            state = self.state_transform(x,y)
            action_p = self.function(state)
            #TODO Test
            action= np.random.choice(self.action_space, p=action_p)
            return action

        x, y = state
        return self.policy[x][y]


    def takeAction(self, state, action):
        x,y=state
        if state == self.goal:
            return ("end",self.grid_word[x][y])



        allTransitions = self.getAllTransition(state,action)

        dist = [allTransitions[i][1] for i in range(len(allTransitions))]
        index_action_state = self.random_i(dist)

        nextState = allTransitions[index_action_state][0]
        x,y=nextState
        reward = self.grid_word[x][y]

        if type(reward) != int:
            reward = 0

        return (nextState,reward)

    #=+++++++++++++ IMPORTANT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def step(self, action):
        done = False
        state = self.state
        self.turn +=1
        x, y = state
        if state == self.goal:
            return ("end", self.grid_word[x][y])

        allTransitions = self.getAllTransition(state, action)

        dist = [allTransitions[i][1] for i in range(len(allTransitions))]
        index_action_state = self.random_i(dist)

        nextState = allTransitions[index_action_state][0]
        x, y = nextState
        reward = self.grid_word[x][y]

        if type(reward) != int:
            #todo
            reward = 0.0

        if nextState == self.goal:
            done = True
        if self.turn > self.max_turn:
            done = True

        self.state = nextState

        #if self.turn > self.max_turn:
        #    done = True

        return nextState, reward, done

    def reset(self):
        self.state = self.start
        self.turn = 0
        return self.state

    # =+++++++++++++ IMPORTANT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def random_i(self,d):

        r = random.random()

        new_d = [0]
        index = 0
        for i in d:
            temp = new_d[index]+i
            new_d.append(temp)
            index = index+1

        index = 0
        for i in new_d:
            if r < i:
                return index-1
            index =index+1

        return -1

    #TODO:hardcode, need to be updated later
    def ifTerminalState(self,state):
        x, y = state
        if x == 4 and y == 4:
            return True
        return False

    def state_transform(self, x, y):
        grid = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, '#', 12, 13],
                [14, 15, '#', 16, 17], [18, 19, 20, 21, 22]]
        return grid[x][y]


    def getAllTransition(self, current_state, action):
        x, y = current_state
        all_transition = []
        toUp = (self.__check(x-1, y) and (x-1, y)) or current_state
        toRight = (self.__check(x, y+1) and (x, y+1)) or current_state
        toDown = (self.__check(x+1, y) and (x+1, y)) or current_state
        toLeft = (self.__check(x, y-1) and (x, y-1)) or current_state
        stay = current_state

        if action == "AU":
            all_transition.append((toUp, 0.8))
            all_transition.append((toRight, 0.05))
            all_transition.append((toLeft, 0.05))
            all_transition.append((stay, 0.1))
        elif action == "AR":
            all_transition.append((toUp, 0.05))
            all_transition.append((toRight, 0.8))
            all_transition.append((toDown, 0.05))
            all_transition.append((stay, 0.1))
        elif action == "AD":
            all_transition.append((toDown, 0.8))
            all_transition.append((toRight, 0.05))
            all_transition.append((toLeft, 0.05))
            all_transition.append((stay, 0.1))
        elif action == "AL":
            all_transition.append((toUp, 0.05))
            all_transition.append((toDown, 0.05))
            all_transition.append((toLeft, 0.8))
            all_transition.append((stay, 0.1))

        return all_transition

    def __check(self, x, y):
        if y < 0 or y >= self.width:
            return False
        if x < 0 or x >= self.height:
            return False

        return self.grid_word[x][y] != '#'


def grid():
    gird = [[' '], [' '], [10]]
    grid = [[' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ',' ',' '],[' ',' ','#',' ',' '],[' ',' ','#',' ',' '],[' ',' ',-10,' ',10]]


    #grid = [[10,10,10,10,10], [10,10,10,10,10], [10,10,'#',10,10], [10,10,'#',10,10],
    #        [10,10,0,10, 20]]
    return grid



def policy():
    p = [['AR','AR','AR','AR','AD'],
         ['AR','AR','AR','AD','AD'],
         ['AU','AU','#','AD','AD'],
         ['AU','AU','#','AD','AD'],
         ['AU','AU','AR','AR',10]]

    p = [['AR', 'AR', 'AR', 'AD', 'AD'],
         ['AR', 'AR', 'AR', 'AD', 'AD'],
         ['AU', 'AU', '#', 'AD', 'AD'],
         ['AU', 'AU', '#', 'AD', 'AD'],
         ['AU', 'AU', 'AR', 'AR', 10]]

    p = [['AR', 'AR', 'AR', 'AD', 'AD'],
         ['AR', 'AR', 'AR', 'AD', 'AD'],
         ['AU', 'AU', '#', 'AD', 'AD'],
         ['AU', 'AU', '#', 'AD', 'AD'],
         ['AU', 'AU', 'AR', 'AR', 10]]

    return p

def runSimulation(start_state=(0,0),num_e = 10000,p = "random",p_function=0,max_step=1000):

    world = grid()
    world = mdp_Gridworld(world)


    #p=policy()


    world.setMyPolicy(p)
    world.setPolicyFunction(p_function)

    discounted_returns = []
    turns = []

    gamma = 0.9

    n_s8 = 0.00001
    n_s19 = 0


    for i in range(num_e):
        state = start_state
        notDone = True
        returns = 0.000
        turn = 0

        at42 = 0
        while notDone:

            x, y = state
            if turn == 0 and x == 3 and y == 4:
                n_s8 = n_s8+1

            if turn == 11 and x == 4 and y == 2:
                n_s19 = n_s19+1


            action = world.chooseAction(state)
            state, reward = world.takeAction(state, action)
            if reward == -10:
                at42=at42+1

            returns = returns + gamma**turn * reward

            #print reward


            #Check if terminate
            x, y = state
            if state==(4,4):
                break

            if turn >= max_step:
                break
            turn = turn + 1

        discounted_returns.append(returns)
        turns.append(turn)

    mean = sum(discounted_returns) / float(len(discounted_returns))
    var = np.var(discounted_returns)
    max = np.max(discounted_returns)
    min = np.min(discounted_returns)


    #print "Mean returns:", mean
    #print "Variance: ", var
    #print "Max: ", max, "turns: ", turns[np.argmax(discounted_returns)]
    #print "Min: ", min, "turns: ", turns[np.argmin(discounted_returns)]
    #print "mean turns", np.mean(turns)

    return mean, discounted_returns





def homework1():
    print("=============Question 1: for random=======================")
    runSimulation()
    print("=============Question 2,3: for optimal policy=======================")
    runSimulation(p=policy())
    print("=============Question 4: for goal-water=======================")
    n_s8,n_s19=runSimulation(start_state=(3,4),num_e=100000)
    print("# 8: ", n_s8)
    print("# 19: ", n_s19)
    print("P = " ,n_s19*1.0/n_s8)







