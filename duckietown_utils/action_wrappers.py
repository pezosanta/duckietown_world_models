import gym
import numpy as np
from gym import spaces


class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Box(low = 0., high = 1., shape = (3,))
        #self.action_space = spaces.Discrete(3)


    def action(self, action):
        argmax_action = np.argmax(action)
        # sampled_action = np.random.sample([0, 1, 2, 3], 1, p=action)
        
        # Turn left
        if argmax_action == 0:
            vels = [0., 1.]
        # Turn right
        elif argmax_action == 1:
            vels = [1., 0.]
        # Go forward
        elif argmax_action == 2:
            vels = [1., 1.]
        # Go backward
        #elif argmax_action == 3:
            #vels = [-1., -1.] #
        # Stop
        #elif argmax_action == 4:
            #vels = [0., 0.]
        else:
            assert False, "unknown action"
       
        return np.array(vels)


class Heading2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env, speed = 1.0):
        super(Heading2WheelVelsWrapper, self).__init__(env)
        self.speed = speed
        self.action_space = spaces.Box(low = -1., high = 1., shape = (1,))

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        # action = [-0.5 * action + 0.5, 0.5 * action + 0.5]
        action = np.clip(np.array([1 + action, 1 - action]), 0., 1.)  # Full speed single value control
        return action
