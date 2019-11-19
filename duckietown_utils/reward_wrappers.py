import gym
import gym_duckietown
import numpy as np
from gym_duckietown.simulator import NotInLane
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, lane_penalty = False):
        super(DtRewardWrapper, self).__init__(env)
        self.orig_env = env
        while not isinstance(self.orig_env, gym_duckietown.simulator.Simulator):
            self.orig_env = self.orig_env.env
        self.lane_penalty = lane_penalty
        self.angles = []
        self.poss = []
        self.steps = 0
        self.past_rewards = []

    def reward(self, reward):
        if reward == -1000:
            reward = -4
        elif reward > 0:
            reward *= 10
        else:
            reward *= 10
        return reward

        self.steps += 1
        # Check angle to eliminate rapid rotating
        #print(self.orig_env.__dict__)
        #print(self.env.__dict__)
        self.angle = self.orig_env.cur_angle
        self.angles.append(self.angle)
        self.avg_angle = np.mean(self.angles)

        self.angles = self.angles[-10:]
        #print('Angle:', self.angle, 'prev:', self.avg_angle)

        # Currently only want to go straight
        self.ang_rew = - 10 * np.abs(self.angle - self.avg_angle)
        # Calculate moved distance
        self.pos = self.orig_env.cur_pos
        self.poss.append(self.pos)
        self.poss = self.poss[-15:]

        self.moved = np.linalg.norm(np.mean(self.poss, axis=0) - self.pos)
        #print('Pos:', self.pos, 'prev pos:', np.mean(self.poss, axis=0))
        #print('Angle:', self.angle, 'prev angle:', self.avg_angle)
        self.move_rew = 10 * (self.moved)
        # based on experience, this is very probably between 0-0.2
        self.move_rew = (self.move_rew - 0.06)
        if self.move_rew < 0:
            self.move_rew *= 10
        # if self.move_rew > 0.05:
        #     self.move_rew = 0.05

        # Go and try not to turn
        self.my_reward = 100 * self.move_rew + self.ang_rew

        # penalize rapid back-turns
        # self.ang_penalty = 0.0 if abs(self.angles[-1]-self.angles[0]) < 90.0 else -5.0
        # This reward didn't have an effect, because angles are measured in rad --> their difference cannot be > 90.

        # self.past_rewards.append(reward)
        # self.past_rewards = self.past_rewards[-10:]
        # Not used

        if self.lane_penalty:
            # penalize going to the other lane
            try:
                lp = self.orig_env.get_lane_pos2(self.pos, self.orig_env.cur_angle)
                # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}". format(lp.dist, lp.dot_dir, lp.angle_deg))
                if lp.dist < -0.09:  # Dist is negative to the left of the lane center and is -0.1 on the lane center.
                    self.lane_penalty = -1
                else:
                    self.lane_penalty = 0
            except NotInLane:
                return -1000.

        #print('Lane penalty:', self.lane_penalty, self.angle, self.avg_angle)
        #if self.past_rewards[-1] * self.past_rewards[0] < 0:
        #    self.lane_penalty = -5


        #else:
        #    reward += 2

        #print('Step:', self.steps)
        #print('Move reward:', self.move_rew)
        #print('Env reward:', reward)
        #print('Angle turned:', abs(self.angles[-1]-self.angles[0]))

        # give smaller reward for moving over time
        #self.decrease_multiplier = (self.steps + 100000) / (self.steps + 1000)
        #self.decrease_multiplier = min(15, self-decrease_multiplier)
        # print("Rew: {:.2f} | MoveRew: {:.2f}".format(reward, self.move_rew))
        return reward + self.my_reward + self.lane_penalty
        #return reward + 10*(self.move_rew)