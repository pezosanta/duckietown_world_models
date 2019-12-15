"""
Script for generating rollouts
"""

from env import launch_env, wrap_env
from os.path import join, exists
import numpy as np
import sys

if len(sys.argv) < 4:
    print("usage: %s no_of_rollouts is_train change_maps" %(sys.argv[0]))
    sys.exit()

rollouts = int(sys.argv[1])

# creating a mode string, which we will use for naming the output file
train = int(int(sys.argv[2]))
if train == 1:
    mode = "train"
elif train == 0:
    mode = "test"
else:
    mode = "valid"

# if change_maps is True, multiple maps are used, otherwise the udem1 map
change_maps = bool(int(sys.argv[3]))

dir_name = 'duckietown_utils/datasets/duckie/'

training_map = [    '4way',
                    'loop_dyn_duckiebots',
                    'loop_empty',
                    'loop_obstacles',
                    'loop_pedestrians',
                    'regress_4way_adam',
                    'regress_4way_drivable',
                    'small_loop',
                    'small_loop_cw',
                    'straight_road',
                    'udem1',
                    'zigzag_dists'   ]

action_type = ['discrete', 'heading'] 

def generate_data(rollouts, data_dir):
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    # Initializing an environment using the action and reward wrappers provided
    if change_maps == False:
        env = launch_env(map_name = training_map[10])
        env = wrap_env(action_type = action_type[0], env = env, add_observation_wrappers = False, add_action_wrappers = True, add_reward_wrappers = True, lane_penalty = False)

    # Enabling the script to use different maps
    # map_counter stores the position in the training_map array
    map_counter = -1

    for i in range(rollouts):
        if change_maps and i % (rollouts / len(training_map)) == 0:
            map_counter += 1
            print(training_map[map_counter])

            env = launch_env(map_name = training_map[map_counter])
            env = wrap_env(action_type = action_type[0], env = env, add_observation_wrappers = False, add_action_wrappers = True, add_reward_wrappers = True, lane_penalty = False)

        # Resetting the environment before start
        env.reset()

        a_rollout = []
        s_rollout = []
        r_rollout = []
        d_rollout = []

        while True:
            action = env.action_space.sample()# taking a random action
            s, r, done, _ = env.step(action)

            # The stored action has to be processed by the action_wrapper
            a_rollout.append(_['Simulator']['action'])

            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]

            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                # Saving the arrays into a single npz file
                np.savez(join(data_dir, 'rollout_%s_%d' %(mode, i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

generate_data(rollouts, dir_name)