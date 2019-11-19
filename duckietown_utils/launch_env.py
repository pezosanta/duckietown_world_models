from env import launch_env, wrap_env


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

env = launch_env(map_name = training_map)
env = wrap_env(action_type = action_type[0], env = env, add_observation_wrappers = False, add_action_wrappers = True, add_reward_wrappers = True, lane_penalty = False)

# TODO #