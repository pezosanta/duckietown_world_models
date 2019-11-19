import numpy as np
import gym
import gym_duckietown
import logging

from duckietown_utils.observation_wrappers import *
from duckietown_utils.action_wrappers import *
from duckietown_utils.reward_wrappers import *

logging.basicConfig()
logger = logging.getLogger('gym-duckietown')
logger.setLevel(logging.WARNING)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def launch_env(id = None, map_name = "Duckietown-loop_pedestrians-v0"):

    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123,  # random seed
            # map_name="zigzag_dists",
            map_name = map_name,
            max_steps = 500001,  # we don't want the gym to reset itself - maybe we do want it
            domain_rand = 0,
            # randomize_maps_on_reset = True,
            camera_width = CAMERA_WIDTH,
            camera_height = CAMERA_HEIGHT,
            accept_start_angle_deg = 4,  # start close to straight
            full_transparency = True,
            distortion = True,
        )
    else:
        env = gym.make(id)

    return env


# Env and wrappers setup
def wrap_env(action_type, env = None, add_observation_wrappers = True, add_action_wrappers = True, add_reward_wrappers = True, lane_penalty = False):
    if env is None:
        # Create a dummy Duckietown-like env if None was passed. This is mainly necessary to easily run
        # dts challenges evaluate
        env = DummyDuckietownGymLikeEnv()

    # Observation Wrappers
    if add_observation_wrappers:
        env = VAEWrapper(env)
    '''
    env = ResizeWrapper(env, shape=args.resized_input_shape)
    if args.crop_image_top:
        env = ClipImageWrapper(env, top_margin_divider=MARGIN_TOP_DIVIDER)
    if not args.vae:
        env = NormalizeWrapper(env)
    if args.vae:
        env = ClipImageWrapper(env, top_margin_divider=MARGIN_TOP_DIVIDER)
        env = VAEWrapper(vae_path=args.vae_weight_file, env=env)  # VAEWrapper performs the normalization, no need for
    if args.frame_stacking:
        # env = DummyVecEnv([lambda: env])
        # env = VecFrameStack(env, n_stack=3)
        env = ObservationBufferWrapper(env, obs_buffer_depth=args.frame_stacking_depth)
    '''

    # Action Wrappers
    if add_action_wrappers:
        if action_type == 'discrete':
            env = DiscreteWrapper(env)
        elif action_type == 'heading':
            env = Heading2WheelVelsWrapper(env)
    else:
        print("Default Environment Action used")

    # Reward Wrappers
    if add_reward_wrappers:
        env = DtRewardWrapper(env, lane_penalty)       # 'Almasip'
        
    else:
        print("Default Environment Reward used")
        

    return env


class DummyDuckietownGymLikeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )