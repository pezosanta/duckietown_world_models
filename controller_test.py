import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from duckietown_utils.env import launch_env, wrap_env
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=8, metavar='N', #256
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N', 
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--render', action="store_true",
                    help='use screen (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# simulator: 1346., 529. sorok
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

actor_paths = './ModelParams/sac_actor.pth'
critic_paths = './ModelParams/sac_critic.pth'

# For saving an episode in GIF
#gif_path = './Gifs/duckie_test.gif'

LSIZE = 64  # SAC latent input size (the output size of VAE)
best_avg_reward = -200.0 # initialize a random previous best average validation reward for cold start

# Start the Duckietown gym env and initialize it with the action, reward and observation wrappers
orig_env = launch_env(map_name = training_map[10])
print(orig_env.action_space)
env = wrap_env(action_type = action_type[0], env = orig_env, add_observation_wrappers = True, add_action_wrappers = True, add_reward_wrappers = True, lane_penalty = True)
print(env.action_space)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
#agent = SAC(env.observation_space.shape[1], env.action_space, args)
agent = SAC(num_inputs = LSIZE, action_space = env.action_space, args = args)

best_avg_reward = agent.load_model(actor_path = actor_paths, critic_path = critic_paths)    # loading parameters into the Actor and Critic networks

def decode_obs_array_to_gif(obs_array):
    img_array = []
    for obs in obs_array:
        img_array.append(Image.fromarray(obs))

    img_array[0].save(gif_path, format='GIF', append_images=img_array[1:],
                      save_all=True, duration=33, loop=1)



avg_reward = 0.
episodes = 10
for _  in range(episodes):
    eval_steps = 0

    #state, orig_state = env.reset()    # For saving an episode in GIF
    #obs_list = []

    state = env.reset()

    if args.render:
        env.render()

    episode_reward = 0
    done = False
    
    while not done:        
        action = agent.select_action(state, eval=True)
        #print('ACTION: {}'.format(action))

        #obs_list.append(orig_state)                                # For saving an episode in GIF
        #next_state, reward, done, orig_state = env.step(action)

        next_state, reward, done, _ = env.step(action)

        if args.render:
            env.render()

        episode_reward += reward

        state = next_state
        eval_steps += 1

        if eval_steps > 200:
            print('BREAK')
            break
    
    #decode_obs_array_to_gif(obs_list)

    avg_reward += episode_reward
avg_reward /= episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
print("----------------------------------------")




'''
def policy_video(env, policy, video_path=None, max_timesteps=500):
    orig_env = env
    if isinstance(orig_env.unwrapped, DummyVecEnv):
        orig_env = orig_env.unwrapped.envs[0]
    while not isinstance(orig_env, gym_duckietown.simulator.Simulator):
        orig_env = orig_env.env
    max_steps_initial = orig_env.max_steps
    orig_env.max_steps = max_timesteps

    print("[duckietown_utils.utils.policy_video] - Video generation started")
    monitored_env = wrappers.Monitor(env, video_path, force=True,) # force = True replaces the previous video in the folder
    obs = monitored_env.reset()
    done = False
    obs_list = []
    while not done:
        obs_list.append(obs)
        action = policy.predict(obs)
        if isinstance(action, tuple):
            action = action[0]
        obs, reward, done, _ = monitored_env.step(action)
    monitored_env.close()

    decode_obs_array_to_gif(env, obs_list, video_path)

    print("[duckietown_utils.utils.policy_video] - Video saved to: {}".format(video_path))
    orig_env.max_steps = max_steps_initial

def decode_obs_array_to_gif(env, obs_array, gif_path):
    vae_wrapped = None
    orig_env = env
    while not isinstance(orig_env, gym_duckietown.simulator.Simulator):
        if isinstance(orig_env, VAEWrapper):
            vae_wrapped = orig_env
        orig_env = orig_env.env

    if vae_wrapped is None:
        return

    img_array = []
    for obs in obs_array:
        z_size = vae_wrapped.vae.z_size
        decoded = vae_wrapped.decode(obs[:z_size][None, ])
        img_array.append(Image.fromarray(decoded[0, ..., ::-1]))

    img_array[0].save(os.path.join(gif_path, 'decoded.gif'), format='GIF', append_images=img_array[1:],
                      save_all=True, duration=33, loop=1)
'''