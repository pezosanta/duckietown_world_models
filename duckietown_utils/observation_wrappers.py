import torch
import gym
import gym_duckietown
import os
from duckietown_utils.VAE_model_with_batchnorm import VAE
#from duckietown_utils.VAE_model_16000 import VAE
#from duckietown_utils.MDRNN_model import MDRNN

'''
# Google Colab paths
VAE_CHECKPOINT_PATH = '/content/drive/My Drive/Duckietown - World Models/model.pth'
MDRNN_CHECKPOINT_PATH = '/content/drive/My Drive/Duckietown - World Models/model.pth'
# Finally we did not use MDRNN
MDRNN_CHECKPOINT_PATH = './duckietown_utils/best_MDRNN.pth'
ASIZE = 2
RSIZE = 25
'''

# VAE latent_obs (output vector) size
LSIZE = 64 
VAE_CHECKPOINT_PATH = './duckietown_utils/best_VAE_with_batchnorm.pth'
#VAE_CHECKPOINT_PATH = './duckietown_utils/best_VAE_16000.pth'


class VAEWrapper(gym.ObservationWrapper):
    def __init__(self, env = None):
        super(VAEWrapper, self).__init__(env)
        
        # Load VAE model
        self.VAE_checkpoint = torch.load(VAE_CHECKPOINT_PATH)
        print('CHECKPOINT: ' + str(len(self.VAE_checkpoint)))

        self.VAE_model = VAE(img_channels = 3, latent_size = LSIZE)#.load_state_dict(self.VAE_checkpoint['model_state_dict'])#.to(device = torch.device("cuda:0")).eval()
        self.VAE_model.load_state_dict(self.VAE_checkpoint['model_state_dict'])
        #self.VAE_model.cuda()
        self.VAE_model.eval()

        '''
        # Finally we did not use MDRNN
        self.MDRNN_checkpoint = torch.load(MDRNN_CHECKPOINT_PATH)
        self.MDRNN_model = MDRNN(latents = LSIZE, actions = ASIZE, hiddens = RSIZE, gaussians = 5)#.load_state_dict(self.MDRNN_checkpoint['model_state_dict']).to(device = torch.device("cuda:0")).eval()
        self.MDRNN_model.load_state_dict(self.MDRNN_checkpoint['model_state_dict'], strict = False)
        self.MDRNN_model.cuda()
        self.MDRNN_model.eval()
        '''
       
    def observation(self, observation):
        
        with torch.no_grad():
            tensor_observation = torch.from_numpy(observation / 255.0).permute(2,0,1).float().unsqueeze(0).to(device = torch.device('cpu'))#"cuda:0")) 

            _, mu, logsigma = self.VAE_model(tensor_observation)

            tensor_latent_obs = mu + logsigma.exp() * torch.randn_like(mu)

            numpy_latent_obs = tensor_latent_obs.squeeze().cpu().detach().numpy()

        return numpy_latent_obs
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        #print('---------- WE ARE IN WRAP RESET! -------------------')

        '''
        # Finally we did not use MDRNN
        with torch.no_grad():
            tensor_obs = torch.from_numpy(observation / 255.0).float().permute(2, 0, 1).unsqueeze(0)
            _, mu, logsigma = vae(tensor_obs)
            latent_tensor_obs = mu + logsigma.exp() * torch.randn_like(mu))

            rand_action = torch.randn(2,)
            next_mu, next_sigma, _, _, _ = mdrnn(rand_action, latent_tensor_obs)
            latent_next_tensor_obs = next_mu + next_logsigma.exp() * torch.randn_like(next_mu))

            concat_latent_obs = torch.cat((latent_tensor_obs, latent_next_tensor_obs))
            numpy_concat_latent_obs = concat_latent_obs.detach().numpy()
        '''

        return self.observation(observation)#, observation

    def step(self, action):
        #print('---------- WE ARE IN WRAP STEP! -------------------')
    
        observation, reward, done, info = self.env.step(action)
        
        return self.observation(observation), reward, done, info# observation
        