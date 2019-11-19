import torch
import gym
import gym_duckietown
from VAE_model import VAE

#import MDRNN

#VAE_CHECKPOINT_PATH = '/content/drive/My Drive/Duckietown - World Models/model.pth'
#MDRNN_CHECKPOINT_PATH = '/content/drive/My Drive/Duckietown - World Models/model.pth'

LSIZE = 64
ASIZE = 2
RSIZE = 25
VAE_CHECKPOINT_PATH = './duckietown_utils/best_VAE.pth'
#VAE_CHECKPOINT_PATH = 'best_VAE.pth'

class VAEWrapper(gym.ObservationWrapper):
    def __init__(self, env = None):
        super(VAEWrapper, self).__init__(env)
        
        # Load VAE model
        self.VAE_checkpoint = torch.load(VAE_CHECKPOINT_PATH)
        print('CHECKPOINT: ' + str(len(self.VAE_checkpoint)))

        self.VAE_model = VAE(img_channels = 3, latent_size = 64)#.load_state_dict(self.VAE_checkpoint['model_state_dict'])#.to(device = torch.device("cuda:0")).eval()
        self.VAE_model.cuda()
        self.VAE_model.eval()
        self.VAE_model.load_state_dict(self.VAE_checkpoint['model_state_dict'])

        #self.VAE_model = VAE_model(img_channels = 3, latent_size = 64).load_state_dict(self.VAE_checkpoint['model_state_dict']).eval()

        #self.MDRNN_checkpoint = torch.load(MDRNN_CHECKPOINT_PATH)
        #self.MDRNN_model = MDRNN(latents = LSIZE, actions = ASIZE, hiddens = RSIZE, gaussians = 5).load_state_dict(self.MDRNN_checkpoint['model_state_dict']).to(device = torch.device("cuda:0")).eval()
        
       
    def observation(self, observation):
        """Note: vae.encode performs the normalization, so do not normalize or scale the image before VAEWrapper!!!"""
        with torch.no_grad():
            tensor_observation = torch.from_numpy(observation / 255.0).permute(2,0,1).float().unsqueeze(0).to(device = torch.device("cuda:0")) # 4D tensort meg kell még oldani !

            _, mu, logsigma = self.VAE_model(tensor_observation)

            tensor_latent_obs = mu + logsigma.exp() * torch.randn_like(mu)

            numpy_latent_obs = tensor_latent_obs.squeeze().cpu().detach().numpy()

        return numpy_latent_obs

        '''
        h = self.render_obs()
        print('h: ' + str(h.shape))

        z = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]     
        print('obs shape: ' + str(observation[0].shape) + str(len(observation[1])))
        print(type(observation[0]))
        #obs = observation[0]
        #z = observation[1]  
        #print('INSIDE WRAPPER, OBS.SHAPE:{}, Z.SHAPE:{} '.format(obs, z)) 
        return z
        '''

    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        print('---------- WE ARE IN WRAP RESET! -------------------')

        '''
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
        
        return self.observation(observation)
        #return numpy_concat_latent_obs


    def step(self, action):
        print('---------- WE ARE IN WRAP STEP! -------------------')
        obs = self.render_obs()
        
        latent_obs = self.observation(obs)
        print('VAE utan a latent_obs:{} '.format(latent_obs.shape))


        observation, reward, done, info = self.env.step(action)
        print('action: ' + str(type(action)) + str(action.shape))
        return self.observation(observation), reward, done, info
        

    '''
    def observation(self, observation):
        raise NotImplementedError
    '''
