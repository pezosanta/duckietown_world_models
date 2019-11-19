#import torch
from torch import optim
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tempfile import TemporaryFile
from importlib import import_module
import VAE_model 
import MDRNN_model
import mdrnn_dataset 
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F



def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool, mdrnn):
    """ Compute losses.
    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).
    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    #latent_obs, action,\
     #   reward, terminal,\
        #latent_next_obs = [arr.transpose(1, 0)
          #                 for arr in [latent_obs, action,
           #                            reward, terminal,
            #                           latent_next_obs]]
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = MDRNN_model.gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = F.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = F.mse_loss(rs, reward)
        scale = 64 + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)



def train(epoch):

  cuda_available = torch.cuda.is_available()
  device = torch.device("cuda" if cuda_available else "cpu")

  vae=VAE_model.VAE(3, latent_size = 64)
  vae.cuda()
  vae.eval()
  state=torch.load('best_VAE.pth')
  vae.load_state_dict(state['model_state_dict'])

  mdrnn = MDRNN_model.MDRNN(64, 2, 256, 5)
  mdrnn.cuda()
  mdrnn.train()
  
  optimizer = optim.Adam(mdrnn.parameters())
  num_of_rollouts=1
  
  for current_epoch in range(epoch):
    train_loss = 0
    for rollout_ind in range(num_of_rollouts):
      npzfile=np.load('./datasets/duckie/rollout_train_%d.npz'%rollout_ind)
      mdrnn_data= mdrnn_dataset.MDRNN_dataset(npzfile)
      mdrnn_loader = DataLoader(mdrnn_data, batch_size=4, shuffle=False)
      
      for batch_idx, data  in enumerate(mdrnn_loader):
        
        obs, reward, action, terminal, next_obs = data

        recon_x, mu, logsigma = vae(obs)
        latent = mu + logsigma.exp() * torch.randn_like(mu)
        latent=latent.float().unsqueeze(0)

        next_recon_x, next_mu, next_logsigma = vae(next_obs)
        next_latent = next_mu + next_logsigma.exp() * torch.randn_like(next_mu)
        next_latent=next_latent.unsqueeze(0)

        action=action.float().unsqueeze(0)
        reward=reward.unsqueeze(0)
        terminal=terminal.unsqueeze(0)
        
        
        
        loss_dict = get_loss(latent, action, reward, terminal, next_latent, True, mdrnn)
        loss=loss_dict['loss']
        train_loss += loss
        
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()
        
        if batch_idx % 20 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              current_epoch, batch_idx * len(data),len(mdrnn_loader.dataset),
              100. * batch_idx / len(mdrnn_loader),
              loss / len(data)))
          
          PATH = 'best_MDRNN.pth'
         
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': mdrnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss
        }, PATH)
          
train(100)
