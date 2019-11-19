import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def MDRNN_dataset(npzfile):
  class MDRNN_dataset_class(Dataset):
    def __init__(self):
    
      self.device = torch.device('cuda:0')

    def transform(self, obs, rew, act, term, next_obs):
                        
      tensor_obs = torch.tensor((obs / 255.0), device = self.device).permute(2,0,1).float()
      tensor_next_obs = torch.tensor((next_obs / 255.0), device = self.device).permute(2,0,1).float()
      tensor_rew=torch.tensor(rew, device = self.device)
      tensor_act=torch.tensor(act, device = self.device)
      tensor_term=torch.tensor(term, device = self.device)

      return tensor_obs, tensor_rew, tensor_act, tensor_term, tensor_next_obs

    def __getitem__(self, index):
     
      obs = npzfile['observations'][index]
      rew = npzfile['rewards'][index+1]
      act = npzfile['actions'][index+1]
      term = npzfile['terminals'][index+1].astype(float)
      next_obs = npzfile['observations'][index + 1]

      tensor_obs, tensor_rew, tensor_act, tensor_term, tensor_next_obs = self.transform(obs, rew, act, term, next_obs)

      return tensor_obs, tensor_rew, tensor_act, tensor_term, tensor_next_obs

    def __len__(self): 
      return (len(npzfile['terminals']) - 1)
  MDRNN_dataset_class=MDRNN_dataset_class()
  return MDRNN_dataset_class


    

