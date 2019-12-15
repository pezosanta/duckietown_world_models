'''
This executes a training process, within which the predefined VAE model learns from 80 training files, containing approx. 1920 images and
is validated by interating through 20 validation files containing approx. 480 images. The earlystopping object keeps track of the changes of 
validation error and in this way governs the saving of model parameters. As for batch_size a pretty high value was adjusted in order to stability.
'''


from torch import optim
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import VAE_model 
import VAE_dataset_modul 
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from learning import EarlyStopping, ReduceLROnPlateau



cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu") # learning takes place on GPU with memory of 16 GB

batch_size=128
latent_size = 64
num_of_train = 80
num_of_valid = 20
num_of_test = 50

precalculated = False # with the help of this flag we can decide if we want to start the training from a pretrained state or not
if precalculated:
  CHECKPOINT_PATH = '/best_VAE.pth'
  checkpoint = torch.load(CHECKPOINT_PATH)

  model = VAE_model.VAE(3, latent_size=latent_size)
  model.load_state_dict(checkpoint['model_state_dict'])
else:
  model = VAE_model.VAE(3, latent_size=latent_size)

model.to(device=device)
# preparing the neccessary things for the learning process
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=10)

def loss_function(recon_x, x, mu, logsigma):
    '''
    This function computes the overall loss which breaks down into the MSE and Kullback-Leibler divergence, which is the "distance"
    between two probability distribution.
    '''
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

def validation():
    '''
    This function iterates through the validation files and computes the average validation loss.
    '''
    model.eval() # the model contains batchnorm layers, so this step is crucial
    valid_loss = 0 #overall valid. loss will be accumulated in this
    sum_length = 0
    with torch.no_grad(): 
      for valid_ind in range(num_of_valid):
        npzfile_valid=np.load('datasets/duckie/rollout_valid_%d.npz'%valid_ind) # loading one of the valid. files 
        valid_dataset = VAE_dataset_modul.VAE_dataset(npzfile_valid, batch_size = batch_size) # a dataset model is linked to each file
        valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)# a dataloader object is linked to each file
        for data in valid_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            valid_loss += loss_function(recon_batch, data, mu, logvar).item()
        sum_length += len(valid_loader.dataset)
    valid_loss /= sum_length
    print('====> Validation loss: {:.4f}'.format(valid_loss))
    return valid_loss	


def train(epoch):
  '''
  This function executes the traning consisting of the number of epochs determined by "epoch". 
  '''
  
  for current_epoch in range(epoch):
    model.train() #adjusting training mode, in this way the batchnorm layers come to life 
    
    for train_ind in range(num_of_train): #iterating through the training files, a dataset and a dataloader object belongs to each and every training file
        npzfile_train=np.load('datasets/duckie/rollout_train_%d.npz'%train_ind)
        train_dataset = VAE_dataset_modul.VAE_dataset(npzfile_train, batch_size = batch_size)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        for batch_idx, data in enumerate(train_loader):
          
          optimizer.zero_grad() # setting the gradients to zero in every step
          recon_batch, mu, logvar = model(data) # propagating forward the given batch through the model 
          loss = loss_function(recon_batch, data, mu, logvar) # computing the loss belonging to the original image-recon. image pair
          loss.backward() # computing the "deltas"
          
          optimizer.step() # updating the weights of the model
          if batch_idx % 20 == 0: #logging the events happening
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain loss: {:.6f}'.format(
              current_epoch, batch_idx * len(data),len(train_loader.dataset),
              100. * batch_idx / len(train_loader),
              loss.item()/len(data))) 
          
          
          
    
    valid_loss = validation() #computing validation loss
    if current_epoch == 0:
      earlystopping.best = 1e6 #in the first iteration step I adjusted a very high value as "best", so that the firstly computed valid. loss will be surely lower than this
    if earlystopping.is_better(a = valid_loss, best = earlystopping.best): # if the next computed loss is lower than the current best, let's update the file containing our model
            curr_best = avg_train_loss
            PATH = '/best_VAE.pth'                        
            torch.save({
              'epoch': current_epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': loss.item(),
              'scheduler': scheduler.state_dict(),
              'earlystopping': earlystopping.state_dict()
              }, PATH)
    scheduler.step(valid_loss)
    earlystopping.step(valid_loss)
    if earlystopping.stop:
        print("End of training because of early stopping at epoch {}".format(epoch))
        break

train(50)
