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
import VAE_dataset_modul 
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

model = VAE_model.VAE(3, latent_size = 64)
model.to(device = device)
optimizer = optim.Adam(model.parameters())
num_of_rollouts=5 



def loss_function(recon_x, x, mu, logsigma):
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch):

	model.train()
   
	
	for current_epoch in range(epoch):
		train_loss = 0
		for rollout_ind in range(num_of_rollouts):
			npzfile=np.load('./datasets/duckie/rollout_train_%d.npz'%rollout_ind)
			train_dataset = VAE_dataset_modul.VAE_dataset(npzfile, batch_size = 2)
			train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)

			for batch_idx, data in enumerate(train_loader):
				optimizer.zero_grad()
				recon_batch, mu, logvar = model(data)
				loss = loss_function(recon_batch, data, mu, logvar)
				loss.backward()
				train_loss += loss.item()
				optimizer.step()
				if batch_idx % 20 == 0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						current_epoch, batch_idx * len(data),len(train_loader.dataset),
						100. * batch_idx / len(train_loader),
						loss.item() / len(data)))

				PATH = os.getcwd() + '/VAE_best.pth'
				torch.save({
					'epoch': current_epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'train_loss': loss.item(),
					}, PATH)


train(3)