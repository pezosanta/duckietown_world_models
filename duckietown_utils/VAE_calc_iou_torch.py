'''
Calculates the average IoU for the pairs created from the original and reconstructed images
with the help of a built-in function involved by torch.
'''
import VAE_dataset_modul
import VAE_model
import torch
from torch.utils.data import DataLoader
import numpy as np


latent_size = 64
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
CHECKPOINT_PATH = '/best_VAE.pth'
checkpoint = torch.load(CHECKPOINT_PATH)

model2 = VAE_model.VAE(3, latent_size=latent_size)
model2.load_state_dict(checkpoint['model_state_dict'])
model2.to(device=device)
model2.eval()


batch_size = 1
count=0
num_of_test = 50 # number of test files
running_IoU = 0
for i in range(num_of_test):
  print(i)
  npzfile_test=np.load('datasets/duckie/rollout_test_%d.npz'%i) # loads the test files one-by-one
  test_dataset = VAE_dataset_modul.VAE_dataset(npzfile_test, batch_size = batch_size)
  test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
  for batch_idx, data in enumerate(test_loader):  # batch_size equals 1, so iterating through batches is equivalent to iterating through images
    count+=1
    test_loader_iter = iter(test_loader)
    first_tensor_img = next(test_loader_iter)
    recon_x, mu, logsigma = model2(first_tensor_img)
    orig_image = first_tensor_img[0].type(torch.int)
    reconstructured_image = recon_x[0].type(torch.int)
    running_IoU += torch.sum(torch.eq(reconstructured_image,orig_image)) / (480*640*3)
avg_iou = running_IoU / count

