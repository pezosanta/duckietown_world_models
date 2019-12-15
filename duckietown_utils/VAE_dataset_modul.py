from torch.utils.data import Dataset
import torch
'''
This is a function which gets an already loaded npz file and a batchsize as input and returns an instance of a dataset class which can be utilized by 
the Dataloader object coming from torch.
'''
def VAE_dataset(npzfile, batch_size):
  class VAE_dataset_class(Dataset):
    def __init__(self, batch_size):
           
      self.batch_size = batch_size
      self.device = torch.device('cuda:0')

    def transform(self, numpy_img): 
      '''
      This function normalizes and converts the images into torch tensors.
      numpy and torch handles the order of dimensions in a different way, that's why permutation is needed.
      '''
      numpy_img = numpy_img/255.0
      cuda_tensor_image = torch.tensor(numpy_img, device = self.device).permute(2,0,1).float() 
                          
      return cuda_tensor_image

    def __getitem__(self, index):
      '''
      Our .npz files contains the images az a numpy array called "observations".
      '''
      cuda_tensor_image = self.transform(npzfile['observations'][index])
      
      return cuda_tensor_image

    def __len__(self): #the dataloader will be able to know the size of a dataset by using this function
      return len(npzfile['observations'])
  
  VAE_dataset_class=VAE_dataset_class(batch_size)
  return VAE_dataset_class
