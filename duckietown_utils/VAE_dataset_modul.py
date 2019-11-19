from torch.utils.data import Dataset
import torch

def VAE_dataset(npzfile, batch_size):
	class VAE_dataset_class(Dataset):
		def __init__(self, batch_size):
                        self.batch_size = batch_size
                        self.device = torch.device('cuda:0')

		def transform(self, numpy_img):

			numpy_img = numpy_img/255.0
			cuda_tensor_image = torch.tensor(numpy_img, device = self.device).permute(2,0,1).float()
		                      
			return cuda_tensor_image

		def __getitem__(self, index):

			cuda_tensor_image = self.transform(npzfile['observations'][index])
			
			return cuda_tensor_image

		def __len__(self): 
			return len(npzfile['observations'])
	
	VAE_dataset_class=VAE_dataset_class(batch_size)
	return VAE_dataset_class
