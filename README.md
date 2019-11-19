# duckietown_world_models
## team: conDUCKtors - milestone 2 for VITMAV45

Implementation of World Models in the Duckietown simulated environment.

### Generating rollouts for training
Rollouts are generated using the rollout_generator.py file, e.g.
```bash
python rollout_generator.py 5 0 0
```
rollout_generator.py script has three command line arguments:
1. number of rollouts to generate
2. integer to switch between training data and testing data generation mode (1 for training data, rollouts will have 'train' in their filenames; 0 for testing data, rollouts will have 'test' in their filenames)
3. integer to switch between map usage modes (0 for exclusively using the 'udem1' map, 1 for changing between available maps)

## Training the VAE
The VAE can be trained using the VAE_training.py file:
```bash
python VAE_training.py
```
The result of the training is the VAE_best.pth file, which contains the trained weights for the VAE.
The VAE_dataset_modul.py file contains the VAE_dataset class 

## Training the MDRNN


## Training the Controller


Our work is based on the following repository, in which the World Models was implemented for the carracing gym environment:
https://github.com/ctallec/world-models/blob/master/trainvae.py
Our Controller is based on the following repository:
https://github.com/pranz24/pytorch-soft-actor-critic
