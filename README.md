# duckietown_world_models
## team: conDUCKtors - milestone 2 for VITMAV45, 19.11.2019

Implementation of World Models in the Duckietown simulated environment. Our work is based on the following repository, in which the World Models was implemented for the carracing gym environment:
https://github.com/ctallec/world-models/blob/master/trainvae.py

Our Controller is based on the following repository: https://github.com/pranz24/pytorch-soft-actor-critic, which we downloaded into the root folder of our new "milestone_2" branch.

The scripts described below are located in the duckietown_utils directory of this branch.

### Generating rollouts for training
The following directory structure needs to be created:
```bash
└── duckietown_utils
    └── datasets
        ├── duckie
        └── images
```
Rollouts are generated using the rollout_generator.py script, e.g.
```bash
python rollout_generator.py 5 0 0
```
rollout_generator.py script has three command line arguments:
1. number of rollouts to generate
2. integer to switch between training data and testing data generation mode (1 for training data, rollouts will have 'train' in their filenames; 0 for testing data, rollouts will have 'test' in their filenames)
3. integer to switch between map usage modes (0 for exclusively using the 'udem1' map, 1 for changing between available maps)



### Visualising the rollouts
The generated rollouts can be visualised using the visualise_rollouts.py script, e.g.
```bash
python visualise_rollouts.py 5 0
```
visualise_rollouts.py script has two command line arguments:
1. ID of the rollout to visualise (number in the rollout's filename)
2. integer to switch between training data and testing data generation mode (1 for training data, rollouts with 'train' in their filenames; 0 for testing data, rollouts with 'test' in their filenames)


### Training the VAE
The VAE can be trained using the VAE_training.py script:
```bash
python VAE_training.py
```
The result of the training is the VAE_best.pth file, which contains the trained weights for the VAE.
The VAE_training.py script uses the following two scripts:
- VAE_dataset_modul.py with the definition of the VAE_dataset class 
- VAE_model.py with the implementation of the VAE model

### Training the MDRNN
The MDRNN can be trained using the MDRNN_training.py script:
```bash
python MDRNN_training.py
```
The result of the training is the MDRNN_best.pth file, which contains the trained weights for the MDRNN.
The MDRNN_training.py script uses the following two scripts:
- MDRNN_dataset_modul.py with the definition of the VAE_dataset class 
- MDRNN_model.py with the implementation of the VAE model


### Training the Controller

