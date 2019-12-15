# duckietown_world_models
## team: conDUCKtors - homework for VITMAV45, 13.12.2019

![Demo](Gifs/duckie_test.gif)

Implementation of World Models in the Duckietown simulated environment. Our work is based on the following repository, in which the World Models was implemented for the carracing gym environment:
https://github.com/ctallec/world-models/

Our Controller is based on the following repository: https://github.com/pranz24/pytorch-soft-actor-critic, which we downloaded into the root folder of our "milestone_3" branch.

The scripts described below usually are located in the duckietown_utils directory of this branch.

### Generating rollouts for training

Running the following Python script generates the rollouts for training the VAE. It requires three command line arguments: the number of rollouts for training, testing and validation, respectively.

```bash
python3.6 create_rollouts.py 100 50 20
```

Using the above script, the following directory structure is automatically created:
```bash
└── duckietown_utils
    └── datasets
        ├── duckie
        └── images
```
Rollouts are generated using the rollout_generator.py script (which is executed automatically by the create_rollouts.py script), e.g.
```bash
python3.6 rollout_generator.py 100 1 0
python3.6 rollout_generator.py 50 0 0
python3.6 rollout_generator.py 20 2 0
```
rollout_generator.py script has three command line arguments:
1. number of rollouts to generate
2. integer to switch between training, validation and testing data generation mode (1 for training, rollouts will have 'train' in their filenames; 0 for testing, rollouts will have 'test' in their filenames; any other integer value for validation, rollouts will have 'valid' in their filenames)
3. integer to switch between map usage modes (0 for exclusively using the 'udem1' map, 1 for changing between available maps)

We use two additional scripts, which are modifying the default behaviour of the duckietown gym. Previous studies have shown that the built-in reward function of the duckietown gym needs improvements, which are implemented in the rewared_wrappers.py script. In addition, we use an action wrapper (action_wrappers.py) which makes it possible to change between discrete and continuous action types.

### Visualising the rollouts
The generated rollouts can be visualised using the visualise_rollouts.py script, e.g.
```bash
python3.6 visualise_rollouts.py 5 0
```
visualise_rollouts.py script has two command line arguments:
1. ID of the rollout to visualise (number in the rollout's filename)
2. integer to switch between training, validation and testing data generation mode (1 for training, rollouts will have 'train' in their filenames; 0 for testing, rollouts will have 'test' in their filenames; any other integer value for validation, rollouts will have 'valid' in their filenames)


### Training the VAE
The VAE can be trained using the VAE_training.py script:
```bash
python3.6 VAE_train_final.py
```
The result of the training is the best_VAE.pth file, which contains the trained weights for the VAE.
The VAE_train_final.py script uses the following two scripts:
- VAE_dataset_modul.py with the definition of the VAE_dataset class 
- VAE_model.py with the implementation of the VAE model
- learning.py with the implementation of EarlyStopping and ReduceLRonPlateau

The VAE can be tested by running either the VAE_calc_iou_torch.py or the VAE_calc_iou_per_channel.py script, depending on what metric is interesting for us. It is important to note that for the evaluation of the VAE, a best_VAE.pth file is needed, which contains the weights of the trained network.


### Training the Controller
The Controller can be trained using the controller_train.py script:
```bash
python3.6 controller_train.py
```
We use a SAC controller, which is a Reinforcement Learning algorithm. As an input it takes a 1D-array (length: 128) containing the latent vector corresponding to the current observation (length: 64) and the predicted observation (length: 64). The observation_wrapper.py script is used to access the already trained VAE model.  
