"""
Visualising observations for a given rollout.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("usage: %s rolloud_ind is_train" %sys.argv[0])
    sys.exit() 

rollout_id = int(sys.argv[1])
train = bool(int(sys.argv[2]))
if train:
    mode = "train"
else:
    mode = "test"

data = np.load("./datasets/duckie/rollout_%s_%d.npz" %(mode, rollout_id))

observations = data['observations']
rewards = data['rewards']
actions = data['actions']
terminals = data['terminals']

for i in range(len(observations)):
    plt.figure()
    plt.imsave("./datasets/images/rollout_%s_%d_%d.png" %(mode, rollout_id, i), observations[i])
