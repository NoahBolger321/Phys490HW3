import os
import sys
import json
from collections import Counter
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from boltzmann_machine import *

command_inputs = list(sys.argv)
input_data = command_inputs[1]

chains = []
losses = []
epochs = []

# initial J (coupler values) set to one
j_vals = [1, 1, 1, 1]
num_epochs = 0
learn_rate = 0.1
loss = 1

KL_div = nn.KLDivLoss()
with open(input_data) as data:
    for line in data:
        chains.append(line.rstrip())

chain_occurences = Counter(chains)
chain_probs = np.array(list(chain_occurences.values())) / 1000
unique_chains = np.array(list(chain_occurences.keys()))

while num_epochs < 200:

    Z = partition_function(j_vals, unique_chains)
    output_probs = torch.from_numpy(np.log(probability(j_vals, unique_chains, Z)))
    actual_probs = torch.from_numpy(chain_probs)

    epochs.append(num_epochs)
    losses.append(KL_div(output_probs, actual_probs))

    gradients = gradient(chain_probs, unique_chains, j_vals)

    j_vals += learn_rate*np.array(gradients)
    num_epochs += 1

    if (num_epochs % 25) == 0:
        print("Loss: {:.4f}".format(KL_div(output_probs, actual_probs)))

print("Training loss: {:.4f}".format(losses[-1]))
rounded_j_vals = np.round(j_vals, 0)

output_dict = {
    "(0,1)": rounded_j_vals[0],
    "(1,2)": rounded_j_vals[1],
    "(2,3)": rounded_j_vals[2],
    "(3,0)": rounded_j_vals[3]
}

if not os.path.exists("output/"):
    os.makedirs("output/")

with open('output/output.json', 'w') as f:
    json.dump(output_dict, f)

plt.plot(epochs, losses)
plt.title("KL Divergence over Training Epochs")
plt.ylabel("KL Divergence (loss)")
plt.xlabel("Epochs")
plt.show()
plt.savefig("output/KL_div.png")
