import sys
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

command_inputs = list(sys.argv)
input_data = command_inputs[1]


def convert_data(spin):
    if spin == "+":
        new_val = 1
    elif spin == "-":
        new_val = -1
    return new_val


def hamiltonian(j_vals, chain):
    return -1*np.sum(
                   [convert_data(chain[i])*convert_data(chain[(i+1) % len(chain)])*j_vals[i]
                    for i in range(len(chain))]
    )


def partition_function(j_vals, chains):
    return np.sum([np.exp(-1*hamiltonian(j_vals, c)) for c in chains])


def probability(j_vals, chains):
    return [np.exp(-1*hamiltonian(j_vals, c)) / partition_function(j_vals, chains) for c in chains]


def expectation_value(probabilities, chains, index1, index2):
    return np.sum([convert_data(chains[i][index1])*convert_data(chains[i][index2])*probabilities[i] for i in range(2**4)])


def gradient(probabilities, chains, index1, j_vals):
    index2 = (index1 + 1) % 4
    pos_phase = expectation_value(probabilities, chains, index1, index2)
    negative_phase = expectation_value(probability(j_vals, chains), chains, index1, index2)

    return np.array(pos_phase) - np.array(negative_phase)

chains = []
losses = []
epochs = []

KL_div = nn.KLDivLoss()
with open(input_data) as data:
    for line in data:
        chains.append(line.rstrip())

chain_occurences = Counter(chains)
chain_probs = np.array(list(chain_occurences.values())) / 1000
unique_chains = np.array(list(chain_occurences.keys()))

# initial J (coupler values) set to one
j_vals = [1, 1, 1, 1]
num_epochs = 0
learn_rate = 0.1
loss = 1

while num_epochs < 200:

    output_probability = torch.from_numpy(np.log(probability(j_vals, unique_chains)))
    targets = torch.from_numpy(chain_probs)

    epochs.append(num_epochs)
    new_loss = KL_div(output_probability, targets)
    losses.append(new_loss)

    update = [gradient(chain_probs, unique_chains, index1, j_vals) for index1 in range(4)]

    j_vals += learn_rate*np.array(update)
    num_epochs += 1

    if (num_epochs % 25) == 0:
        print("Loss: {:.4f}".format(KL_div(output_probability, targets)))

print("Training loss: {:.4f}".format(losses[-1]))
rounded_j_vals = np.round(j_vals, 0)

output_dict = {
    "(0,1)": rounded_j_vals[0],
    "(1,2)": rounded_j_vals[1],
    "(2,3)": rounded_j_vals[2],
    "(3,0)": rounded_j_vals[3]
}

plt.plot(epochs, losses)
plt.show()
plt.savefig("output/KL_div.png")
