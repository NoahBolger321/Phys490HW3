import numpy as np


def convert_data(spin):
    """
    :param spin: either +/-
    :return: +1 or -1 depending on spin orientation
    """
    if spin == "+":
        new_val = 1
    elif spin == "-":
        new_val = -1
    return new_val


def hamiltonian(j_vals, chain):
    """
    :param j_vals: J coupler values
    :param chain: single spin coupler chain
    :return: Hamiltonian value of coupler chain given J coupler value
    """
    return -1*np.sum(
                   [convert_data(chain[i])*convert_data(chain[(i+1) % len(chain)])*j_vals[i]
                    for i in range(len(chain))]
    )


def partition_function(j_vals, chains):
    """
    :return: partition function value across all spin chains
    """
    return np.sum([np.exp(-1*hamiltonian(j_vals, c)) for c in chains])


def probability(j_vals, chains, Z):
    """
    :return: Boltzmann distribution probabilities for each chain
    """
    return [np.exp(-1*hamiltonian(j_vals, c)) / Z for c in chains]


def expectation_value(probabilities, chains):
    """
    :return: expectation values for each pair of spins in the set of unique coupler chains
    """
    storage = []
    for spin in range(4):
        spin2 = (spin + 1) % 4
        storage.append(np.sum([convert_data(chains[i][spin])*convert_data(chains[i][spin2])*probabilities[i] for i in range(2**4)]))
    return storage


def gradient(probabilities, chains, j_vals):
    """
    :return: gradient/update value for training of model
    """
    Z = partition_function(j_vals, chains)
    model_prob = probability(j_vals, chains, Z)
    pos_phase = expectation_value(probabilities, chains)
    negative_phase = expectation_value(model_prob, chains)
    return np.array(pos_phase) - np.array(negative_phase)
