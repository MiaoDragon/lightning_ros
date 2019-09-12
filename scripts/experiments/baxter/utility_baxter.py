import torch
from torch.autograd import Variable
import copy
import numpy as np

def moveit_scrambler(positions, dim):
    new_positions = copy.deepcopy(positions)
    #new_positions = list(positions)
    if dim == 2:
        new_positions[:,2:4] = positions[:,5:7]
        new_positions[:,4:7] = positions[:,2:5]
    elif dim == 1:
        new_positions[2:4] = positions[5:7]
        new_positions[4:7] = positions[2:5]

    return new_positions


def moveit_unscrambler(positions, dim):
    new_positions = copy.deepcopy(positions)
    #new_positions = list(positions)
    if dim == 2:
        new_positions[:,2:5] = positions[:,4:]
        new_positions[:,5:] = positions[:,2:4]
    elif dim == 1:
        new_positions[2:5] = positions[4:]
        new_positions[5:] = positions[2:4]

    return new_positions

def normalize(x, bound):
    # perform unscramble and normalize to transform to neural network input
    # depending on the type of x, change bound
    if isinstance(x, np.ndarray):
        bound = np.array(bound)
        dim = len(x.shape)
    elif isinstance(x, torch.Tensor):
        bound = torch.Tensor(bound)
        dim = len(x.size())
    else:
        bound = torch.Tensor(bound)
        dim = len(x.size())
    if dim == 2:
        if len(x[0]) == len(bound):
            x = moveit_unscrambler(x, dim)
            x = x / bound
        else:
            # concatenation of obstacle, start and goal
            state = x[:, -len(bound)*2:-len(bound)]
            state = moveit_unscrambler(state, dim)
            state = state / bound
            x[:, -len(bound)*2:-len(bound)] = state
            state = x[:, -len(bound):]
            state = moveit_unscrambler(state, dim)
            state = state / bound
            x[:, -len(bound):] = state
    elif dim == 1:
        if len(x) == len(bound):
            x = moveit_unscrambler(x, dim)
            x = x / bound
        else:
            # concatenation of obstacle, start and goal
            state = x[-len(bound)*2:-len(bound)]
            state = moveit_unscrambler(state, dim)
            state = state / bound
            x[-len(bound)*2:-len(bound)] = state
            state = x[-len(bound):]
            state = moveit_unscrambler(state, dim)
            state = state / bound
            x[-len(bound):] = state
    return x
def unnormalize(x, bound):
    # depending on the type of x, change bound
    if isinstance(x, np.ndarray):
        bound = np.array(bound)
        dim = len(x.shape)
    elif isinstance(x, torch.Tensor):
        bound = torch.Tensor(bound)
        dim = len(x.size())
    else:
        bound = torch.Tensor(bound)
        dim = len(x.size())
    if dim ==2:
        if len(x[0]) == len(bound):
            x = x * bound
            x = moveit_scrambler(x, dim)
        else:
            state = x[:,-len(bound)*2:-len(bound)]
            state = state * bound
            state = moveit_scrambler(state, dim)
            x[:,-len(bound)*2:-len(bound)] = state
            state = x[:,-len(bound):]
            state = state * bound
            state = moveit_scrambler(state, dim)
            x[:,-len(bound):] = state
    elif dim ==1:
        if len(x) == len(bound):
            x = x * bound
            x = moveit_scrambler(x, dim)
        else:
            # concatenation of obs, start and goal
            state = x[-len(bound)*2:-len(bound)]
            state = state * bound
            state = moveit_scrambler(state, dim)
            x[-len(bound)*2:-len(bound)] = state
            state = x[-len(bound):]
            state = state * bound
            state = moveit_scrambler(state, dim)
            x[-len(bound):] = state
    return x
