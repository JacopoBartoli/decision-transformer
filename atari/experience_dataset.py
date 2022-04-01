import torch
from torch.utils.data import Dataset

import numpy as np

from collections import namedtuple

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "done"])


class ExperienceDataset(Dataset):
    """
    Class that implements a Dataset of Experiences.
    """

    def __init__(self, states, actions, rtgs, done_idxs, timesteps, block_size):
        """
        Constructor of the object.
        """
        self.states = states
        self.actions = actions
        self.rtgs = rtgs
        self.done_idxs = done_idxs
        self.timesteps = timesteps

        self.block_size = block_size
        self.vocab_size = int(max(actions) + 1)

    def __len__(self):
        return self.states.shape[0] - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)
        # (block_size, 8)

        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps


def load_data(file_path):
    """
    Given a path to a file that contains a dataset, return the Dataset object.
    :param file_path: path to the dataset.
    :return: the ExperienceDataset object.
    """
    data = np.load(file_path, allow_pickle=True)
    item, = data
    srt = np.array(data[item])

    return srt


def create_dataset(data):
    states = np.empty((data.shape[0], 8))
    actions = np.empty(data.shape[0])
    rwds = np.empty(data.shape[0])
    timesteps = np.empty(data.shape[0])
    done_idxs = []

    idx = 0
    ts_count = 0

    # Create done_idx array.
    for item in data:
        state, action, rwd, done = item

        states[idx] = state
        actions[idx] = action

        rwds[idx] = rwd

        ts_count += 1
        timesteps[idx] = ts_count

        if done:
            ts_count = 0
            done_idxs += [idx]
        idx += 1

    done_idxs = np.array(done_idxs)

    # Create reward to go dataset.
    start_index = 0
    rtgs = np.zeros_like(rwds)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = rwds[start_index:i]
        for j in range(i - 1, start_index - 1, -1):  # start from i-1
            rtg_j = curr_traj_returns[j - start_index:i - start_index]
            rtgs[j] = sum(rtg_j)
        start_index = i

    return states, actions, rtgs, done_idxs, timesteps
