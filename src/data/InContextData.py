import torch
from torch.utils.data import Dataset

import os


class InContextDataset(Dataset):
    def __init__(self, split, traj_length=512):
        data_directory = os.path.join('data', 'real_data', 'validation' if split == 'val' else 'test')

        self.traj_length = traj_length

    def prepare_singlefile_data(self, data):
        # selects longest trajectory as T and then

        data = data[:, [3, 5, 10, 6, 11, 8, 13]],   # features are (traj_ id, state, control, next_state), [data_len, 7]
        all_trajectories_id = torch.sort(torch.unique(data[:, 3]))
        print(all_trajectories_id.shape)

        trajectories = []
        for traj_id in all_trajectories_id:
            cur_trajectory = data[data[:, 0] == traj_id]
            padded_traj = torch.ones((self.traj_length, 7))  # pad
            padded_traj[:cur_trajectory.shape[0]] = cur_trajectory[:self.traj_length]  # truncate
            trajectories.append(padded_traj)

        torch_trajectories = torch.stack(trajectories)  # [num_trajectories, traj_length, 7 (features)]

        data[:, 0] = torch.remainder(data[:, 0] + 1, len(all_trajectories_id)) # calculate joined trajectory (roll 1)

        X_trajectories = torch_trajectories[data[:, 0], :, 1:] # [data, traj_length, 7]
        X_point = data[:, 1:5] # features, [data, 5]
        y_point = data[:, 5:] # next_state, [data, 2]

        return X_point, X_trajectories, y_point

    def






