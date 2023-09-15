import torch
from torch.utils.data import Dataset

import os
import numpy as np


class InContextDataset(Dataset):
    def __init__(self, split, traj_length=256):
        self.split = split

        self.traj_length = traj_length

        self.dataset_X_points = None
        self.dataset_X_trajectories = None
        self.dataset_y_points = None

        self.prepare_dataset()

    def prepare_singlefile_data(self, data, max_num_trajectories=100):
        data = data[:, [3, 5, 10, 6, 11, 8, 13]]  # features are (traj_ id, state, control, next_state), [data_len, 7]

        # change next_state to next_state - state
        data[:, 5:] = data[:, 5:] - data[:, [1,2]]

        data[:, 0] = data[:, 0] - 1  # fix the indexing after matlab

        all_trajectories_id = torch.sort(torch.unique(data[:, 0])).values

        # take only first 100 trajectories
        all_trajectories_id = all_trajectories_id[:max_num_trajectories]
        data = data[data[:, 0] < max_num_trajectories]

        trajectories = []
        for traj_id in all_trajectories_id:
            cur_trajectory = data[data[:, 0] == traj_id]
            padded_traj = torch.ones((self.traj_length, 7)) * -1  # pad
            padded_traj[:cur_trajectory.shape[0]] = cur_trajectory[:self.traj_length]  # truncate
            trajectories.append(padded_traj)

        torch_trajectories = torch.stack(trajectories)  # [num_trajectories, traj_length, 7 (features)]

        data[:, 0] = torch.remainder(data[:, 0] + 1, len(all_trajectories_id))  # calculate joined trajectory (roll 1)

        X_trajectories = torch_trajectories[data[:, 0].int(), :, 1:]  # [data, traj_length, 6]
        X_point = data[:, 1:5]  # features, [data, 4]
        y_point = data[:, 5:]  # next_state, [data, 2]

        return X_point, X_trajectories, y_point

    def prepare_dataset(self):
        data_trajectory = os.path.join('data', 'real_data', self.split)
        if self.split == 'training':
            filepaths = [os.path.join(data_trajectory, fname) for fname in os.listdir(data_trajectory)]
        else:
            filepaths = [os.path.join(data_trajectory, fname) for subdir in os.listdir(data_trajectory)
                         for fname in os.listdir(os.path.join(data_trajectory, subdir))
                         ]

        files_X_points, files_X_trajectories, files_y_points = [], [], []
        for i, filepath in enumerate(filepaths):
            data = torch.from_numpy(
                np.genfromtxt(filepath, delimiter=",", dtype=np.float32),
            )
            X_point, X_trajectories, y_point = self.prepare_singlefile_data(data)
            files_X_points.append(X_point)
            files_X_trajectories.append(X_trajectories)
            files_y_points.append(y_point)

            if i % 50 == 0:
                print(i, ' files out of ', len(filepaths))

        self.dataset_X_points = torch.concatenate(files_X_points)
        self.dataset_X_trajectories = torch.concatenate(files_X_trajectories)
        self.dataset_y_points = torch.concatenate(files_y_points)

        print(self.dataset_X_trajectories.shape)
        print('dataset size: ', torch.prod(torch.tensor(self.dataset_X_trajectories.shape)) * 4 / (1024) ** 3, 'GB')

    def __len__(self):
        return len(self.dataset_X_trajectories)

    def __getitem__(self, idx):
        return self.dataset_X_points[idx], self.dataset_X_trajectories[idx], self.dataset_y_points[idx]



"""
training_data = InContextDataset('training')

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

X_points, X_trajectories, y_points = next(iter(train_dataloader))

print(X_points.shape)
print(X_trajectories.shape)
print(y_points.shape)
"""