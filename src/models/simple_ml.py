import numpy as np

from src.Simulator import RealSimulator

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import torch
from torch import nn as nn

import lightgbm as lgb


class DirectKNN:
    def __init__(self, k):
        self.model = None
        self.k = k

    def adapt_to_traj(self, T):
        self.model = KNeighborsRegressor(self.k, weights='distance')
        X, y = T[:, [5, 10, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]
        self.model.fit(X, y)

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return state + np.abs(control) * diff


class DirectRandomForest:
    def __init__(self):
        self.model = None

    def adapt_to_traj(self, T):
        self.model = RandomForestRegressor(n_jobs=-1)
        X, y = T[:, [5, 10, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]
        self.model.fit(X, y)

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return state + np.abs(control) * diff

    def batch_call(self, inputs):
        abs_vals = np.abs(inputs[:, [2,3]])
        inputs[:, [2, 3]] = np.sign(inputs[:, [2,3]])

        diff = self.model.predict(inputs)
        return inputs[:, :2] + abs_vals* diff



class DirectLightGBM:
    def __init__(self):
        self.model1, self.model2 = None, None

    def adapt_to_traj(self, T):
        self.model1, self.model2 = lgb.LGBMRegressor(), lgb.LGBMRegressor()
        X, y = T[:, [5, 10, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]
        self.model1.fit(X, y[:, 0])
        self.model2.fit(X, y[:, 1])

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return state + np.abs(control) * diff

    def batch_call(self, inputs):
        abs_vals = np.abs(inputs[:, [2,3]])
        inputs[:, [2, 3]] = np.sign(inputs[:, [2,3]])

        diff1, diff2 = self.model1.predict(inputs), self.model2.predict(inputs)
        diff = np.stack((diff1, diff2)).T
        return inputs[:, :2] + abs_vals* diff

class DirectLinearRegression:
    def __init__(self):
        self.model = None

    def adapt_to_traj(self, T):
        self.model = LinearRegression()
        X, y = T[:, [5, 10, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]
        self.model.fit(X, y)

    def __call__(self, state, control):
        # get a prediction of the model
        state, control = np.array(state), np.array(control)
        inp = np.expand_dims(np.concatenate((state, np.sign(control))), 0)
        diff = self.model.predict(inp)[0]
        return state + np.abs(control) * diff


class DirectNeuralNetwork:
    def __init__(self):
        self.model = None

    def adapt_to_traj(self, T):
        hid_dim = 32
        self.model = nn.Sequential(nn.Linear(4, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, 2)
                                   )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        loss_function = torch.nn.MSELoss()

        X, y = T[:, [5, 10, 6, 11]], T[:, [8, 13]] - T[:, [5, 10]]
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        n_epochs = 100  # or whatever
        batch_size = 32  # or whatever

        for epoch in range(n_epochs):

            permutation = torch.randperm(X.size()[0])

            for i in range(0, X.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X[indices], y[indices]

                outputs = self.model(batch_x)
                loss = loss_function(outputs, batch_y)

                loss.backward()
                optimizer.step()

    def batch_call(self, inputs):
        abs_vals = np.abs(inputs[:, [2,3]])
        inputs[:, [2, 3]] = np.sign(inputs[:, [2,3]])

        with torch.no_grad():
            diff = self.model(torch.tensor(inputs, dtype=torch.float32))
        return inputs[:, :2] + abs_vals * diff.numpy()



class Oracle:
    def __init__(self):
        self.sim = RealSimulator()
        self.corr_ids = None

    def adapt_to_traj(self, T):
        self.corr_ids = T[0, [1, 2]]

    def __call__(self, state, control):
        return self.sim.get_next(state, control, self.corr_ids)


class Naive:
    def __init__(self):
        pass

    def adapt_to_traj(self, T):
        pass

    def __call__(self, state, control):
        return state[0] + control[0], state[1] + control[1]
