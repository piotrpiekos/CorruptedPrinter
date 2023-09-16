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
        self.model = None

    def adapt_to_traj(self, T):
        self.model = lgb.LGBMRegressor()
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
        hid_dim = 128
        self.model = nn.Sequential(nn.Linear(4, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, hid_dim),
                                   nn.ReLU(),
                                   nn.Linear(hid_dim, 2)
                                   )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        pass


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
