import torch
from torch import nn


class LSTMInContext_detailed(nn.Module):
    """
    lstm done over a sequence (cur_point, control, T_point, T_ctrl, T_next_point)
    """

    def __init__(self, hid_dim, num_layers=2):
        input_size = 10  # (cur_point, control, T_point, T_ctrl, T_next_point)

        self.model = nn.LSTM(input_size, hid_dim, num_layers)
        self.T = None

    def adapt_to_traj(self, T):
        self.T = T # [seq_len, 6]

    def __call__(self, state, control):
        # [2], [2]

        torch.concatenate(torch.concatenate([state, control]).repeat(self.T.shape[0], 1),)


class LSTMencoder_incontext(nn.Module):
    """
    lstm done over a sequence T, then a neural net that combines the embedding with current point and control
    """
    def __init__(self, lstm_hid_dim, nn_hid_dim):
        super().__init__()

        self.lstm = nn.LSTM(6, lstm_hid_dim, 2, batch_first=True)

        self.nn = nn.Sequential(
            nn.Linear(4+lstm_hid_dim, nn_hid_dim),
            nn.ReLU(),
            nn.Linear(nn_hid_dim, nn_hid_dim),
            nn.ReLU(),
            nn.Linear(nn_hid_dim, nn_hid_dim),
            nn.ReLU(),
            nn.Linear(nn_hid_dim, 2)
        )

        self.T_emb = None

    def adapt_to_traj(self, T):
        # (seq_len, 6)

        T_emb,_ = self.lstm(T)
        self.T_emb = T_emb[-1]

    def __call__(self, state, control):
        total_emb = torch.cat((state.unsqueeze(0), control.unsqueeze(0), self.T_emb), dim=1)
        return self.nn(total_emb)

    def forward(self, X, T):
        # X [batch, 4] T [batch, seq_len, 6]

        T_emb, _ = self.lstm(T)
        input = torch.cat((X, T_emb[:, -1]), dim=1)
        return self.nn(input)

    