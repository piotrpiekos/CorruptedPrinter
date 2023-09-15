import torch.nn

from src.data.InContextData import InContextDataset
from torch.utils.data import DataLoader

from src.models.LSTM_incontext import LSTMencoder_incontext


import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
lr = 1e-3
hid_dim = 128
batch_size = 64

training_data = InContextDataset('training')
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

model = LSTMencoder_incontext(hid_dim, 2 * hid_dim).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

wandb.init(project='CorruptedPrinter')

logging_frequency = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    torch.save(model.state_dict(), 'models/lstmemb.pth')

    for i, data in enumerate(train_dataloader):
        X_points, X_trajectories, y_points = data
        X_points, X_trajectories, y_points = X_points.to(device), X_trajectories.to(device), y_points.to(device)

        optimizer.zero_grad()

        y_pred = model.forward(X_points, X_trajectories)
        loss = criterion(y_pred, y_points)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


        if i % logging_frequency == logging_frequency - 1:  # print every 2000 mini-batches
            wandb.log({'train/loss': running_loss / logging_frequency})
            print(f'[{epoch + 1}, {i + 1:5d} out of {len(train_dataloader)}] loss: {running_loss / logging_frequency:.3f}')
            running_loss = 0.0

X_points, X_trajectories, y_points = next(iter(train_dataloader))
