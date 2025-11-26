import torch
import torch.nn as nn
from torch.utils.data import Dataset

class LayerDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data.iloc[idx], dtype=torch.long)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        return data, label

class LayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LayerRNN, self).__init__()
        self.hidden_size = hidden_size

        #TODO: Try bidirectional true and false
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, nonlinearity='relu')

        self.linear1 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        h0 = torch.zeros(1, X.size(0), self.hidden_size).to(X.device)
        X, _ = self.rnn(X, h0)
        X = self.linear1(X[:,-1,:])
        return X

def train_loop(train_loader, model, device, loss_func, optimizer):
    """
    Train loop for a given deep learning model
    :param train_loader: Train Dataloader
    :param model: Model
    :param device: Device (CPU or CUDA)
    :param loss_func: Loss function
    :param optimizer: Optimizer
    :return: Total train loop loss
    """
    total_loss = 0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        loss = loss_func(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

    return total_loss

def test_loop(test_loader, model, device, loss_func):
    """
    Test loop for a given deep learning model
    :param test_loader: Test Dataloader
    :param model: Model
    :param device: Device (CPU or CUDA)
    :param loss_func: Loss function
    :return: Total test loop loss, Test loop accuracy
    """
    model.eval()

    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = loss_func(outputs, y)

            total_loss = total_loss + loss.item()

            labels = torch.argmax(outputs.data)
            correct = correct + (labels == y).sum().item()

    return total_loss, correct/len(test_loader)