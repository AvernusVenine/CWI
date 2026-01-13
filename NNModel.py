import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import warnings

import Data
from Data import Field

class StratDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y[Field.STRAT].values).long()
        self.thickness = torch.from_numpy(y[Field.THICKNESS].values).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.thickness[idx]

class IntervalNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(3, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 6)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)

        return X

class IntervalIntegratedLoss(nn.Module):

    def __init__(self, num=50, lam=.9):
        super().__init__()
        self.num = num
        self.lam = lam

    def forward(self, model, depth_top, depth_bot, X, y, t, device):
        batch_size = X.shape[0]

        alphas = torch.linspace(0, 1, self.num, device=device).view(1, self.num)
        depth_top_expanded = depth_top.view(batch_size, 1)
        depth_bot_expanded = depth_bot.view(batch_size, 1)

        depths = depth_top_expanded + alphas * (depth_bot_expanded - depth_top_expanded)

        spatial = X[:, 2:].unsqueeze(1).expand(-1, self.num, -1)

        depths_expanded = depths.unsqueeze(2)
        X = torch.cat([depths_expanded, spatial], dim=2)
        X = X.view(batch_size * self.num, 3)

        logits = model(X)
        logits = logits.view(batch_size, self.num, 6)

        interval_width = (depth_bot - depth_top) / (self.num - 1)

        weights = torch.ones(self.num, device=device)
        weights[0] = .5
        weights[-1] = .5
        weights = weights.view(1, self.num, 1)

        logits = (logits * weights).sum(dim=1) * interval_width.view(batch_size, 1)
        logits = logits / t.view(batch_size, 1)

        y = y.long()
        loss = F.cross_entropy(logits, y, reduction='none')

        loss = loss * (1 / t)**self.lam

        return loss.mean()

def load_data():
    print('Loading Dataset')
    df = Data.load('weighted.parquet')

    valid_codes = ['CJDN', 'CTCG', 'CSTL', 'CWOC', 'CMTS', 'CTLR', 'CECR', 'CTMZ']

    df = df.sort_values([Field.RELATEID, Field.DEPTH_TOP, Field.DEPTH_BOT])
    df = df[df[Field.STRAT].isin(valid_codes)]

    df = df.dropna(subset=[Field.DEPTH_TOP, Field.DEPTH_BOT, Field.UTME, Field.UTMN])

    df[Field.THICKNESS] = df[Field.DEPTH_BOT] - df[Field.DEPTH_TOP]

    df = df[df[Field.THICKNESS] > 0.0]

    df[Field.ELEVATION_TOP] = df[Field.ELEVATION] - df[Field.DEPTH_TOP]
    df[Field.ELEVATION_BOT] = df[Field.ELEVATION] - df[Field.DEPTH_BOT]

    utme_scaler = StandardScaler()
    df[Field.UTME] = utme_scaler.fit_transform(df[[Field.UTME]])
    joblib.dump(utme_scaler, 'nn/utme.scl')

    utmn_scaler = StandardScaler()
    df[Field.UTMN] = utmn_scaler.fit_transform(df[[Field.UTMN]])
    joblib.dump(utmn_scaler, 'nn/utmn.scl')

    elevation_scaler = StandardScaler()
    combined_elevations = np.concatenate([
        df[Field.ELEVATION_TOP].values.reshape(-1, 1),
        df[Field.ELEVATION_BOT].values.reshape(-1, 1)
    ], axis=0)
    elevation_scaler.fit(combined_elevations)
    df[Field.ELEVATION_TOP] = elevation_scaler.transform(df[[Field.ELEVATION_TOP]])
    df[Field.ELEVATION_BOT] = elevation_scaler.transform(df[[Field.ELEVATION_BOT]])
    joblib.dump(elevation_scaler, 'nn/elevation.scl')

    df[Field.STRAT] = df[Field.STRAT].replace({
        'CJDN' : 'Jordan Sandstone',
        'CTCG' : 'Tunnel City',
        'CSTL' : 'St Lawrence',
        'CWOC' : 'Wonewoc',
        'CMTS' : 'Mt Simon',
        'CTLR' : 'Tunnel City',
        'CECR' : 'Eau Claire',
        'CTMZ' : 'Tunnel City'
    })

    encoder = LabelEncoder()
    df[Field.STRAT] = encoder.fit_transform(df[Field.STRAT])
    joblib.dump(encoder, 'nn/strat.enc')

    X = df[[Field.ELEVATION_TOP, Field.ELEVATION_BOT, Field.UTME, Field.UTMN]]
    y = df[[Field.STRAT, Field.THICKNESS]]

    data = StratDataset(X, y)

    train_size = int(.8 * len(data))
    train, test = random_split(data, [train_size, len(data) - train_size])

    return train, test

def train_model(data=None, max_epochs=15, lr=1e-3, lam=0.0):
    if data is None:
        train, test = load_data()
    else:
        train = data[0]
        test = data[1]

    train_loader = DataLoader(train, batch_size=256, shuffle=True)
    test_loader = DataLoader(test, batch_size=256, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IntervalNeuralNetwork()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = IntervalIntegratedLoss(lam=lam)

    best_loss = np.inf

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        model.train()

        train_loss = 0
        for X, y, t in train_loader:
            X = X.to(device)
            y = y.to(device)
            t = t.to(device)

            loss = loss_func(model, X[:, 0], X[:, 1], X, y, t, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss {train_loss / len(train_loader)}')

        model.eval()

        test_loss = 0
        with torch.no_grad():
            for X, y, t in test_loader:
                X = X.to(device)
                y = y.to(device)
                t = t.to(device)

                loss = loss_func(model, X[:, 0], X[:, 1], X, y, t, device)

                test_loss += loss.item()

        print(f'Test Loss {test_loss / len(test_loader)}')

        if test_loss < best_loss:
            torch.save(model.state_dict(), 'nn/interval.pth')

def test_model(utme, utmn, depth, elevation):
    warnings.filterwarnings('ignore')

    model = IntervalNeuralNetwork()
    state_dict = torch.load('nn/interval.pth')
    model.load_state_dict(state_dict)
    model.eval()

    utme_scaler = joblib.load('nn/utme.scl')
    utmn_scaler = joblib.load('nn/utmn.scl')
    elevation_scaler = joblib.load('nn/elevation.scl')

    utme = utme_scaler.transform([[utme]])[0][0]
    utmn = utmn_scaler.transform([[utmn]])[0][0]
    elevation = elevation_scaler.transform([[elevation - depth]])[0][0]

    X = torch.tensor([elevation, utme, utmn]).float().unsqueeze(0)

    with torch.no_grad():
        output = model(X)

    return F.softmax(output, dim=1)

def test_borehole(utme, utmn, elevation, depth_top, depth_bot):
    encoder = joblib.load('nn/strat.enc')

    depths = []
    labels = []
    probs = []

    for depth in range(depth_top, depth_bot):
        output = test_model(utme, utmn, depth, elevation)

        labels.append(output.argmax().item())
        probs.append(output.max().item())
        depths.append(elevation - depth)

    labels = encoder.inverse_transform(labels)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

    # Setup colors
    unique_labels = encoder.classes_
    colors = plt.cm.tab10(range(len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plot stratigraphy - simple approach
    for i in range(len(depths)):
        ax1.barh(
            y=depths[i],
            width=1,
            height=1,
            color=color_map[labels[i]],
            edgecolor='none'
        )

    # Add legend with unique labels only
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in unique_labels]
    ax1.legend(handles, unique_labels, loc='upper right', fontsize=10)

    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title('Predicted Stratigraphy', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_xticks([])

    # Plot confidence
    ax2.plot(probs, depths, linewidth=2, color='darkblue')
    ax2.fill_betweenx(depths, 0, probs, alpha=0.3, color='lightblue')
    ax2.set_xlabel('Prediction Confidence', fontsize=12)
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_title('Model Confidence', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('borehole.png')

def cross_section_utmn(utme_min, utme_max, elevation_min, elevation_max, utmn, count=1000):

    utme_scaler = joblib.load('nn/utme.scl')
    utmn_scaler = joblib.load('nn/utmn.scl')
    elevation_scaler = joblib.load('nn/elevation.scl')

    encoder = LabelEncoder()
    encoder.fit(['Jordan Sandstone', 'Tunnel City', 'St Lawrence', 'Wonewoc', 'Mt Simon', 'Eau Claire'])



    pass