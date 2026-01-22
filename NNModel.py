import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import random

import Data
from Data import Field
from SignedDistanceFunction import SignedDistanceFunction

class StratDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y[Field.SDF].values).float()
        self.label = torch.from_numpy(y[Field.STRAT].values).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.label[idx]

class IntervalNeuralNetwork(nn.Module):

    NUM_CLASSES = 1

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(3, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear4 = nn.Linear(64, self.NUM_CLASSES)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        X = self.linear4(X)

        return X

class SignedDistanceLoss(nn.Module):

    NUM_CLASSES = 1

    def __init__(self, sdf, num=50, alpha=2.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.sdf = sdf
        self.num = num
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.utme_scaler = joblib.load('nn/utme.scl')
        self.utmn_scaler = joblib.load('nn/utmn.scl')
        self.elevation_scaler = joblib.load('nn/elevation.scl')

    def forward(self, model, X, label, device):
        batch_size = X.shape[0]

        alphas = torch.linspace(0, 1, self.num, device=device).view(1, self.num)
        depth_top_expanded = X[0, :].view(batch_size, 1)
        depth_bot_expanded = X[1, :].view(batch_size, 1)

        depths = depth_top_expanded + alphas * (depth_bot_expanded - depth_top_expanded)

        spatial = X[:, 2:].unsqueeze(1).expand(-1, self.num, -1)

        depths_expanded = depths.unsqueeze(2)
        X = torch.cat([depths_expanded, spatial], dim=2)
        X = X.view(batch_size * self.num, 3)

        output = model(X)
        output = output.view(batch_size, self.num, self.NUM_CLASSES)

        utme = self.utme_scaler.inverse_transform(X[:, 1].cpu().numpy().reshape(-1, 1)).flatten()
        utmn = self.utmn_scaler.inverse_transform(X[:, 2].cpu().numpy().reshape(-1, 1)).flatten()
        elevation = self.elevation_scaler.inverse_transform(X[:, 0].cpu().numpy().reshape(-1, 1)).flatten()

        label_expanded = label.repeat_interleave(self.num, dim=0).cpu().numpy().flatten()

        dist = self.sdf.compute_all(utme, utmn, elevation, label_expanded)
        dist = torch.tensor(dist, dtype=torch.float32, device=device)
        dist = dist.view(batch_size, self.num, self.NUM_CLASSES)

        central_weight = F.tanh(self.alpha/(torch.abs(dist) + 1e-5))

        loss = F.mse_loss(output, dist, reduction='none')

        loss = central_weight * loss

        return loss.mean()

class IntervalIntegratedLoss(nn.Module):

    NUM_CLASSES = 18

    def __init__(self, num=50, lam=.9, omega=2):
        super().__init__()
        self.num = num
        self.lam = lam
        self.omega = omega

    def forward(self, model, depth_top, depth_bot, X, y, t, w, device):
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
        logits = logits.view(batch_size, self.num, self.NUM_CLASSES)

        weights = torch.ones(self.num, device=device)
        weights[0] = .5
        weights[-1] = .5
        weights = weights.view(1, self.num, 1)

        logits = (logits * weights).sum(dim=1) / weights.sum()
        logits = logits / t.view(batch_size, 1)

        y = y.long()
        loss = F.cross_entropy(logits, y, reduction='none')

        """Layer thickness normalization"""
        loss = w * loss * (1 / t)**self.lam

        return loss.mean()

def condense_layers(df):
    df = df.sort_values([Field.RELATEID, Field.ELEVATION_TOP, Field.ELEVATION_BOT], ascending=[True, False, True])
    new_df = pd.DataFrame(columns=df.columns)

    for _, hole in df.groupby(Field.RELATEID):
        strat = None
        elevation_top = 0
        elevation_bot = 0

        relateid = 0
        utme = 0
        utmn = 0

        for _, row in hole.iterrows():
            if row[Field.STRAT] == strat and row[Field.ELEVATION_TOP] == elevation_bot:
                elevation_bot = row[Field.ELEVATION_BOT]
            elif strat is None:
                elevation_top = row[Field.ELEVATION_TOP]
                elevation_bot = row[Field.ELEVATION_BOT]
                strat = row[Field.STRAT]

                relateid = row[Field.RELATEID]
                utme = row[Field.UTME]
                utmn = row[Field.UTMN]
            else:
                section = pd.DataFrame({
                    Field.RELATEID : [relateid],
                    Field.UTME : [utme],
                    Field.UTMN : [utmn],
                    Field.ELEVATION_TOP : [elevation_top],
                    Field.ELEVATION_BOT : [elevation_bot],
                    Field.STRAT : [strat]
                })
                new_df = pd.concat([new_df, section])

                elevation_top = row[Field.ELEVATION_TOP]
                elevation_bot = row[Field.ELEVATION_BOT]
                strat = row[Field.STRAT]

        section = pd.DataFrame({
            Field.RELATEID: [relateid],
            Field.UTME: [utme],
            Field.UTMN: [utmn],
            Field.ELEVATION_TOP: [elevation_top],
            Field.ELEVATION_BOT: [elevation_bot],
            Field.STRAT: [strat]
        })
        new_df = pd.concat([new_df, section])

    return new_df

def load_data_single():
    warnings.filterwarnings('ignore')

    print('LOADING DATASET')
    df = Data.load('weighted.parquet')
    raw = Data.load_well_raw()

    df[Field.COUNTY] = df[Field.RELATEID].map(raw.set_index(Field.RELATEID)[Field.COUNTY])
    df = df[df[Field.COUNTY] == 50]

    df = df.dropna(subset=[Field.DEPTH_TOP, Field.DEPTH_BOT, Field.UTME, Field.UTMN, Field.ELEVATION])

    df[df[Field.STRAT].astype(str).str[0] == 'D'] = 0
    df[df[Field.STRAT].astype(str).str[0] != 'D'] = 1

    df[Field.ELEVATION_TOP] = df[Field.ELEVATION] - df[Field.DEPTH_TOP]
    df[Field.ELEVATION_BOT] = df[Field.ELEVATION] - df[Field.DEPTH_BOT]

    df = df[[Field.RELATEID, Field.STRAT, Field.UTME, Field.UTMN, Field.ELEVATION_TOP, Field.ELEVATION_BOT]]

    df = condense_layers(df)

    sdf = SignedDistanceFunction(df, 1)

    relateids = list(set(df[Field.RELATEID].values))
    random.shuffle(relateids)

    split = int(len(relateids) * .85)

    train_ids = relateids[:split]
    test_ids = relateids[split:]

    utme_scaler = MinMaxScaler()
    df[Field.UTME] = utme_scaler.fit_transform(df[[Field.UTME]])
    joblib.dump(utme_scaler, 'nn/utme.scl')

    utmn_scaler = MinMaxScaler()
    df[Field.UTMN] = utmn_scaler.fit_transform(df[[Field.UTMN]])
    joblib.dump(utmn_scaler, 'nn/utmn.scl')

    elevation_scaler = MinMaxScaler()
    df[Field.ELEVATION] = elevation_scaler.fit_transform(df[[Field.ELEVATION]])
    joblib.dump(elevation_scaler, 'nn/elevation.scl')

    train_df = df[df[Field.RELATEID].isin(train_ids)]
    test_df = df[df[Field.RELATEID].isin(test_ids)]

    X_train = train_df[[Field.ELEVATION, Field.UTME, Field.UTMN]]
    X_test = test_df[[Field.ELEVATION, Field.UTME, Field.UTMN]]

    y_train = train_df[[Field.STRAT, Field.SDF]]
    y_test = train_df[[Field.STRAT, Field.SDF]]

    train = StratDataset(X_train, y_train)
    test = StratDataset(X_test, y_test)

    train_loader = DataLoader(train, batch_size=512, shuffle=True)
    test_loader = DataLoader(test, batch_size=512)

    return train_loader, test_loader, sdf

def load_data():
    warnings.filterwarnings('ignore')

    print('LOADING DATASET')
    df = Data.load('weighted.parquet')
    raw = Data.load_well_raw()

    df[Field.COUNTY] = df[Field.RELATEID].map(raw.set_index(Field.RELATEID)[Field.COUNTY])
    df = df[df[Field.COUNTY] == 50]

    valid_codes = {
        'DCVL' : 'Lower Cedar',
        'DCVU' : 'Upper Cedar',
        'DSPL' : 'Wapsipinicon',
        'DWPR' : 'Wapsipinicon',
        'OPDC' : 'Prairie Du Chien',
        'OSTP' : 'St Peter Sandstone',
        'OPVL' : 'Platteville',
        'OGWD' : 'Glenwood',
        'ODCR' : 'Decorah Shale',
        'OPOD' : 'Prairie Du Chien',
        'OGCM' : 'Galena',
        'OPSH' : 'Prairie Du Chien',
        'OPGW' : 'Platteville',
        'OGSC' : 'Galena',
        'OGSV' : 'Galena',
        'OMAQ' : 'Maquoketa',
        'OGPR' : 'Galena',
        'OGPC' : 'Galena',
        'OPWR' : 'Prairie Du Chien',
        'OGVP' : 'Galena',
        'ODUB' : 'Galena',
        'CJDN' : 'Jordan Sandstone',
        'CTCG' : 'Tunnel City',
        'CSTL' : 'St Lawrence',
        'CWOC' : 'Wonewoc',
        'CMTS' : 'Mt Simon',
        'CTLR' : 'Tunnel City',
        'CECR' : 'Eau Claire',
        'CTMZ' : 'Tunnel City'
    }

    df = df.dropna(subset=[Field.DEPTH_TOP, Field.DEPTH_BOT, Field.UTME, Field.UTMN, Field.ELEVATION])

    qdf = df[df[Field.STRAT].astype(str).str[0].isin(['Q', 'R', 'W'])]
    qdf[Field.STRAT] = 'Quaternary'

    pdf = df[df[Field.STRAT].astype(str).str[0].isin(['P'])]
    pdf = pdf[~pdf[Field.STRAT].isin(['PUDF', 'PITT', 'PVMT'])]
    pdf[Field.STRAT] = 'Precambrian'

    df = df[df[Field.STRAT].isin(list(valid_codes.keys()))]

    df[Field.STRAT] = df[Field.STRAT].replace(valid_codes)

    df = pd.concat([df, qdf])

    df[Field.ELEVATION_TOP] = df[Field.ELEVATION] - df[Field.DEPTH_TOP]
    df[Field.ELEVATION_BOT] = df[Field.ELEVATION] - df[Field.DEPTH_BOT]

    df = df[[Field.RELATEID, Field.STRAT, Field.UTME, Field.UTMN, Field.ELEVATION_TOP, Field.ELEVATION_BOT]]

    df = condense_layers(df)

    df = df[df[Field.ELEVATION_TOP] - df[Field.ELEVATION_BOT] > 0.0]

    encoder = LabelEncoder()
    df[Field.STRAT] = encoder.fit_transform(df[Field.STRAT])
    joblib.dump(encoder, 'nn/strat.enc')

    count = df[Field.STRAT].value_counts()
    df = df[df[Field.STRAT].isin(count[count > 10].index)]

    """Split the dataset by entire wells instead of by individual layers"""
    relateids = list(set(df[Field.RELATEID].values))
    random.shuffle(relateids)

    split = int(len(relateids) * .85)

    train_ids = relateids[:split]
    test_ids = relateids[split:]

    sdf = SignedDistanceFunction(df, len(encoder.classes_))

    """Create intermediate points to emulate integration"""
    num_points = 50

    size = len(df)

    df = df.reset_index(drop=True)
    df = df.iloc[df.index.repeat(num_points)].reset_index(drop=True)

    alphas = np.tile(np.linspace(0, 1, num_points), size)

    df[Field.ELEVATION] = df[Field.ELEVATION_BOT] + alphas * (df[Field.ELEVATION_TOP] - df[Field.ELEVATION_BOT])

    df[Field.SDF] = df.apply(lambda x: sdf.compute_all(x[Field.UTME], x[Field.UTMN], x[Field.ELEVATION], 18), axis=1)

    def remove_unknown(group):
        elevation_bot = group[Field.ELEVATION_BOT].min()

        mask = group[Field.ELEVATION_BOT] == elevation_bot

        indices = group[mask].sort_values(Field.ELEVATION).index
        count = len(indices) // 2

        return group[~group.index.isin(indices[:count])]

    df = df.groupby(Field.RELATEID, group_keys=False).apply(remove_unknown).reset_index(drop=True)

    utme_scaler = MinMaxScaler()
    df[Field.UTME] = utme_scaler.fit_transform(df[[Field.UTME]])
    joblib.dump(utme_scaler, 'nn/utme.scl')

    utmn_scaler = MinMaxScaler()
    df[Field.UTMN] = utmn_scaler.fit_transform(df[[Field.UTMN]])
    joblib.dump(utmn_scaler, 'nn/utmn.scl')

    elevation_scaler = MinMaxScaler()
    df[Field.ELEVATION] = elevation_scaler.fit_transform(df[[Field.ELEVATION]])
    joblib.dump(elevation_scaler, 'nn/elevation.scl')

    train_df = df[df[Field.RELATEID].isin(train_ids)]
    test_df = df[df[Field.RELATEID].isin(test_ids)]

    X_train = train_df[[Field.ELEVATION, Field.UTME, Field.UTMN]]
    X_test = test_df[[Field.ELEVATION, Field.UTME, Field.UTMN]]

    y_train = train_df[[Field.STRAT, Field.SDF]]
    y_test = train_df[[Field.STRAT, Field.SDF]]

    count = y_train[Field.STRAT].value_counts()
    weights = [(1/count[i])**.25 for i in y_train[Field.STRAT].values]

    sampler = WeightedRandomSampler(weights=weights, num_samples=int(len(y_train)), replacement=True)

    train = StratDataset(X_train, y_train)
    test = StratDataset(X_test, y_test)

    train_loader = DataLoader(train, batch_size=512, sampler=sampler)
    test_loader = DataLoader(test, batch_size=512)

    return train_loader, test_loader, sdf

def train_model(data=None, max_epochs=15, lr=1e-3):
    if data is None:
        train_loader, test_loader, sdf = load_data()
    else:
        train_loader = data[0]
        test_loader = data[1]
        sdf = data[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IntervalNeuralNetwork()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = SignedDistanceLoss(sdf)

    best_loss = np.inf

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        model.train()

        train_loss = 0
        for X, y, label in train_loader:
            X = X.to(device)
            y = y.to(device)

            loss = loss_func(model, X, y, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss {train_loss / len(train_loader)}')

        model.eval()

        test_loss = 0
        with torch.no_grad():
            for X, y, label in test_loader:
                X = X.to(device)
                y = y.to(device)

                loss = loss_func(model, X, y, label)

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

    return output

def test_borehole(utme, utmn, elevation, depth_top, depth_bot):
    encoder = joblib.load('nn/strat.enc')

    depths = []
    labels = []

    for depth in range(depth_top, depth_bot):
        output = test_model(utme, utmn, depth, elevation)

        labels.append(output.argmax().item())
        depths.append(elevation - depth)

    labels = encoder.inverse_transform(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

    unique_labels = encoder.classes_
    colors = plt.cm.tab20(range(len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    for i in range(len(depths)):
        ax1.barh(
            y=depths[i],
            width=1,
            height=1,
            color=color_map[labels[i]],
            edgecolor='none'
        )

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in unique_labels]
    ax1.legend(handles, unique_labels, loc='upper right', fontsize=10)

    ax1.set_ylabel('Elevation', fontsize=12)
    ax1.set_title('Predicted Stratigraphy', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_xticks([])

    ax2.plot(probs, depths, linewidth=2, color='darkblue')
    ax2.fill_betweenx(depths, 0, probs, alpha=0.3, color='lightblue')
    ax2.set_xlabel('Prediction Confidence', fontsize=12)
    ax2.set_ylabel('Elevation', fontsize=12)
    ax2.set_title('Model Confidence', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('borehole.png')

def cross_section_utmn(utme_min, utme_max, elevation_min, elevation_max, utmn, count=1000):
    encoder = joblib.load('nn/strat.enc')

    total_depths = []
    total_labels = []
    total_probs = []

    for utme in range(utme_min, utme_max, int((utme_max-utme_min)/count)):

        depths = []
        labels = []
        probs = []

        for elevation in range(elevation_min, elevation_max):

            output = test_model(utme, utmn, 0, elevation)

            depths.append(elevation)
            labels.append(output.argmax().item())
            probs.append(output.max().item())

        labels = encoder.inverse_transform(labels)

        total_depths.append(depths)
        total_labels.append(labels)
        total_probs.append(probs)

    pass