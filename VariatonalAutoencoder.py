import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import Data, utils
import Age
from Data import Field

class LayerDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Encoder(nn.Module):
    def __init__(self, latent_dims, features):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(features, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.normal = torch.distributions.Normal(0, 1)
        #self.normal.loc = self.normal.loc.cuda()
        #self.normal.scale = self.normal.scale.cuda()
        self.kl = 0

    def forward(self, X):
        X = self.linear1(X)
        X = self.relu(X)
        mu = self.linear2(X)
        sigma = torch.exp(self.linear3(X))

        X = mu + sigma*self.normal.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return X


class Decoder(nn.Module):
    def __init__(self, latent_dims, features):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, features)

    def forward(self, X):
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        X = torch.sigmoid(X)

        return X

class Autoencoder(nn.Module):
    def __init__(self, latent_dims, features):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, features)
        self.decoder = Decoder(latent_dims, features)

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)

        return X

def load_data():
    print('LOADING DATA')

    df = Data.load('data.parquet')

    X = df.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM])
    y = df[[Field.STRAT]]

    print('ENCODING DATA')

    y = Age.encode_age(y)
    X[Field.UTME] = X[Field.UTME].fillna(X[Field.UTME].min() * .9)
    X[Field.UTMN] = X[Field.UTMN].fillna(X[Field.UTMN].min() * .9)
    X[Field.ELEVATION] = X[Field.ELEVATION].fillna(X[Field.ELEVATION].min() * .8)
    X[Field.DEPTH_BOT] = X[Field.DEPTH_BOT].fillna(-25)
    X[Field.DEPTH_TOP] = X[Field.DEPTH_TOP].fillna(-25)

    X[Field.ELEVATION_TOP] = X[Field.ELEVATION] - X[Field.DEPTH_TOP]
    X[Field.ELEVATION_BOT] = X[Field.ELEVATION] - X[Field.DEPTH_BOT]

    X = X.drop(columns=[Field.ELEVATION, Field.COLOR, Field.HARDNESS] + [f"emb_{i}" for i in range(384)])

    y[Field.AGE] = y[Field.AGE].replace(-100, -1)

    #X[Field.PREVIOUS_AGE] = y.groupby(df[Field.RELATEID])[Field.AGE].shift(-1).fillna(-1)

    mask = y[Field.AGE] != -1
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print('FITTING SCALERS')

    utme_scaler = StandardScaler()
    utme_scaler.fit(X[[Field.UTME]])

    utmn_scaler = StandardScaler()
    utmn_scaler.fit(X[[Field.UTMN]])

    elevation_scaler = StandardScaler()
    elevation_scaler.fit(X[[Field.ELEVATION_TOP]].values.tolist() + X[[Field.ELEVATION_BOT]].values.tolist())

    depth_scaler = StandardScaler()
    depth_scaler.fit(X[[Field.DEPTH_TOP]].values.tolist() + X[[Field.DEPTH_BOT]].values.tolist())

    X[Field.UTME] = utme_scaler.transform(X[[Field.UTME]])
    X[Field.UTMN] = utmn_scaler.transform(X[[Field.UTMN]])
    X[Field.ELEVATION_TOP] = elevation_scaler.transform(X[[Field.ELEVATION_TOP]])
    X[Field.ELEVATION_BOT] = elevation_scaler.transform(X[[Field.ELEVATION_BOT]])
    X[Field.DEPTH_BOT] = depth_scaler.transform(X[[Field.DEPTH_BOT]])
    X[Field.DEPTH_TOP] = depth_scaler.transform(X[[Field.DEPTH_TOP]])

    X = X.astype(float)

    X = X.to_numpy().astype(np.float32)
    y = y[Field.AGE].to_numpy().astype(np.float32)

    data = LayerDataset(X, y)
    dataloader = DataLoader(data, batch_size=128, shuffle=True)

    return dataloader

def plot_latent(autoencoder, dataloader, batches=100, axis=(0, 1)):
    for i, (X, y) in enumerate(dataloader):
        z = autoencoder.encoder(X)

        z = z.detach().numpy()

        plt.scatter(z[:, axis[0]], z[:, axis[1]], c=y, cmap='tab20')

        if i > batches:
            plt.colorbar()
            plt.show()
            break

def train_age(max_epochs, dataloader=None, lr=1e-3):
    if dataloader is None:
        dataloader = load_data()

    print('INITIALIZING MODEL')

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    autoencoder = Autoencoder(3, 6)
    autoencoder = autoencoder.to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    print('TRAINING MODEL')

    for epoch in range(max_epochs):
        total_loss = 0

        for X, y in dataloader:
            X = X.to(device)

            optimizer.zero_grad()

            X_hat = autoencoder(X)
            loss = ((X - X_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()

            total_loss += loss.item()

            optimizer.step()

        print(f'EPOCH {epoch} | {total_loss/len(dataloader)}')

        torch.save(autoencoder.state_dict(), 'autoencoders/age.aem')

    return autoencoder