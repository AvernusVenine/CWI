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
        self.y = torch.from_numpy(y[Field.STRAT].values).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class IntervalNeuralNetwork(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.linear1 = nn.Linear(3, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, self.num_classes)

        self.dropout = nn.Dropout(p=.2)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = self.dropout(X)
        X = F.relu(self.linear2(X))
        X = self.dropout(X)
        X = F.relu(self.linear3(X))
        X = self.linear4(X)

        return X

class SignedDistanceLoss(nn.Module):

    def __init__(self, sdf, num_classes : int, num=25, lam=2.0, gamma=2.0, alpha=2.0):
        """
        Initializes the SignedDistanceLoss function
        Args:
            sdf: SignedDistanceFunction used to calculate assumed signed distances
            num: Number of points to use when emulating integration over a depth range
            lam: Lambda value used to regularize the weight lambda/NUM_CLASSES for formation membership
            alpha: Weight used for incorrect classification loss
        """
        super().__init__()
        self.sdf = sdf
        self.num_classes = num_classes

        self.num = num
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha

        self.utme_scaler = joblib.load('nn/utme.scl')
        self.utmn_scaler = joblib.load('nn/utmn.scl')
        self.elevation_scaler = joblib.load('nn/elevation.scl')

    def forward(self, model, X, label, data_type, device):

        total_loss = 0

        for idx in range(1, 4):
            mask = data_type == idx

            if not mask.any():
                continue

            X_type = X[mask]
            label_type = label[mask]
            batch_size = X_type.shape[0]

            """Entire section is known for SDF"""
            if idx == 1:
                alphas = torch.linspace(0, 1, self.num, device=device).view(1, self.num)
                elevation_top_expanded = X_type[:, 0].view(batch_size, 1)
                elevation_bot_expanded = X_type[:, 1].view(batch_size, 1)

                elevations = elevation_top_expanded + alphas * (elevation_bot_expanded - elevation_top_expanded)

                spatial = X_type[:, 2:].unsqueeze(1).expand(-1, self.num, -1)

                elevations_expanded = elevations.unsqueeze(2)

                X_type = torch.cat([elevations_expanded, spatial], dim=2)
                X_type = X_type.view(batch_size * self.num, 3)

                output = model(X_type)
                output = output.view(batch_size, self.num, self.num_classes)

                utme = self.utme_scaler.inverse_transform(X_type[:, 1].cpu().numpy().reshape(-1, 1)).flatten()
                utmn = self.utmn_scaler.inverse_transform(X_type[:, 2].cpu().numpy().reshape(-1, 1)).flatten()
                elevation = self.elevation_scaler.inverse_transform(X_type[:, 0].cpu().numpy().reshape(-1, 1)).flatten()

                label_expanded = label_type.repeat_interleave(self.num, dim=0).cpu().numpy().flatten()

                dist = self.sdf.compute_all(utme, utmn, elevation, label_expanded)
                dist = torch.tensor(dist, dtype=torch.float32, device=device)
                dist = dist.view(batch_size, self.num, self.num_classes)

                member_weight = torch.where(dist >= 0,
                                            torch.tensor(1, dtype=torch.float32, device=device),
                                            torch.tensor(0.5, dtype=torch.float32, device=device))

                sdf_loss = F.mse_loss(output, dist, reduction='none')
                sdf_loss = member_weight * sdf_loss

                """Classification Loss"""

                predicted_labels = output.argmax(dim=2)
                label_type_expanded = label_type.unsqueeze(1).expand(-1, self.num)

                classification_loss = torch.tensor(predicted_labels != label_type_expanded, dtype=torch.float32, device=device)
                classification_loss = self.alpha * classification_loss

                """Total Loss"""

                loss = sdf_loss.sum() + classification_loss.sum()

                total_loss = total_loss + loss

            """Only check classification over integral"""
            if idx == 2:
                alphas = torch.linspace(0, 1, self.num, device=device).view(1, self.num)
                elevation_top_expanded = X_type[:, 0].view(batch_size, 1)
                elevation_bot_expanded = X_type[:, 1].view(batch_size, 1)

                elevations = elevation_top_expanded + alphas * (elevation_bot_expanded - elevation_top_expanded)

                spatial = X_type[:, 2:].unsqueeze(1).expand(-1, self.num, -1)

                elevations_expanded = elevations.unsqueeze(2)

                X_type = torch.cat([elevations_expanded, spatial], dim=2)
                X_type = X_type.view(batch_size * self.num, 3)

                output = model(X_type)
                output = output.view(batch_size, self.num, self.num_classes)

                """Possibly add SDF Loss but only taking the loss of values > SDF(x)"""

                """Classification Loss"""

                predicted_labels = output.argmax(dim=2)
                label_type_expanded = label_type.unsqueeze(1).expand(-1, self.num)

                classification_loss = torch.tensor(predicted_labels != label_type_expanded, dtype=torch.float32, device=device)
                classification_loss = self.alpha * classification_loss

                """Total Loss"""

                loss = classification_loss.sum()

                total_loss = total_loss + loss

            """Only endpoints are known for SDF (Mixed code layer)"""
            if idx == 3:
                elevations = X_type[:, 0:1]

                spatial = X_type[:, 2:4]

                X_type = torch.cat([elevations, spatial], dim=1)
                X_type = X_type.view(batch_size, 3)

                output = model(X_type)
                output = output.view(batch_size, self.num_classes)

                utme = self.utme_scaler.inverse_transform(X_type[:, 1].cpu().numpy().reshape(-1, 1)).flatten()
                utmn = self.utmn_scaler.inverse_transform(X_type[:, 2].cpu().numpy().reshape(-1, 1)).flatten()
                elevation = self.elevation_scaler.inverse_transform(X_type[:, 0].cpu().numpy().reshape(-1, 1)).flatten()

                dist = self.sdf.compute_all(utme, utmn, elevation, label_type)
                dist = torch.tensor(dist, dtype=torch.float32, device=device)
                dist = dist.view(batch_size, self.num_classes)

                member_weight = torch.where(dist >= 0,
                                            torch.tensor(1, dtype=torch.float32, device=device),
                                            torch.tensor(0.5, dtype=torch.float32, device=device))

                sdf_loss = F.mse_loss(output, dist, reduction='none')
                sdf_loss = member_weight * sdf_loss

                loss = sdf_loss.sum()

                total_loss = total_loss + loss

        return total_loss.mean() / X.shape[0]

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

def train_model(data, max_epochs=15, lr=1e-3, retrain=False):
    train_loader = data[0]
    test_loader = data[1]
    sdf = data[2]
    encoder = data[3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IntervalNeuralNetwork(len(encoder.classes_))

    if retrain:
        state_dict = torch.load('nn/sdf.pth')
        model.load_state_dict(state_dict)

    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = SignedDistanceLoss(sdf, len(encoder.classes_))

    best_loss = np.inf

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}')

        model.train()

        train_loss = 0
        for X, y, data_type in train_loader:
            X = X.to(device)
            y = y.to(device)
            data_type = data_type.to(device)

            loss = loss_func(model, X, y, data_type, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Train Loss {train_loss / len(train_loader)}')

        model.eval()

        total = 0
        correct = 0

        test_loss = 0
        with torch.no_grad():
            for X, y, data_type in test_loader:
                X = X.to(device)
                y = y.to(device)
                data_type = data_type.to(device)

                loss = loss_func(model, X, y, data_type, device)

                test_loss += loss.item()

        print(f'Test Loss {test_loss / len(test_loader)}')

        if test_loss < best_loss:
            torch.save(model.state_dict(), 'nn/sdf.pth')

def test_model(utme, utmn, depth, elevation):
    warnings.filterwarnings('ignore')

    encoder = joblib.load('nn/strat.enc')

    model = IntervalNeuralNetwork(len(encoder.classes_))
    state_dict = torch.load('nn/sdf.pth')
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

def elevation_cross_section(utme, utmn, elevation, utm_count=100):
    """
    Creates a 2D image of the predicted cross-section between two UTM points at a set elevation

    Args:
        utme: Range of UTME min and UTME max for cross-section bounds
        utmn: UTMN
        elevation: Range of elevation min and elevation max for cross-section bounds
        utm_count: Number of pixels on UTME axis
    Returns:

    """
    warnings.filterwarnings('ignore')

    model = IntervalNeuralNetwork()
    state_dict = torch.load('nn/sdf.pth')
    model.load_state_dict(state_dict)
    model.eval()

    utme_scaler = joblib.load('nn/utme.scl')
    utmn_scaler = joblib.load('nn/utmn.scl')
    elevation_scaler = joblib.load('nn/elevation.scl')

    utme_size = (utme[1] - utme[0])/utm_count
    utmn_size = (utmn[1] - utmn[1])/utm_count

    data = np.full([utm_count, utm_count], -1)

    for idx in range(0, utm_count):

        x = utme_scaler.transform([[utme[1] + utme_size * idx]])[0][0]
        z = elevation_scaler.transform([[elevation]])[0][0]

        for jdx in range(0, utm_count):

            y = utmn_scaler.transform([[utmn[1] + utmn_size * jdx]])[0][0]

            X = torch.tensor([z, x, y]).float().unsqueeze(0)

            with torch.no_grad():
                output = model(X)

            data[jdx, idx] = int(torch.argmax(output))

    plt.figure()

    plt.imshow(data, cmap='jet', origin='lower')
    plt.colorbar()

    plt.savefig('elevation_cross.png')
    plt.close()

def utme_cross_section(utme, utmn, elevation, utm_count=100):
    """
    Creates a 2D image of the predicted cross-section between two UTME points between two elevations

    Args:
        utme: Range of UTME min and UTME max for cross-section bounds
        utmn: UTMN
        elevation: Range of elevation min and elevation max for cross-section bounds
        utm_count: Number of pixels on UTME axis
    Returns:

    """
    warnings.filterwarnings('ignore')

    encoder = joblib.load('nn/strat.enc')

    model = IntervalNeuralNetwork(len(encoder.classes_))
    state_dict = torch.load('nn/sdf.pth')
    model.load_state_dict(state_dict)
    model.eval()

    utme_scaler = joblib.load('nn/utme.scl')
    utmn_scaler = joblib.load('nn/utmn.scl')
    elevation_scaler = joblib.load('nn/elevation.scl')

    utme_size = (utme[1] - utme[0])/utm_count

    data = np.full([(elevation[0] - elevation[1]), utm_count], -1)

    for idx in range(0, utm_count):

        x = utme_scaler.transform([[utme[1] + utme_size * idx]])[0][0]
        y = utmn_scaler.transform([[utmn]])[0][0]

        for jdx in range(0, elevation[0] - elevation[1]):

            z = elevation_scaler.transform([[jdx + elevation[1]]])[0][0]

            X = torch.tensor([z, x, y]).float().unsqueeze(0)

            with torch.no_grad():
                output = model(X)

            data[jdx, idx] = int(torch.argmax(output))

    plt.figure()

    plt.imshow(data, cmap='jet', origin='lower')
    plt.colorbar()

    plt.savefig('utme_cross.png')
    plt.close()