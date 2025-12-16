from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
import json

import Age, Texture, Bedrock, Precambrian
import Data
from Data import Field
import config, utils
from FocalLoss import FocalBCELoss, FocalCELoss


class LayerDataset(Dataset):
    def __init__(self, X, y):
        group_idx = len(Bedrock.GROUP_LIST) + 2
        formation_idx = group_idx + len(Bedrock.FORMATION_LIST)
        member_idx = formation_idx + len(Bedrock.MEMBER_LIST)
        category_idx = member_idx + len(Precambrian.CATEGORY_LIST)

        self.data = X
        self.ages = [hole[:, 0].astype(np.int64) for hole in y]
        self.textures = [hole[:, 1].astype(np.int64) for hole in y]
        self.groups = [hole[:, 2:group_idx].astype(np.float32) for hole in y]
        self.formations = [hole[:, group_idx:formation_idx].astype(np.float32) for hole in y]
        self.members = [hole[:, formation_idx:member_idx].astype(np.float32) for hole in y]
        self.categories = [hole[:, member_idx:category_idx].astype(np.float32) for hole in y]
        self.lithologies = [hole[:, category_idx:].astype(np.float32) for hole in y]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label_dict = {
            'age': torch.tensor(self.ages[idx], dtype=torch.long),
            'texture': torch.tensor(self.textures[idx], dtype=torch.long),
            'group': torch.tensor(self.groups[idx], dtype=torch.float32),
            'formation': torch.tensor(self.formations[idx], dtype=torch.float32),
            'member': torch.tensor(self.members[idx], dtype=torch.float32),
            'category': torch.tensor(self.categories[idx], dtype=torch.float32),
            'lithology': torch.tensor(self.lithologies[idx], dtype=torch.float32),
        }

        return data, label_dict

class LayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        super(LayerRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_ages = len(Age.AGE_LIST)
        self.num_textures = len(Texture.TEXTURE_LIST)
        self.num_groups = len(Bedrock.GROUP_LIST)
        self.num_formations = len(Bedrock.FORMATION_LIST)
        self.num_members = len(Bedrock.MEMBER_LIST)
        self.num_categories = len(Precambrian.CATEGORY_LIST)
        self.num_lithologies = len(Precambrian.LITHOLOGY_LIST)

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.age_linear = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_ages),
        )

    def forward(self, X):
        """This originally was going to compute more labels, but was never implemented"""
        output = {}

        h0 = torch.zeros(
            self.num_layers*2,
            X.size(0),
            self.hidden_size
        ).to(X.device)

        c0 = torch.zeros(
            self.num_layers * 2,
            X.size(0),
            self.hidden_size
        ).to(X.device)

        """Pass that feature set through the Recurrent layer"""
        X_rnn, _ = self.rnn(X, (h0, c0))

        batch_size, seq_len, hidden_dim = X_rnn.shape

        X_rnn = X_rnn.reshape(batch_size * seq_len, hidden_dim)

        age = self.age_linear(X_rnn)
        age = age.reshape(batch_size * seq_len, self.num_ages)
        output['age'] = age

        return output

class RNNLoss(nn.Module):
    def __init__(self, device, age_weight=1.0, texture_weight=1.0, group_weight=1.0, formation_weight=1.0, member_weight=1.0,
                 category_weight=1.0, lithology_weight=1.0):
        super().__init__()

        self.device = device

        self.no_label_weight = .5

        """Since certain categories are more important than others, its key that we weigh results from different losses"""
        self.age_weight = age_weight
        self.texture_weight = texture_weight
        self.group_weight = group_weight
        self.formation_weight = formation_weight
        self.member_weight = member_weight
        self.category_weight = category_weight
        self.lithology_weight = lithology_weight

        """Age/Texture can only be one label, while the rest can be multiclass requiring two different loss functions"""

        self.age_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.texture_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        age_pred_flat = y_pred['age'].reshape(-1, y_pred['age'].size(-1))
        age_true_flat = y_true['age'].reshape(-1).long()

        age_loss = self.age_loss(age_pred_flat, age_true_flat)

        return age_loss


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
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    for idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = {k: v.to(device) for k, v in y.items()}

        outputs = model(X)
        loss = loss_func(outputs, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss = total_loss + loss.item()

    return total_loss/len(train_loader)

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

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            outputs = model(X)
            loss = loss_func(outputs, y)

            total_loss = total_loss + loss.item()

    return total_loss/len(test_loader)


class LayerRNNModel:
    """
    This encapsulates the Recurrent Neural Network solution.  It can be trained via its respective function
    however the test function was never fully completed.
    """
    def __init__(self, path):
        self.model = None
        self.pca = None

        self.utme_scaler = None
        self.utmn_scaler = None
        self.elevation_scaler = None
        self.depth_top_scaler = None
        self.depth_bot_scaler = None

        self.path = path
        self.loss_func = RNNLoss(torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                 , 1, 0, 0, 0, 0, 0, 0)

        self.df = None

        self.n_text_cols = 384

    def test(self, relate_id):
        if self.df is None:
            print("LOADING DATA SET")

            df = Data.load('data.parquet')

            print('ENCODING DATA SET')
            df = Age.encode_age(df)
            df = Texture.encode_texture(df)
            df = Bedrock.encode_bedrock(df)
            df = Precambrian.encode_precambrian(df)

            df = utils.encode_hardness(df)
            df = utils.encode_color(df)

            """Have to reorder the columns before we convert them into a numpy array"""
            embedded_cols = [f"emb_{i}" for i in range(384)]
            other_cols = [col for col in df.columns if col not in embedded_cols]
            df = df[other_cols + embedded_cols]

            self.df = df

        if self.pca is None:
            self.pca = joblib.load(f'{self.path}.pca')

        if self.model is None:
            with open(f'{self.path}.json', 'r') as f:
                params = json.load(f)

            self.model = LayerRNN(params['INPUT_SIZE'], params['HIDDEN_SIZE'], params['NUM_LAYERS'])
            self.model.load_state_dict(torch.load(f'{self.path}.mdl'))

        if self.utme_scaler is None:
            self.utme_scaler = joblib.load(f'{self.path}.utme.scl')
        if self.utmn_scaler is None:
            self.utmn_scaler = joblib.load(f'{self.path}.utmn.scl')
        if self.elevation_scaler is None:
            self.elevation_scaler = joblib.load(f'{self.path}.elevation.scl')
        if self.depth_top_scaler is None:
            self.depth_top_scaler = joblib.load(f'{self.path}.top.scl')
        if self.depth_bot_scaler is None:
            self.depth_bot_scaler = joblib.load(f'{self.path}.bot.scl')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X, y = utils.sequence_individual(self.df, relate_id)

        utme_idx = self.df.columns.get_loc(Field.UTME)
        utmn_idx = self.df.columns.get_loc(Field.UTMN)
        elevation_idx = self.df.columns.get_loc(Field.ELEVATION)
        top_idx = self.df.columns.get_loc(Field.DEPTH_TOP)
        bot_idx = self.df.columns.get_loc(Field.DEPTH_BOT)

        non_embedding_cols = X[:, :-self.n_text_cols]

        non_embedding_cols[:, utme_idx] = self.utme_scaler.transform(non_embedding_cols[:, utme_idx])
        non_embedding_cols[:, utmn_idx] = self.utmn_scaler.transform(non_embedding_cols[:, utmn_idx])
        non_embedding_cols[:, elevation_idx] = self.elevation_scaler.transform(non_embedding_cols[:, elevation_idx])
        non_embedding_cols[:, top_idx] = self.depth_top_scaler.transform(non_embedding_cols[:, top_idx])
        non_embedding_cols[:, bot_idx] = self.depth_bot_scaler.transform(non_embedding_cols[:, bot_idx])

        pca_cols = self.pca.transform(X[:, -self.n_text_cols:])

        X = np.concatenate([non_embedding_cols, pca_cols], axis=1)

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(X)

        return output

    def load_and_encode(self):

        pass

    def train(self, random_state=0, max_epochs=10, lr=1e-4, retrain=True, X=None, y=None):
        if X is not None and y is not None:
            self.X = X
            self.y = y

        if self.df is None:
            print("LOADING DATA SET")

            if os.path.isfile(f'{self.path}_data.parquet'):
                self.df = pd.read_parquet(f'{self.path}_data.parquet')
            else:
                df = Data.load('data.parquet')

                print('ENCODING DATA')
                df = Age.encode_age(df)
                df = Texture.encode_texture(df)
                df = Bedrock.encode_bedrock(df)
                df = Precambrian.encode_precambrian(df)

                df = utils.encode_hardness(df)
                df = utils.encode_color(df)

                df[Field.UTME] = df[Field.UTME].fillna(df[Field.UTME].min() * .9)
                df[Field.UTMN] = df[Field.UTMN].fillna(df[Field.UTMN].min() * .9)
                df[Field.ELEVATION] = df[Field.ELEVATION].fillna(df[Field.ELEVATION].min() * .8)

                """Have to reorder the columns before we convert them into a numpy array"""
                embedded_cols = [f"emb_{i}" for i in range(384)]
                other_cols = [col for col in df.columns if col not in embedded_cols + [Field.HARDNESS]]
                df = df[other_cols + [Field.HARDNESS] + embedded_cols]

                df.to_parquet(f'{self.path}_data.parquet')

                self.df = df

        print("SEQUENCING DATA")
        X, y = utils.sequence_layers(self.df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

        utme_idx = self.df.columns.get_loc(Field.UTME)
        utmn_idx = self.df.columns.get_loc(Field.UTMN)
        elevation_idx = self.df.columns.get_loc(Field.ELEVATION)
        top_idx = self.df.columns.get_loc(Field.DEPTH_TOP)
        bot_idx = self.df.columns.get_loc(Field.DEPTH_BOT)

        if retrain:

            print("FITTING SCALERS")

            """Set all null spatial points to a set coordinate not found within Minnesota"""
            self.utme_scaler = StandardScaler()
            utme = np.concatenate(
                [hole[:, utme_idx] for hole in X_train],
                axis=0
            )
            utme[np.isnan(utme)] = self.df[Field.UTME].min() * .9
            self.utme_scaler.fit(utme.reshape(-1, 1))

            self.utmn_scaler = StandardScaler()
            utmn = np.concatenate(
                [hole[:, utmn_idx] for hole in X_train],
                axis=0
            )
            utmn[np.isnan(utmn)] = self.df[Field.UTMN].min() * .9
            self.utmn_scaler.fit(utmn.reshape(-1, 1))

            self.elevation_scaler = StandardScaler()
            elevation = np.concatenate(
                [hole[:, elevation_idx] for hole in X_train],
                axis=0
            )
            elevation[np.isnan(elevation)] = self.df[Field.ELEVATION].min() * .8
            self.elevation_scaler.fit(elevation.reshape(-1, 1))

            """Since depth will always be greater than 0, we set null values to a relatively small negative value"""
            self.depth_top_scaler = StandardScaler()
            depth_top = np.concatenate(
                [hole[:, top_idx] for hole in X_train],
                axis=0
            )
            depth_top[np.isnan(depth_top)] = -25
            self.depth_top_scaler.fit(depth_top.reshape(-1, 1))

            self.depth_bot_scaler = StandardScaler()
            depth_bot = np.concatenate(
                [hole[:, bot_idx] for hole in X_train],
                axis=0
            )
            depth_bot[np.isnan(depth_bot)] = -25
            self.depth_bot_scaler.fit(depth_bot.reshape(-1, 1))

            joblib.dump(self.utme_scaler, f'{self.path}.utme.scl')
            joblib.dump(self.utmn_scaler, f'{self.path}.utmn.scl')
            joblib.dump(self.elevation_scaler, f'{self.path}.elevation.scl')
            joblib.dump(self.depth_top_scaler, f'{self.path}.top.scl')
            joblib.dump(self.depth_bot_scaler, f'{self.path}.bot.scl')

            print("FITTING PCA")

            #Claude used to bugfix this line that unpacks holes into their layers again for PCA
            embeddings = np.concatenate(
                [hole[:, -self.n_text_cols:] for hole in X_train],
                axis=0
            )

            self.pca = Data.fit_pca(embeddings, .9)
            joblib.dump(self.pca, f'{self.path}.pca')
        else:
            print('LOADING PRETRAINED SCALERS')

            self.utme_scaler = joblib.load(f'{self.path}.utme.scl')
            self.utmn_scaler = joblib.load(f'{self.path}.utmn.scl')
            self.elevation_scaler = joblib.load(f'{self.path}.elevation.scl')
            self.depth_top_scaler = joblib.load(f'{self.path}.top.scl')
            self.depth_bot_scaler = joblib.load(f'{self.path}.bot.scl')
            self.pca = joblib.load(f'{self.path}.pca')

        print('BALANCING DATA SET')
        X_train, y_train = utils.reduce_majority(X_train, y_train, {
            'Q': .1,
            'O': .5,
            'E': .5,
            'C': .5,
            'R': .5,
        })

        print("APPLYING DATA REFINEMENTS")

        X_train_pca = []

        for hole in X_train:
            non_embedding_cols = hole[:, :-self.n_text_cols]

            non_embedding_cols[:, utme_idx] = self.utme_scaler.transform(
                non_embedding_cols[:, utme_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, utmn_idx] = self.utmn_scaler.transform(
                non_embedding_cols[:, utmn_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, elevation_idx] = self.elevation_scaler.transform(
                non_embedding_cols[:, elevation_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, top_idx] = self.depth_top_scaler.transform(
                non_embedding_cols[:, top_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, bot_idx] = self.depth_bot_scaler.transform(
                non_embedding_cols[:, bot_idx].reshape(-1, 1)).ravel()

            pca_cols = self.pca.transform(hole[:, -self.n_text_cols:])

            hole_transformed = np.concatenate([non_embedding_cols, pca_cols], axis=1)
            X_train_pca.append(hole_transformed)

        X_test_pca = []
        for hole in X_test:
            non_embedding_cols = hole[:, :-self.n_text_cols]

            non_embedding_cols[:, utme_idx] = self.utme_scaler.transform(
                non_embedding_cols[:, utme_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, utmn_idx] = self.utmn_scaler.transform(
                non_embedding_cols[:, utmn_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, elevation_idx] = self.elevation_scaler.transform(
                non_embedding_cols[:, elevation_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, top_idx] = self.depth_top_scaler.transform(
                non_embedding_cols[:, top_idx].reshape(-1, 1)).ravel()
            non_embedding_cols[:, bot_idx] = self.depth_bot_scaler.transform(
                non_embedding_cols[:, bot_idx].reshape(-1, 1)).ravel()

            pca_cols = self.pca.transform(hole[:, -self.n_text_cols:])

            hole_transformed = np.concatenate([non_embedding_cols, pca_cols], axis=1)
            X_test_pca.append(hole_transformed)

        train_dataset = LayerDataset(X_train_pca, y_train)
        test_dataset = LayerDataset(X_test_pca, y_test)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=utils.rnn_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=256, collate_fn=utils.rnn_collate_fn)

        print("INITIATING MODEL")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LayerRNN(X_train_pca[0].shape[1])
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_loss = np.inf

        train_losses = []
        test_losses = []

        for epoch in range(max_epochs):
            train_loss = train_loop(train_loader, self.model, device, self.loss_func, optimizer)
            train_losses.append(train_loss)
            print(f'EPOCH [{epoch+1}|{max_epochs}] TRAIN: {train_loss:.4f}')

            test_loss = test_loop(test_loader, self.model, device, self.loss_func)
            test_losses.append(test_loss)
            print(f'EPOCH [{epoch+1}|{max_epochs}] TEST: {test_loss:.4f}')

            if test_loss < best_loss:
                torch.save(self.model.state_dict(), f'{self.path}.mdl')
                params = {
                    'INPUT_SIZE': self.model.input_size,
                    'HIDDEN_SIZE': self.model.hidden_size,
                    'NUM_LAYERS': self.model.num_layers
                }

                with open(f'{self.path}.json', 'w') as f:
                    json.dump(params, f)

        return train_losses, test_losses