import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import joblib
import json

import Age, Texture, Bedrock, Precambrian
import Data
from Data import Field
import config, utils


class LayerDataset(Dataset):
    def __init__(self, X, ages, textures, groups, formations, members, categories, lithologies):
        self.data = X
        self.ages = ages
        self.textures = textures
        self.groups = groups
        self.formations = formations
        self.members = members
        self.categories = categories
        self.lithologies = lithologies

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
    def __init__(self, input_size, hidden_size=512, num_layers=2):
        super(LayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_ages = len(Age.AGE_LIST)
        self.num_textures = len(Texture.TEXTURE_LIST)
        self.num_groups = len(Bedrock.GROUP_LIST)
        self.num_formations = len(Bedrock.FORMATION_LIST)
        self.num_members = len(Bedrock.MEMBER_LIST)
        self.num_categories = len(Precambrian.CATEGORY_LIST)
        self.num_lithologies = len(Precambrian.LITHOLOGY_LIST)

        self.feature_rnn = nn.RNN(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.age_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_ages)
        )

        self.texture_linear = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.num_ages, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_textures)
        )

        self.group_linear = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.num_ages, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_groups)
        )

        self.formation_linear = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.num_groups, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_formations)
        )

        self.member_linear = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.num_formations, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_members)
        )

        self.category_linear = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.num_ages, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_textures)
        )

        self.lithology_linear = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.num_ages + self.num_categories, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_textures)
        )


    def forward(self, X):
        output = {}

        h0 = torch.zeros(
            self.num_layers*2,
            X.size(0),
            self.hidden_size
        ).to(X.device)

        X, _ = self.feature_rnn(X, h0)
        X = X[:,-1,:]

        age = self.age_linear(X)
        output['age'] = age

        X_age = torch.cat([X, age], dim=1)

        output['texture'] = self.texture_linear(X_age)

        group = self.group_linear(X_age)
        output['group'] = group

        X_group = torch.cat([X, group], dim=1)
        formation = self.formation_linear(X_group)
        output['formation'] = formation

        X_formation = torch.cat([X, formation], dim=1)
        output['member'] = self.member_linear(X_formation)

        category = self.category_linear(X_age)
        output['category'] = category

        X_category = torch.cat([X, age, category], dim=1)
        output['lithology'] = self.lithology_linear(X_category)

        return output

class RNNLoss(nn.Module):
    def __init__(self, age_weight=1.0, texture_weight=1.0, group_weight=1.0, formation_weight=1.0, member_weight=1.0,
                 category_weight=1.0, lithology_weight=1.0):
        super().__init__()

        """Since certain categories are more important than others, its key that we weigh results from different losses"""
        self.age_weight = age_weight
        self.texture_weight = texture_weight
        self.group_weight = group_weight
        self.formation_weight = formation_weight
        self.member_weight = member_weight
        self.category_weight = category_weight
        self.lithology_weight = lithology_weight

        """Age/Texture can only be one label, while the rest can be multiclass requiring two different loss functions"""
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        total_loss = 0
        type_losses = {}

        age_loss = self.ce_loss(y_pred['age'], y_true['age'])
        type_losses['age'] = age_loss.item()
        total_loss = total_loss + self.age_weight * age_loss

        texture_loss = self.ce_loss(y_pred['texture'], y_true['texture'])
        type_losses['texture'] = texture_loss.items()
        total_loss = total_loss + self.texture_weight * texture_loss

        group_loss = self.bce_loss(y_pred['group'], y_true['group'])
        type_losses['group'] = group_loss.item()
        total_loss = total_loss + self.group_weight * group_loss

        formation_loss = self.bce_loss(y_pred['formation'], y_true['formation'])
        type_losses['formation'] = formation_loss.items()
        total_loss = total_loss + self.formation_weight * formation_loss

        member_loss = self.bce_loss(y_pred['member'], y_true['member'])
        type_losses['member'] = member_loss.items()
        total_loss = total_loss + self.member_weight * member_loss

        category_loss = self.bce_loss(y_pred['category'], y_true['category'])
        type_losses['category'] = category_loss.items()
        total_loss = total_loss + self.category_weight * category_loss

        lithology_loss = self.bce_loss(y_pred['lithology'], y_true['lithology'])
        type_losses['lithology'] = lithology_loss.items()
        total_loss = total_loss + self.lithology_weight * lithology_loss

        return total_loss, type_losses


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

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        loss, _ = loss_func(outputs, y)

        optimizer.zero_grad()
        loss.backward()
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
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = loss_func(outputs, y)

            total_loss, _ = total_loss + loss.item()

            labels = torch.argmax(outputs.data, dim=1)
            correct = correct + (labels == y).sum().item()
            total = total + X.size(0)

    return total_loss/len(test_loader), correct/total

class LayerRNNModel:
    def __init__(self, path):
        self.model = None
        self.pca = None

        self.path = path
        self.loss_func = RNNLoss()

        self.df = None

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X, y = utils.sequence_individual(self.df, relate_id)

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(X)

        print('f')

    def train(self, random_state=0, max_epochs=10, lr=1e-3):
        if self.df is None:
            print("LOADING DATA SET")

            df = Data.load('data.parquet')

            print('ENCODING DATA')
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

        X, y = utils.sequence_layers(self.df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

        print("PERFORMING PCA")
        n_text_cols = 384

        #Claude used to bugfix this line that unpacks holes into their layers again for PCA
        embeddings = np.concatenate(
            [hole[:, -n_text_cols:] for hole in X_train],
            axis=0
        )

        pca = Data.fit_pca(embeddings)
        joblib.dump(pca, f'{self.path}.pca')
        self.pca = pca

        X_train_pca = []

        for hole in X_train:
            non_embedding_cols = hole[:, :-n_text_cols]

            pca_cols = pca.transform(hole[:, -n_text_cols:])

            hole_transformed = np.concatenate([non_embedding_cols, pca_cols], axis=1)
            X_train_pca.append(hole_transformed)

        X_test_pca = []
        for hole in X_test:
            non_embedding_cols = hole[:, :-n_text_cols]

            pca_cols = pca.transform(hole[:, -n_text_cols:])

            hole_transformed = np.concatenate([non_embedding_cols, pca_cols], axis=1)
            X_test_pca.append(hole_transformed)


        train_dataset = LayerDataset(X_train_pca, y_train)
        test_dataset = LayerDataset(X_test_pca, y_test)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=utils.rnn_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=utils.rnn_collate_fn)

        print("INITIATING MODEL")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LayerRNN(X_train_pca[0].shape[1])
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_loss = np.inf

        for epoch in range(max_epochs):
            train_loss = train_loop(train_loader, self.model, device, self.loss_func, optimizer)
            print(f'EPOCH [{epoch+1}|{max_epochs}] : {train_loss:.4f}')

            test_loss, test_acc = test_loop(test_loader, self.model, device, self.loss_func)
            print(f'EPOCH [{epoch+1}|{max_epochs}] : {test_loss:.4f} : {test_acc * 100:.2f}')

            if test_loss < best_loss:
                torch.save(self.model.state_dict(), f'{self.path}.mdl')
                params = {
                    'INPUT_SIZE': self.model.input_size,
                    'HIDDEN_SIZE': self.model.hidden_size,
                    'NUM_LAYERS': self.model.num_layers
                }

                with open(f'{self.path}.json', 'w') as f:
                    json.dump(params, f)