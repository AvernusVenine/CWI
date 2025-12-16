import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import Bedrock
import Precambrian
import Age
import Texture
import config
import xgboost
import cupy
from sklearn.multiclass import OneVsRestClassifier
import os

import Data
from Data import Field
import utils

class GBTModel:
    """
    This encapsulates the Gradient Boosted Trees solution to the data set.  In it, there are currently 4 implemented
    sub-models which can be trained via their respective function.  All the auxiliary files have also been included,
    but there are a lot so all things related to the model itself are included here.
    """
    def __init__(self, path, random_state=0):
        self.random_state = random_state

        self.age_model = None
        self.texture_model = None
        self.group_model = None
        self.formation_model = None
        self.member_model = None
        self.category_model = None
        self.lithology_model = None

        self.pca = None

        self.utme_scaler = None
        self.utmn_scaler = None
        self.elevation_scaler = None
        self.depth_top_scaler = None
        self.depth_bot_scaler = None

        self.path = path

        self.df = None

        self.n_text_cols = 384

        self.load_data()

    def test(self, relate_id):
        if not os.path.isfile(f'{self.path}.mdl'):
            return None
        elif self.age_model is None:
            self.age_model = joblib.load(f'{self.path}.age.mdl')
            self.texture_model = joblib.load(f'{self.path}.txt.mdl')

            self.utme_scaler = joblib.load(f'{self.path}.utme.scl')
            self.utmn_scaler = joblib.load(f'{self.path}.utmn.scl')
            self.elevation_scaler = joblib.load(f'{self.path}.elevation.scl')
            self.depth_top_scaler = joblib.load(f'{self.path}.top.scl')
            self.depth_bot_scaler = joblib.load(f'{self.path}.bot.scl')
            self.pca = joblib.load(f'{self.path}.pca')

        if self.df is None:
            self.df = Data.load('data.parquet')

        X = self.df[self.df[Field.RELATEID] == relate_id.astype(str)]
        X = X.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM])
        X = X.sort_values([Field.DEPTH_BOT, Field.DEPTH_TOP], ascending=[False, False])

        X[Field.UTME] = self.utme_scaler.transform(X[[Field.UTME]])
        X[Field.UTMN] = self.utmn_scaler.transform(X[[Field.UTMN]])
        X[Field.ELEVATION] = self.elevation_scaler.transform(X[[Field.ELEVATION]])
        X[Field.DEPTH_TOP] = self.depth_top_scaler.transform(X[[Field.DEPTH_TOP]])
        X[Field.DEPTH_BOT] = self.depth_bot_scaler.transform(X[[Field.DEPTH_BOT]])

        X[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X[[f"emb_{i}" for i in range(384)]])
        X = X.drop(columns=[f"emb_{i}" for i in range(384)])

        sequential_data = {
            Field.AGE : -1,
            Field.TEXTURE : -1
        }

        codes = []

        for _, layer in X.iterrows():
            layer[Field.PREVIOUS_AGE] = sequential_data[Field.AGE]
            layer[Field.PREVIOUS_TEXTURE] = sequential_data[Field.TEXTURE]

            age = self.age_model.predict(X)
            sequential_data[Field.AGE] = age

            texture = self.texture_model.predict(X)
            sequential_data[Field.TEXTURE] = texture

            pass

    def load_data(self):
        print("LOADING DATA SET")

        df = Data.load('data.parquet')
        df = df.sort_values([Field.RELATEID, Field.DEPTH_TOP])
        self.df = df

        y = df[[Field.STRAT, Field.LITH_PRIM]]
        X = df.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM])

        print('ENCODING DATA')
        y = Age.encode_age(y)
        y = Texture.encode_texture(y)
        y = Bedrock.encode_bedrock(y)
        y = Precambrian.encode_precambrian(y)

        X = utils.encode_hardness(X)
        X = utils.encode_color(X)

        X[Field.UTME] = X[Field.UTME].fillna(X[Field.UTME].min() * .9)
        X[Field.UTMN] = X[Field.UTMN].fillna(X[Field.UTMN].min() * .9)
        X[Field.ELEVATION] = X[Field.ELEVATION].fillna(X[Field.ELEVATION].min() * .8)
        X[Field.DEPTH_BOT] = X[Field.DEPTH_BOT].fillna(-25)
        X[Field.DEPTH_TOP] = X[Field.DEPTH_TOP].fillna(-25)

        y[Field.AGE] = y[Field.AGE].replace(-100, -1)
        y[Field.TEXTURE] = y[Field.TEXTURE].replace(-100, -1)

        """Get the info of previous layer for sequential information"""
        X[Field.PREVIOUS_AGE] = y.groupby(df[Field.RELATEID])[Field.AGE].shift(-1).fillna(-1)
        X[Field.PREVIOUS_TEXTURE] = y.groupby(df[Field.RELATEID])[Field.TEXTURE].shift(-1).fillna(-1)
        X[[f'prev_{group.name}' for group in Bedrock.GROUP_LIST]] = (y.groupby(df[Field.RELATEID])[[group.name for group in Bedrock.GROUP_LIST]]
                                                                .shift(-1).fillna(0))
        X[[f'prev_{form.name}' for form in Bedrock.FORMATION_LIST]] = (y.groupby(df[Field.RELATEID])[[form.name for form in Bedrock.FORMATION_LIST]]
                                                                  .shift(-1).fillna(0))
        X[[f'prev_{member.name}' for member in Bedrock.MEMBER_LIST]] = (y.groupby(df[Field.RELATEID])[[member.name for member in Bedrock.MEMBER_LIST]]
                                                                  .shift(-1).fillna(0))
        X[[f'prev_{cat.name}' for cat in Precambrian.CATEGORY_LIST]] = (y.groupby(df[Field.RELATEID])[[cat.name for cat in Precambrian.CATEGORY_LIST]]
                                                                   .shift(-1).fillna(0))
        X[[f'prev_{lith.name}' for lith in Precambrian.LITHOLOGY_LIST]] = (y.groupby(df[Field.RELATEID])[[lith.name for lith in Precambrian.LITHOLOGY_LIST]]
            .shift(-1).fillna(0))

        mask = y[Field.AGE] != -1
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

        print('FITTING SCALERS')

        self.utme_scaler = StandardScaler()
        self.utme_scaler.fit(X_train[[Field.UTME]])

        self.utmn_scaler = StandardScaler()
        self.utmn_scaler.fit(X_train[[Field.UTMN]])

        self.elevation_scaler = StandardScaler()
        self.elevation_scaler.fit(X_train[[Field.ELEVATION]])

        self.depth_top_scaler = StandardScaler()
        self.depth_top_scaler.fit(X_train[[Field.DEPTH_TOP]])

        self.depth_bot_scaler = StandardScaler()
        self.depth_bot_scaler.fit(X_train[[Field.DEPTH_BOT]])

        print('FITTING PCA')

        self.pca = PCA(n_components=50)
        self.pca.fit(X_train[[f"emb_{i}" for i in range(384)]])

        joblib.dump(self.utme_scaler, f'{self.path}.utme.scl')
        joblib.dump(self.utmn_scaler, f'{self.path}.utmn.scl')
        joblib.dump(self.elevation_scaler, f'{self.path}.elevation.scl')
        joblib.dump(self.depth_top_scaler, f'{self.path}.top.scl')
        joblib.dump(self.depth_bot_scaler, f'{self.path}.bot.scl')
        joblib.dump(self.pca, f'{self.path}.pca')

        print('APPLYING DATA REFINEMENTS')
        X_train[Field.UTME] = self.utme_scaler.transform(X_train[[Field.UTME]])
        X_test[Field.UTME] = self.utme_scaler.transform(X_test[[Field.UTME]])

        X_train[Field.UTMN] = self.utmn_scaler.transform(X_train[[Field.UTMN]])
        X_test[Field.UTMN] = self.utmn_scaler.transform(X_test[[Field.UTMN]])

        X_train[Field.ELEVATION] = self.elevation_scaler.transform(X_train[[Field.ELEVATION]])
        X_test[Field.ELEVATION] = self.elevation_scaler.transform(X_test[[Field.ELEVATION]])

        X_train[Field.DEPTH_BOT] = self.depth_bot_scaler.transform(X_train[[Field.DEPTH_BOT]])
        X_test[Field.DEPTH_BOT] = self.depth_bot_scaler.transform(X_test[[Field.DEPTH_BOT]])

        X_train[Field.DEPTH_TOP] = self.depth_top_scaler.transform(X_train[[Field.DEPTH_TOP]])
        X_test[Field.DEPTH_TOP] = self.depth_top_scaler.transform(X_test[[Field.DEPTH_TOP]])

        X_train[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X_train[[f"emb_{i}" for i in range(384)]])
        X_test[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X_test[[f"emb_{i}" for i in range(384)]])

        X_train = X_train.drop(columns=[f"emb_{i}" for i in range(384)])
        X_test = X_test.drop(columns=[f"emb_{i}" for i in range(384)])

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_age(self, n_estimators=100):
        print('BALANCING DATA SET')
        encoder = Age.init_encoder()

        X_train = self.X_train.copy()
        y_train = self.y_train.copy()

        drop_dict = {
            'Q': .25
        }

        for age, percentage in drop_dict.items():
            mask = y_train[Field.AGE] == encoder.transform([age])[0]
            indices = y_train[mask].index

            """Necessary to recreate random conditions"""
            np.random.seed(self.random_state)
            kept_indices = np.random.choice(indices, size=int(len(indices) * percentage))

            mask = ~mask | y_train.index.isin(kept_indices)

            X_train = X_train[mask].reset_index(drop=True)
            y_train = y_train[mask].reset_index(drop=True)

        encoder = Age.init_encoder()

        X_train, y_train = utils.SMOTE_gbt(X_train, y_train, encoder.transform(['B'])[0], 4000, Field.AGE)
        X_train, y_train = utils.SMOTE_gbt(X_train, y_train, encoder.transform(['X'])[0], 4000, Field.AGE)
        X_train, y_train = utils.SMOTE_gbt(X_train, y_train, encoder.transform(['Y'])[0], 4000, Field.AGE)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            verbosity=1,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train[Field.AGE])

        joblib.dump(model, f'{self.path}.age.mdl')
        self.age_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.age_model.predict(self.X_test)

        print('Accuracy: ', accuracy_score(self.y_test[Field.AGE].tolist(), y_pred))
        print(classification_report(self.y_test[Field.AGE].tolist(), y_pred, zero_division=0))

        report = classification_report(
            self.y_test[Field.AGE].tolist(),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        return report['macro avg']['f1-score'], report['accuracy']

    def train_texture(self, n_estimators=100):
        print('BALANCING DATA SET')
        encoder = Age.init_encoder()

        X_train = self.X_train.copy()
        y_train = self.y_train.copy()

        """Only want Quaternary/Recent values for training"""
        mask = y_train[Field.AGE].isin(encoder.transform(['Q', 'R']))

        X_train = X_train[mask].reset_index(drop=True)
        y_train = y_train[mask].reset_index(drop=True)

        encoder = Texture.init_encoder()

        mask = y_train[Field.TEXTURE] != -100

        X_train = X_train[mask].reset_index(drop=True)
        y_train = y_train[mask].reset_index(drop=True)

        mask = y_train[Field.TEXTURE] != -1

        X_train = X_train[mask].reset_index(drop=True)
        y_train = y_train[mask].reset_index(drop=True)

        """Test data filtering"""

        X_test = self.X_test.copy()
        y_test = self.y_test.copy()

        mask = y_test[Field.TEXTURE] != -1

        X_test = X_test[mask].reset_index(drop=True)
        y_test = y_test[mask].reset_index(drop=True)

        drop_dict = {
            'C': .33,
            'F': .33,
        }

        for texture, percentage in drop_dict.items():
            mask = y_train[Field.TEXTURE] == encoder.transform([texture])[0]
            indices = y_train[mask].index

            """Necessary to recreate random conditions"""
            np.random.seed(self.random_state)
            kept_indices = np.random.choice(indices, size=int(len(indices) * percentage))

            mask = ~mask | y_train.index.isin(kept_indices)

            X_train = X_train[mask].reset_index(drop=True)
            y_train = y_train[mask].reset_index(drop=True)

        X_train, y_train = utils.SMOTE_gbt(X_train, y_train, encoder.transform(['S'])[0], 8000, Field.TEXTURE)
        X_train, y_train = utils.SMOTE_gbt(X_train, y_train, encoder.transform(['I'])[0], 5000, Field.TEXTURE)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            verbosity=1,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train[Field.TEXTURE])

        joblib.dump(model, f'{self.path}.txt.mdl')
        self.texture_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.texture_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.TEXTURE].tolist(), y_pred))

        report = classification_report(
            y_test[Field.TEXTURE].tolist(),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        print(classification_report(y_test[Field.TEXTURE].tolist(), y_pred, zero_division=0))

        return report['macro avg']['f1-score'], report['accuracy']

    def train_group(self, n_estimators=100):
        print('BALANCING DATA SET')
        encoder = Age.init_encoder()

        X_train = self.X_train.copy()
        y_train = self.y_train.copy()

        """Only want Bedrock values for training"""
        mask = y_train[Field.AGE].isin(encoder.transform(['C', 'D', 'K', 'O', 'G']))

        X_train = X_train[mask].reset_index(drop=True)
        y_train = y_train[mask].reset_index(drop=True)

        """Test data filtering"""

        X_test = self.X_test.copy()
        y_test = self.y_test.copy()

        mask = y_test[Field.AGE].isin(encoder.transform(['C', 'D', 'K', 'O', 'G']))

        X_test = X_test[mask].reset_index(drop=True)
        y_test = y_test[mask].reset_index(drop=True)

        print('TRAINING MODEL')

        binary_model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            verbosity=1,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )

        model = OneVsRestClassifier(binary_model, verbose=1)
        model.fit(X_train, y_train[[group.name for group in Bedrock.GROUP_LIST]])

        joblib.dump(model, f'{self.path}.grp.mdl')
        self.group_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.group_model.predict(X_test)

        print('Accuracy: ', accuracy_score(np.array(y_test[[group.name for group in Bedrock.GROUP_LIST]]), y_pred))

        report = classification_report(
            np.array(y_test[[group.name for group in Bedrock.GROUP_LIST]]),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        print(classification_report(np.array(y_test[[group.name for group in Bedrock.GROUP_LIST]]), y_pred, zero_division=0))

        return report['macro avg']['f1-score'], accuracy_score(np.array(y_test[[group.name for group in Bedrock.GROUP_LIST]]), y_pred)

    def train_formation(self, n_estimators=100):
        print('BALANCING DATA SET')
        encoder = Age.init_encoder()

        X_train = self.X_train.copy()
        y_train = self.y_train.copy()

        """Only want Bedrock values for training"""
        mask = y_train[Field.AGE].isin(encoder.transform(['C', 'D', 'K', 'O', 'G']))

        X_train = X_train[mask].reset_index(drop=True)
        y_train = y_train[mask].reset_index(drop=True)

        """Test data filtering"""

        X_test = self.X_test.copy()
        y_test = self.y_test.copy()

        mask = y_test[Field.AGE].isin(encoder.transform(['C', 'D', 'K', 'O', 'G']))

        X_test = X_test[mask].reset_index(drop=True)
        y_test = y_test[mask].reset_index(drop=True)

        print('TRAINING MODEL')

        binary_model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            verbosity=1,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )

        model = OneVsRestClassifier(binary_model, verbose=1)
        model.fit(X_train, y_train[[form.name for form in Bedrock.FORMATION_LIST]])

        joblib.dump(model, f'{self.path}.frm.mdl')
        self.group_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.group_model.predict(X_test)

        print('Accuracy: ', accuracy_score(np.array(y_test[[form.name for form in Bedrock.FORMATION_LIST]]), y_pred))

        report = classification_report(
            np.array(y_test[[form.name for form in Bedrock.FORMATION_LIST]]),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        print(classification_report(np.array(y_test[[form.name for form in Bedrock.FORMATION_LIST]]), y_pred, zero_division=0))

        return report['macro avg']['f1-score'], accuracy_score(np.array(y_test[[form.name for form in Bedrock.FORMATION_LIST]]), y_pred)