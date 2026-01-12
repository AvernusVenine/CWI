import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

import Bedrock
import Precambrian
import Age
import Texture
import xgboost
from SyntheticData import create_shallow, create_bedrock
import Data
from Data import Field
import utils

class GBTModel:
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
        self.depth_scaler = None

        self.path = path

        self.df = None

        self.n_text_cols = 384

        self.X_train = None

    def test_database(self, start=0):
        warnings.filterwarnings('ignore')

        if self.age_model is None or self.texture_model is None:
            self.age_model = joblib.load(f'{self.path}.age.mdl')
            self.age_model.set_params(device='cpu')
            self.texture_model = joblib.load(f'{self.path}.txt.mdl')
            self.texture_model.set_params(device='cpu')

            self.utme_scaler = joblib.load(f'{self.path}.utme.scl')
            self.utmn_scaler = joblib.load(f'{self.path}.utmn.scl')
            self.elevation_scaler = joblib.load(f'{self.path}.elevation.scl')
            self.depth_scaler = joblib.load(f'{self.path}.depth.scl')
            self.pca = joblib.load(f'{self.path}.pca')

        df = Data.load('data.parquet')

        df = df.sort_values([Field.RELATEID, Field.DEPTH_TOP], ascending=[True, False])
        y = df[Field.STRAT]
        X = df.drop(columns=[Field.STRAT, Field.LITH_PRIM])

        X[Field.ELEVATION_TOP] = X[Field.ELEVATION] - X[Field.DEPTH_TOP]
        X[Field.ELEVATION_BOT] = X[Field.ELEVATION] - X[Field.DEPTH_BOT]
        X = X.drop(columns=[Field.ELEVATION])

        X[Field.UTME] = self.utme_scaler.transform(X[Field.UTME].values.reshape(-1, 1)).ravel()
        X[Field.UTMN] = self.utmn_scaler.transform(X[Field.UTMN].values.reshape(-1, 1)).ravel()
        X[Field.ELEVATION_TOP] = self.elevation_scaler.transform(X[Field.ELEVATION_TOP].values.reshape(-1, 1)).ravel()
        X[Field.ELEVATION_BOT] = self.elevation_scaler.transform(X[Field.ELEVATION_BOT].values.reshape(-1, 1)).ravel()
        X[Field.DEPTH_TOP] = self.depth_scaler.transform(X[Field.DEPTH_TOP].values.reshape(-1, 1)).ravel()
        X[Field.DEPTH_BOT] = self.depth_scaler.transform(X[Field.DEPTH_BOT].values.reshape(-1, 1)).ravel()

        X = utils.encode_color(X)
        X = utils.encode_hardness(X)

        X[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X[[f"emb_{i}" for i in range(384)]])
        X = X.drop(columns=[f"emb_{i}" for i in range(384)])

        age_encoder = Age.init_encoder()
        texture_encoder = Texture.init_encoder()
        group_encoder, formation_encoder, member_encoder = Bedrock.init_encoders()

        age_cols = joblib.load(f'{self.path}.age.fts')
        texture_cols = joblib.load(f'{self.path}.txt.fts')

        err_df = pd.DataFrame(columns=['relateid', 'index', 'true', 'prediction'])

        for relate_id in df[Field.RELATEID].unique().tolist():

            if relate_id < start:
                continue

            mask = X[Field.RELATEID] == relate_id

            well = X[mask].drop(columns=[Field.RELATEID])

            old_codes = y[mask].values.tolist()

            print(old_codes)

            sequential_data = {
                Field.AGE: -1,
                Field.TEXTURE: -1,
                Field.GROUP_BOT: group_encoder.transform([None])[0],
                Field.FORMATION_BOT: formation_encoder.transform([None])[0],
                Field.MEMBER_BOT: member_encoder.transform([None])[0],
            }

            idx = 0

            for _, layer in well.iterrows():

                layer[Field.PREVIOUS_AGE] = sequential_data[Field.AGE]
                layer[Field.PREVIOUS_TEXTURE] = sequential_data[Field.TEXTURE]
                layer[Field.PREVIOUS_GROUP] = sequential_data[Field.GROUP_BOT]
                layer[Field.PREVIOUS_FORMATION] = sequential_data[Field.FORMATION_BOT]

                layer = layer.astype(float)
                layer = layer.to_frame().T

                label_dict = {}

                age_X = layer[age_cols]

                age = self.age_model.predict(age_X)
                sequential_data[Field.AGE] = age

                label_dict['age'] = age

                if age_encoder.inverse_transform([age])[0] in ('Q', 'R'):
                    texture_X = layer[texture_cols]

                    texture = self.texture_model.predict(texture_X)
                    sequential_data[Field.TEXTURE] = texture

                    label_dict['texture'] = texture

                    colors = ['BLACK', 'BROWN', 'BLUE', 'GREEN', 'ORANGE', 'PINK', 'PURPLE', 'RED', 'GRAY', 'WHITE',
                              'YELLOW', 'VARIED']

                    for color in colors:
                        label_dict[color] = layer[color].values[0]

                    if age_encoder.inverse_transform([age])[0] != old_codes[idx][0]:
                        temp = pd.DataFrame({
                            'relateid': [relate_id],
                            'index': [idx],
                            'true': [old_codes[idx]],
                            'prediction': [utils.compile_geocode(label_dict)]
                        })

                        err_df = pd.concat([err_df, temp])
                    elif texture_encoder.inverse_transform([texture])[0] != old_codes[idx][1]:
                        if age_encoder.inverse_transform([age])[0] != old_codes[idx][0]:
                            temp = pd.DataFrame({
                                'relateid': [relate_id],
                                'index': [idx],
                                'true': [old_codes[idx]],
                                'prediction': [utils.compile_geocode(label_dict)]
                            })

                            err_df = pd.concat([err_df, temp])

                elif age_encoder.inverse_transform([age])[0] in ('F', 'B', 'U'):
                    if utils.compile_geocode(label_dict) != old_codes[idx]:
                        temp = pd.DataFrame({
                            'relateid': [relate_id],
                            'index': [idx],
                            'true': [old_codes[idx]],
                            'prediction': [utils.compile_geocode(label_dict)]
                        })

                        err_df = pd.concat([err_df, temp])
                else:
                    sequential_data[Field.TEXTURE] = -1

                    age_code = age_encoder.inverse_transform([age])[0]
                    if age_code in ('A', 'E', 'M', 'G'):
                        age_code = 'P'

                    if age_code != old_codes[idx][0]:
                        temp = pd.DataFrame({
                            'relateid': [relate_id],
                            'index': [idx],
                            'true': [old_codes[idx]],
                            'prediction': [age_encoder.inverse_transform([age])[0]]
                        })

                        err_df = pd.concat([err_df, temp])

                idx += 1

            err_df.to_csv('database_comparison.csv')

    def test(self, relate_ids, unlabelled=False):
        warnings.filterwarnings('ignore')

        if self.age_model is None or self.texture_model is None:
            self.age_model = joblib.load(f'{self.path}.age.mdl')
            self.age_model.set_params(device='cpu')
            self.texture_model = joblib.load(f'{self.path}.txt.mdl')
            self.texture_model.set_params(device='cpu')

            self.utme_scaler = joblib.load(f'{self.path}.utme.scl')
            self.utmn_scaler = joblib.load(f'{self.path}.utmn.scl')
            self.elevation_scaler = joblib.load(f'{self.path}.elevation.scl')
            self.depth_scaler = joblib.load(f'{self.path}.depth.scl')
            self.pca = joblib.load(f'{self.path}.pca')

        if unlabelled:
            df = Data.load('unlabelled.parquet')
        else:
            df = Data.load('data.parquet')

        df = df.sort_values([Field.RELATEID, Field.DEPTH_TOP], ascending=[False, False])
        y = df[Field.STRAT]
        X = df.drop(columns=[Field.STRAT, Field.LITH_PRIM])

        X[Field.ELEVATION_TOP] = X[Field.ELEVATION] - X[Field.DEPTH_TOP]
        X[Field.ELEVATION_BOT] = X[Field.ELEVATION] - X[Field.DEPTH_BOT]
        X = X.drop(columns=[Field.ELEVATION])

        X[Field.UTME] = self.utme_scaler.transform(X[Field.UTME].values.reshape(-1, 1)).ravel()
        X[Field.UTMN] = self.utmn_scaler.transform(X[Field.UTMN].values.reshape(-1, 1)).ravel()
        X[Field.ELEVATION_TOP] = self.elevation_scaler.transform(X[Field.ELEVATION_TOP].values.reshape(-1, 1)).ravel()
        X[Field.ELEVATION_BOT] = self.elevation_scaler.transform(X[Field.ELEVATION_BOT].values.reshape(-1, 1)).ravel()
        X[Field.DEPTH_TOP] = self.depth_scaler.transform(X[Field.DEPTH_TOP].values.reshape(-1, 1)).ravel()
        X[Field.DEPTH_BOT] = self.depth_scaler.transform(X[Field.DEPTH_BOT].values.reshape(-1, 1)).ravel()

        X = utils.encode_color(X)
        X = utils.encode_hardness(X)

        X[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X[[f"emb_{i}" for i in range(384)]])
        X = X.drop(columns=[f"emb_{i}" for i in range(384)])

        age_encoder = Age.init_encoder()
        texture_encoder = Texture.init_encoder()
        group_encoder, formation_encoder, member_encoder = Bedrock.init_encoders()

        age_cols = joblib.load(f'{self.path}.age.fts')
        texture_cols = joblib.load(f'{self.path}.txt.fts')

        old_wells = {}
        new_wells = {}
        probs = {}

        for relate_id in relate_ids:
            mask = X[Field.RELATEID] == relate_id

            well = X[mask].drop(columns=[Field.RELATEID])

            old_codes = y[mask].values.tolist()
            old_codes.reverse()
            old_wells[relate_id] = old_codes

            sequential_data = {
                Field.AGE: -1,
                Field.TEXTURE: -1,
                Field.GROUP_BOT: group_encoder.transform([None])[0],
                Field.FORMATION_BOT: formation_encoder.transform([None])[0],
                Field.MEMBER_BOT: member_encoder.transform([None])[0],
            }

            for _, layer in well.iterrows():
                layer[Field.PREVIOUS_AGE] = sequential_data[Field.AGE]
                layer[Field.PREVIOUS_TEXTURE] = sequential_data[Field.TEXTURE]
                layer[Field.PREVIOUS_GROUP] = sequential_data[Field.GROUP_BOT]
                layer[Field.PREVIOUS_FORMATION] = sequential_data[Field.FORMATION_BOT]

                layer = layer.astype(float)
                layer = layer.to_frame().T

                label_dict = {}

                age_X = layer[age_cols]

                age = self.age_model.predict(age_X)
                sequential_data[Field.AGE] = age

                label_dict['age'] = age

                age_prob = self.age_model.predict_proba(age_X)[0]
                top3_idx = np.argsort(age_prob)[-3:][::-1]
                top3_probs = age_prob[top3_idx]
                top3_classes = age_encoder.inverse_transform(self.age_model.classes_[top3_idx])

                age_prob = list(zip(top3_classes, top3_probs))
                age_prob.reverse()

                if relate_id not in probs.keys():
                    probs[relate_id] = []

                probs[relate_id].append(age_prob)

                if age_encoder.inverse_transform([age])[0] in ('Q', 'R'):
                    texture_X = layer[texture_cols]

                    texture = self.texture_model.predict(texture_X)
                    sequential_data[Field.TEXTURE] = texture

                    label_dict['texture'] = texture

                    colors = ['BLACK', 'BROWN', 'BLUE', 'GREEN', 'ORANGE', 'PINK', 'PURPLE', 'RED', 'GRAY', 'WHITE',
                              'YELLOW', 'VARIED']

                    for color in colors:
                        label_dict[color] = layer[color].values[0]

                    if relate_id not in new_wells.keys():
                        new_wells[relate_id] = []

                    new_wells[relate_id].append(utils.compile_geocode(label_dict))
                elif age_encoder.inverse_transform([age])[0] in ('F', 'B', 'U'):
                    if relate_id not in new_wells.keys():
                        new_wells[relate_id] = []

                    new_wells[relate_id].append(utils.compile_geocode(label_dict))
                else:
                    sequential_data[Field.TEXTURE] = -1

                    if relate_id not in new_wells.keys():
                        new_wells[relate_id] = []
                    new_wells[relate_id].append(age_encoder.inverse_transform([age])[0])

            if relate_id in new_wells.keys():
                new_wells[relate_id].reverse()

        return old_wells, new_wells, probs

    def load_and_split_data(self):
        """
        Loads then splits the data into its top and bottom components for bedrock classification
        :return:
        """
        print("LOADING DATA SET")

        df = Data.load('weighted.parquet')

        print('ENCODING DATA')
        df = utils.encode_color(df)
        df = utils.encode_hardness(df)
        df = utils.encode_weights(df)

        df[Field.LENGTH] = df[Field.DEPTH_BOT] - df[Field.DEPTH_TOP]

        df_top = df.drop(columns=[Field.DEPTH_BOT]).rename(columns={Field.DEPTH_TOP: Field.DEPTH})
        df_top[Field.ORDER] = 0
        df_bot = df.drop(columns=[Field.DEPTH_TOP]).rename(columns={Field.DEPTH_BOT: Field.DEPTH})
        df_bot[Field.ORDER] = 1

        df = pd.concat([df_top, df_bot])

        df = df.sort_values([Field.RELATEID, Field.DEPTH, Field.ORDER])
        df = df[df[Field.STRAT] != 'BSMT']
        self.df = df

        y = df[[Field.RELATEID, Field.STRAT, Field.LITH_PRIM, Field.ORDER]]
        X = df.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM, Field.ORDER])

        X[Field.UTME] = X[Field.UTME].fillna(X[Field.UTME].min() * .9)
        X[Field.UTMN] = X[Field.UTMN].fillna(X[Field.UTMN].min() * .9)
        X[Field.ELEVATION] = X[Field.ELEVATION].fillna(X[Field.ELEVATION].min() * .8)
        X[Field.DEPTH] = X[Field.DEPTH].fillna(-25)

        """Create a depth based elevation feature to replace elevation"""
        X[Field.ELEVATION] = X[Field.ELEVATION] - X[Field.DEPTH]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

        print('FITTING SCALERS')

        self.utme_scaler = StandardScaler()
        self.utme_scaler.fit(X_train[[Field.UTME]])

        self.utmn_scaler = StandardScaler()
        self.utmn_scaler.fit(X_train[[Field.UTMN]])

        self.elevation_scaler = StandardScaler()
        self.elevation_scaler.fit(X_train[[Field.ELEVATION]].values.tolist())

        self.depth_scaler = StandardScaler()
        self.depth_scaler.fit(X_train[[Field.DEPTH]].values.tolist())

        print('FITTING PCA')

        self.pca = PCA(n_components=50)
        self.pca.fit(X_train[[f"emb_{i}" for i in range(384)]])

        joblib.dump(self.utme_scaler, f'{self.path}.utme.scl')
        joblib.dump(self.utmn_scaler, f'{self.path}.utmn.scl')
        joblib.dump(self.elevation_scaler, f'{self.path}.elevation.scl')
        joblib.dump(self.depth_scaler, f'{self.path}.depth.scl')
        joblib.dump(self.pca, f'{self.path}.pca')

        print('APPLYING DATA REFINEMENTS')
        X[Field.UTME] = self.utme_scaler.transform(X[[Field.UTME]])
        X[Field.UTMN] = self.utmn_scaler.transform(X[[Field.UTMN]])
        X[Field.ELEVATION] = self.elevation_scaler.transform(X[[Field.ELEVATION]])
        X[Field.DEPTH] = self.depth_scaler.transform(X[[Field.DEPTH]])

        X[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X[[f"emb_{i}" for i in range(384)]])
        X = X.drop(columns=[f"emb_{i}" for i in range(384)])

        X = X.astype(float)

        return train_test_split(X, y, random_state=self.random_state)

    def load_data(self):
        """
        Loads the data set for training and testing
        :return:
        """
        print("LOADING DATA SET")

        df = Data.load('weighted.parquet')

        df = df.sort_values([Field.RELATEID, Field.DEPTH_TOP])
        df = df[df[Field.STRAT] != 'BSMT'] #TODO: Fully drop this label at some point
        self.df = df

        df = Age.encode_type(df)

        y = df[[Field.RELATEID, Field.STRAT, Field.LITH_PRIM, Field.AGE, Field.TYPE]]
        X = df.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM, Field.AGE, Field.TYPE])

        print('ENCODING DATA')
        X = utils.encode_hardness(X)
        X = utils.encode_color(X)

        X = utils.encode_weights(X)

        X[Field.UTME] = X[Field.UTME].fillna(X[Field.UTME].min() * .9)
        X[Field.UTMN] = X[Field.UTMN].fillna(X[Field.UTMN].min() * .9)
        X[Field.ELEVATION] = X[Field.ELEVATION].fillna(X[Field.ELEVATION].min() * .8)
        X[Field.DEPTH_BOT] = X[Field.DEPTH_BOT].fillna(-25)
        X[Field.DEPTH_TOP] = X[Field.DEPTH_TOP].fillna(-25)

        """Create elevation top and elevation bot features to replace elevation"""
        X[Field.ELEVATION_TOP] = X[Field.ELEVATION] - X[Field.DEPTH_TOP]
        X[Field.ELEVATION_BOT] = X[Field.ELEVATION] - X[Field.DEPTH_BOT]

        X = X.drop(columns=[Field.ELEVATION])

        """Get the info of previous layer for sequential information"""
        X[Field.PREVIOUS_AGE] = y.groupby(df[Field.RELATEID])[Field.AGE].shift(-1).fillna(-1)
        #TODO: Possibly add back other sequential features but just age for now

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
        self.elevation_scaler.fit(X_train[[Field.ELEVATION_TOP]].values.tolist()
                                  + X_train[[Field.ELEVATION_BOT]].values.tolist())

        self.depth_scaler = StandardScaler()
        self.depth_scaler.fit(X_train[[Field.DEPTH_TOP]].values.tolist()
                                  + X_train[[Field.DEPTH_BOT]].values.tolist())

        print('FITTING PCA')

        self.pca = PCA(n_components=50)
        self.pca.fit(X_train[[f"emb_{i}" for i in range(384)]])

        joblib.dump(self.utme_scaler, f'{self.path}.utme.scl')
        joblib.dump(self.utmn_scaler, f'{self.path}.utmn.scl')
        joblib.dump(self.elevation_scaler, f'{self.path}.elevation.scl')
        joblib.dump(self.depth_scaler, f'{self.path}.depth.scl')
        joblib.dump(self.pca, f'{self.path}.pca')

        print('APPLYING DATA REFINEMENTS')
        X_train[Field.UTME] = self.utme_scaler.transform(X_train[[Field.UTME]])
        X_test[Field.UTME] = self.utme_scaler.transform(X_test[[Field.UTME]])

        X_train[Field.UTMN] = self.utmn_scaler.transform(X_train[[Field.UTMN]])
        X_test[Field.UTMN] = self.utmn_scaler.transform(X_test[[Field.UTMN]])

        X_train[Field.ELEVATION_TOP] = self.elevation_scaler.transform(X_train[[Field.ELEVATION_TOP]])
        X_test[Field.ELEVATION_TOP] = self.elevation_scaler.transform(X_test[[Field.ELEVATION_TOP]])

        X_train[Field.ELEVATION_BOT] = self.elevation_scaler.transform(X_train[[Field.ELEVATION_BOT]])
        X_test[Field.ELEVATION_BOT] = self.elevation_scaler.transform(X_test[[Field.ELEVATION_BOT]])

        X_train[Field.DEPTH_BOT] = self.depth_scaler.transform(X_train[[Field.DEPTH_BOT]])
        X_test[Field.DEPTH_BOT] = self.depth_scaler.transform(X_test[[Field.DEPTH_BOT]])

        X_train[Field.DEPTH_TOP] = self.depth_scaler.transform(X_train[[Field.DEPTH_TOP]])
        X_test[Field.DEPTH_TOP] = self.depth_scaler.transform(X_test[[Field.DEPTH_TOP]])

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

        return X_train, X_test, y_train, y_test

    def train_type(self, data=None, n_estimators=100, depth=4, drop=.1, eta=.3):
        """
        Differentiates between Recent and Bedrock layers
        """
        if data is None:
            X_train, X_test, y_train, y_test = self.load_data()
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

        print('BALANCING DATA SET')

        encoder = Age.init_type_encoder()

        drop_dict = {
            'Quaternary': .25
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

        X_train, y_train = create_shallow(X_train, y_train, encoder.transform(['Pitt'])[0], 4000, Field.TYPE)
        X_train, y_train = create_shallow(X_train, y_train, encoder.transform(['Pavement'])[0], 4000, Field.TYPE)

        print("APPLYING WEIGHTS")

        weights = X_train[Field.INTERPRETATION_METHOD].values.tolist()

        X_train = X_train.drop(columns=[Field.INTERPRETATION_METHOD])
        X_test = X_test.drop(columns=[Field.INTERPRETATION_METHOD])

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=self.random_state)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            max_depth=depth,
            eta=eta,
            verbosity=1,
            device='cuda',

            rate_drop=drop,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train[Field.TYPE], eval_set=[(X_val, y_val[Field.TYPE])], sample_weight=weights, verbose=True)

        joblib.dump(model, f'{self.path}.type.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.type.fts')
        self.type_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.type_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.TYPE].tolist(), y_pred))
        print(classification_report(y_test[Field.TYPE].tolist(), y_pred, zero_division=0))

        report = classification_report(
            y_test[Field.TYPE].tolist(),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        return report['macro avg']['f1-score'], report['accuracy']

    def load_lithology(self):
        warnings.filterwarnings('ignore')

        print("LOADING DATA SET")

        df = Data.load('weighted.parquet')

        df = df.sort_values([Field.RELATEID, Field.DEPTH_TOP, Field.DEPTH_BOT])

        df = df[df['strat'].astype(str).str[0].isin(['R', 'Q'])]

        lith_prim = df['lith_prim'].value_counts()
        lith_prim = lith_prim[lith_prim > 20].index.tolist()
        df = df[df['lith_prim'].isin(lith_prim)]

        encoder = LabelEncoder()
        encoder.fit(lith_prim)

        df['lith_prim'] = encoder.transform(df[['lith_prim']])

        df = df.drop(columns=[Field.LITH_SEC, Field.LITH_MINOR])

        df[Field.THICKNESS] = df[Field.DEPTH_BOT] - df[Field.DEPTH_TOP]

        y = df[[Field.RELATEID, Field.STRAT, Field.LITH_PRIM]]
        X = df.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM])

        X = utils.encode_color(X)
        X = utils.encode_hardness(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

        print('FITTING SCALERS')

        self.utme_scaler = StandardScaler()
        self.utme_scaler.fit(X_train[[Field.UTME]])

        self.utmn_scaler = StandardScaler()
        self.utmn_scaler.fit(X_train[[Field.UTMN]])

        self.depth_scaler = StandardScaler()
        self.depth_scaler.fit(X_train[[Field.DEPTH_TOP]].values.tolist()
                                  + X_train[[Field.DEPTH_BOT]].values.tolist())

        print('FITTING PCA')

        self.pca = PCA(n_components=50)
        self.pca.fit(X_train[[f"emb_{i}" for i in range(384)]])

        joblib.dump(self.utme_scaler, f'{self.path}.utme.scl')
        joblib.dump(self.utmn_scaler, f'{self.path}.utmn.scl')
        joblib.dump(self.depth_scaler, f'{self.path}.depth.scl')
        joblib.dump(self.pca, f'{self.path}.pca')

        print('APPLYING DATA REFINEMENTS')
        X_train[Field.UTME] = self.utme_scaler.transform(X_train[[Field.UTME]])
        X_test[Field.UTME] = self.utme_scaler.transform(X_test[[Field.UTME]])

        X_train[Field.UTMN] = self.utmn_scaler.transform(X_train[[Field.UTMN]])
        X_test[Field.UTMN] = self.utmn_scaler.transform(X_test[[Field.UTMN]])

        X_train[Field.DEPTH_BOT] = self.depth_scaler.transform(X_train[[Field.DEPTH_BOT]])
        X_test[Field.DEPTH_BOT] = self.depth_scaler.transform(X_test[[Field.DEPTH_BOT]])

        X_train[Field.DEPTH_TOP] = self.depth_scaler.transform(X_train[[Field.DEPTH_TOP]])
        X_test[Field.DEPTH_TOP] = self.depth_scaler.transform(X_test[[Field.DEPTH_TOP]])

        X_train[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X_train[[f"emb_{i}" for i in range(384)]])
        X_test[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X_test[[f"emb_{i}" for i in range(384)]])

        X_train = X_train.drop(columns=[f"emb_{i}" for i in range(384)])
        X_test = X_test.drop(columns=[f"emb_{i}" for i in range(384)])

        return X_train, X_test, y_train, y_test, encoder

    def train_lithology(self, data=None, n_estimators=100, depth=4, drop=.1, eta=.3):

        if data is None:
            X_train, X_test, y_train, y_test, encoder = self.load_lithology()
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]
            encoder = data[4]

        X_train = X_train.drop(columns=[Field.INTERPRETATION_METHOD, Field.UTME, Field.DEPTH_BOT, Field.UTMN] + Data.COLORS)
        X_test = X_test.drop(columns=[Field.INTERPRETATION_METHOD, Field.UTME, Field.DEPTH_BOT, Field.UTMN] + Data.COLORS)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=self.random_state)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            max_depth=depth,
            eta=eta,
            verbosity=1,
            device='cuda',

            rate_drop=drop,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train[Field.LITH_PRIM], eval_set=[(X_val, y_val[Field.LITH_PRIM])], verbose=True)

        joblib.dump(model, f'{self.path}.ltp.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.ltp.fts')
        self.lith_prim_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.lith_prim_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.LITH_PRIM].tolist(), y_pred))
        print(classification_report(encoder.inverse_transform(y_test[Field.LITH_PRIM].tolist()), encoder.inverse_transform(y_pred), zero_division=0))

    def load_texture(self):
        print("LOADING DATA SET")

        df = Data.load('weighted.parquet')

        df = df.sort_values([Field.RELATEID, Field.DEPTH_TOP])

        df = Texture.encode_texture(df)

        y = df[[Field.RELATEID, Field.STRAT, Field.LITH_PRIM, Field.TEXTURE]]
        X = df.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM, Field.TEXTURE])

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

        print('FITTING PCA')

        self.pca = PCA(n_components=50)
        self.pca.fit(X_train[[f"emb_{i}" for i in range(384)]])

        X_train[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X_train[[f"emb_{i}" for i in range(384)]])
        X_test[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X_test[[f"emb_{i}" for i in range(384)]])

        X_train = X_train.drop(columns=[f"emb_{i}" for i in range(384)])
        X_test = X_test.drop(columns=[f"emb_{i}" for i in range(384)])

        return X_train, X_test, y_train, y_test

    def train_texture(self, data=None, n_estimators=100, depth=4, drop=.1, eta=.3):

        if data is None:
            X_train, X_test, y_train, y_test = self.load_data()
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]

        X_train = X_train[[f"pca_{i}" for i in range(50)]]
        X_test = X_test[[f"pca_{i}" for i in range(50)]]

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=self.random_state)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            max_depth=depth,
            eta=eta,
            verbosity=1,
            device='cuda',

            rate_drop=drop,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train[Field.TEXTURE], eval_set=[(X_val, y_val[Field.TEXTURE])], verbose=True)

        joblib.dump(model, f'{self.path}.txt.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.txt.fts')
        self.texture_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.texture_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.TEXTURE].tolist(), y_pred))
        print(classification_report(y_test[Field.TEXTURE].tolist(), y_pred, zero_division=0))

    def load_age(self):
        print("LOADING DATA SET")

        df = Data.load('weighted.parquet')

        print('ENCODING DATA')
        df = utils.encode_color(df)
        df = utils.encode_hardness(df)
        df = utils.encode_weights(df)

        df[Field.LENGTH] = df[Field.DEPTH_BOT] - df[Field.DEPTH_TOP]

        df_top = df.drop(columns=[Field.DEPTH_BOT]).rename(columns={Field.DEPTH_TOP: Field.DEPTH})
        df_top[Field.ORDER] = 0
        df_bot = df.drop(columns=[Field.DEPTH_TOP]).rename(columns={Field.DEPTH_BOT: Field.DEPTH})
        df_bot[Field.ORDER] = 1

        df = pd.concat([df_top, df_bot])

        df = df.sort_values([Field.RELATEID, Field.DEPTH, Field.ORDER])
        df = df[df[Field.STRAT] != 'BSMT']
        self.df = df

        y = df[[Field.RELATEID, Field.STRAT, Field.LITH_PRIM, Field.ORDER]]
        X = df.drop(columns=[Field.STRAT, Field.RELATEID, Field.LITH_PRIM, Field.ORDER])

        X[Field.UTME] = X[Field.UTME].fillna(X[Field.UTME].min() * .9)
        X[Field.UTMN] = X[Field.UTMN].fillna(X[Field.UTMN].min() * .9)
        X[Field.ELEVATION] = X[Field.ELEVATION].fillna(X[Field.ELEVATION].min() * .8)
        X[Field.DEPTH] = X[Field.DEPTH].fillna(-25)

        """Create a depth based elevation feature to replace elevation"""
        X[Field.ELEVATION] = X[Field.ELEVATION] - X[Field.DEPTH]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

        print('FITTING SCALERS')

        self.utme_scaler = StandardScaler()
        self.utme_scaler.fit(X_train[[Field.UTME]])

        self.utmn_scaler = StandardScaler()
        self.utmn_scaler.fit(X_train[[Field.UTMN]])

        self.elevation_scaler = StandardScaler()
        self.elevation_scaler.fit(X_train[[Field.ELEVATION]].values.tolist())

        self.depth_scaler = StandardScaler()
        self.depth_scaler.fit(X_train[[Field.DEPTH]].values.tolist())

        print('FITTING PCA')

        self.pca = PCA(n_components=50)
        self.pca.fit(X_train[[f"emb_{i}" for i in range(384)]])

        joblib.dump(self.utme_scaler, f'{self.path}.utme.scl')
        joblib.dump(self.utmn_scaler, f'{self.path}.utmn.scl')
        joblib.dump(self.elevation_scaler, f'{self.path}.elevation.scl')
        joblib.dump(self.depth_scaler, f'{self.path}.depth.scl')
        joblib.dump(self.pca, f'{self.path}.pca')

        print('APPLYING DATA REFINEMENTS')
        X[Field.UTME] = self.utme_scaler.transform(X[[Field.UTME]])
        X[Field.UTMN] = self.utmn_scaler.transform(X[[Field.UTMN]])
        X[Field.ELEVATION] = self.elevation_scaler.transform(X[[Field.ELEVATION]])
        X[Field.DEPTH] = self.depth_scaler.transform(X[[Field.DEPTH]])

        X[[f"pca_{i}" for i in range(50)]] = self.pca.transform(X[[f"emb_{i}" for i in range(384)]])
        X = X.drop(columns=[f"emb_{i}" for i in range(384)])

        X = X.astype(float)

        X[Field.PREVIOUS_AGE] = y.groupby(df[Field.RELATEID])[Field.AGE].shift(-1).fillna(-1)

        X_train, y_train = Age.encode_age(X_train, y_train)
        X_test, y_test = Age.encode_age(X_test, y_test)

        weights = X_train[Field.INTERPRETATION_METHOD]

        X_train = X_train.drop(columns=[Field.INTERPRETATION_METHOD])
        X_test = X_test.drop(columns=[Field.INTERPRETATION_METHOD])

        X_train[Field.ORDER] = y_train[Field.ORDER]
        X_test[Field.ORDER] = y_test[Field.ORDER]

        return X_train, X_test, y_train, y_test, weights

    def train_age(self, data=None, n_estimators=100, depth=4, drop=.1, eta=.3):
        """
        Differentiates between bedrock ages
        """
        if data is None:
            X_train, X_test, y_train, y_test, weights = self.load_age()
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]
            weights = data[4]

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=self.random_state)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            max_depth=depth,
            eta=eta,
            verbosity=1,
            device='cuda',

            rate_drop=drop,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train[Field.AGE], eval_set=[(X_val, y_val[Field.AGE])], sample_weight=weights)

        joblib.dump(model, f'{self.path}.age.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.age.fts')
        self.age_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.age_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.AGE].tolist(), y_pred))
        print(classification_report(y_test[Field.AGE].tolist(), y_pred, zero_division=0))

    def load_devonian(self):
        X_train, X_test, y_train, y_test = self.load_and_split_data()

        """Drop unnecessary features"""
        X_train = X_train.drop(columns=[f"pca_{i}" for i in range(50)] + Data.COLORS + [Field.HARDNESS])
        X_test = X_test.drop(columns=[f"pca_{i}" for i in range(50)] + Data.COLORS + [Field.HARDNESS])

        X_train, y_train = Age.encode_age(X_train, y_train)
        X_test, y_test = Age.encode_age(X_test, y_test)

        """Filter and encode data"""
        mask = y_train[Field.AGE] == 1

        X_train = X_train[mask]
        y_train = y_train[mask]

        y_train['label'] = y_train[Field.STRAT].map(Bedrock.BEDROCK_CODE_MAP)

        y_train.loc[y_train[Field.ORDER] == 0, 'label'] = y_train.loc[y_train[Field.ORDER] == 0, 'label'].apply(
            lambda x: x.get_top(1) if isinstance(x, Bedrock.GeoCode) else x)
        y_train.loc[y_train[Field.ORDER] == 1, 'label'] = y_train.loc[y_train[Field.ORDER] == 1, 'label'].apply(
            lambda x: x.get_bot(1) if isinstance(x, Bedrock.GeoCode) else x)

        mask = y_test[Field.AGE] == 1

        X_test = X_test[mask]
        y_test = y_test[mask]

        y_test['label'] = y_test[Field.STRAT].map(Bedrock.BEDROCK_CODE_MAP)

        y_test.loc[y_test[Field.ORDER] == 0, 'label'] = y_test.loc[y_test[Field.ORDER] == 0, 'label'].apply(
            lambda x: x.get_top(1) if isinstance(x, Bedrock.GeoCode) else x)
        y_test.loc[y_test[Field.ORDER] == 1, 'label'] = y_test.loc[y_test[Field.ORDER] == 1, 'label'].apply(
            lambda x: x.get_bot(1) if isinstance(x, Bedrock.GeoCode) else x)

        encoder = LabelEncoder()
        encoder.fit(['Cedar Valley', 'Wapsipinicon'])

        mask = y_train['label'].isna()
        X_train = X_train[~mask]
        y_train = y_train[~mask]

        mask = y_test['label'].isna()
        X_test = X_test[~mask]
        y_test = y_test[~mask]

        y_train['label'] = encoder.transform(y_train[['label']])
        y_test['label'] = encoder.transform(y_test[['label']])

        print("APPLYING WEIGHTS")

        weights = X_train[Field.INTERPRETATION_METHOD]

        X_train = X_train.drop(columns=[Field.INTERPRETATION_METHOD])
        X_test = X_test.drop(columns=[Field.INTERPRETATION_METHOD])

        X_train[Field.ORDER] = y_train[Field.ORDER]
        X_test[Field.ORDER] = y_test[Field.ORDER]

        return X_train, X_test, y_train, y_test, weights

    def train_devonian(self, data=None, n_estimators=100, depth=4, drop=.1, eta=.3):
        if data is None:
            X_train, X_test, y_train, y_test, weights = self.load_devonian()
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]
            weights = data[4]

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=self.random_state)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            max_depth=depth,
            eta=eta,
            verbosity=1,
            device='cuda',

            rate_drop=drop,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train['label'], eval_set=[(X_val, y_val['label'])], sample_weight=weights.values.tolist(), verbose=True)

        joblib.dump(model, f'{self.path}.dev.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.dev.fts')
        self.dev_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.dev_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test['label'].tolist(), y_pred))
        print(classification_report(y_test['label'].tolist(), y_pred, zero_division=0))

    def load_cedar_valley(self):
        X_train, X_test, y_train, y_test, weights = self.load_devonian()

        """Filter and encode data"""
        mask = y_train['label'] == 0

        X_train = X_train[mask]
        y_train = y_train[mask]
        weights = weights[mask]

        y_train['label'] = y_train[Field.STRAT].map(Bedrock.BEDROCK_CODE_MAP)

        y_train.loc[y_train[Field.ORDER] == 0, 'label'] = y_train.loc[y_train[Field.ORDER] == 0, 'label'].apply(
            lambda x: x.get_top(2) if isinstance(x, Bedrock.GeoCode) else x)
        y_train.loc[y_train[Field.ORDER] == 1, 'label'] = y_train.loc[y_train[Field.ORDER] == 1, 'label'].apply(
            lambda x: x.get_bot(2) if isinstance(x, Bedrock.GeoCode) else x)

        mask = y_test['label'] == 0

        X_test = X_test[mask]
        y_test = y_test[mask]

        y_test['label'] = y_test[Field.STRAT].map(Bedrock.BEDROCK_CODE_MAP)

        y_test.loc[y_test[Field.ORDER] == 0, 'label'] = y_test.loc[y_test[Field.ORDER] == 0, 'label'].apply(
            lambda x: x.get_top(2) if isinstance(x, Bedrock.GeoCode) else x)
        y_test.loc[y_test[Field.ORDER] == 1, 'label'] = y_test.loc[y_test[Field.ORDER] == 1, 'label'].apply(
            lambda x: x.get_bot(2) if isinstance(x, Bedrock.GeoCode) else x)

        encoder = LabelEncoder()
        encoder.fit(['Upper Cedar Valley', 'Lower Cedar Valley', None])

        y_train['label'] = encoder.transform(y_train[['label']])
        y_test['label'] = encoder.transform(y_test[['label']])

        return X_train, X_test, y_train, y_test, weights

    def train_cedar_valley(self, data=None, n_estimators=100, depth=4, drop=.1, eta=.3):
        if data is None:
            X_train, X_test, y_train, y_test, weights = self.load_cedar_valley()
        else:
            X_train = data[0]
            X_test = data[1]
            y_train = data[2]
            y_test = data[3]
            weights = data[4]

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=self.random_state)

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            max_depth=depth,
            eta=eta,
            verbosity=1,
            device='cuda',

            rate_drop=drop,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train['label'], eval_set=[(X_val, y_val['label'])], sample_weight=weights.values.tolist(), verbose=True)

        joblib.dump(model, f'{self.path}.cdv.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.cdv.fts')
        self.cdv_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.cdv_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test['label'].tolist(), y_pred))
        print(classification_report(y_test['label'].tolist(), y_pred, zero_division=0))

    def train_cambrian(self, n_estimators=100):
        X_train, X_test, y_train, y_test = self.load_and_split_data()

        """Drop unnecessary features"""
        X_train = X_train.drop(columns=[f"pca_{i}" for i in range(50)] + [Field.LENGTH, 'DARK', 'LIGHT', 'PURPLE',
                                                                          'ORANGE', 'PINK', 'VARIED', 'RED', 'GRAY'
                                                                          'BROWN', Field.DEPTH, 'BLACK'])
        X_test = X_test.drop(columns=[f"pca_{i}" for i in range(50)] + [Field.LENGTH])

        X_train, y_train = Age.encode_age(X_train, y_train)
        X_test, y_test = Age.encode_age(X_test, y_test)

        """Filter and encode data"""
        mask = y_train[Field.AGE] == 0

        X_train = X_train[mask]
        y_train = y_train[mask]

        y_train['label'] = y_train[Field.STRAT].map(Bedrock.BEDROCK_CODE_MAP)

        y_train.loc[y_train[Field.ORDER] == 0, 'label'] = y_train.loc[y_train[Field.ORDER] == 0, 'label'].apply(
            lambda x : x.get_top(1) if isinstance(x, Bedrock.GeoCode) else x)
        y_train.loc[y_train[Field.ORDER] == 1, 'label'] = y_train.loc[y_train[Field.ORDER] == 1, 'label'].apply(
            lambda x : x.get_bot(1) if isinstance(x, Bedrock.GeoCode) else x)

        mask = y_test[Field.AGE] == 0

        X_test = X_test[mask]
        y_test = y_test[mask]

        y_test['label'] = y_test[Field.STRAT].map(Bedrock.BEDROCK_CODE_MAP)

        y_test.loc[y_test[Field.ORDER] == 0, 'label'] = y_test.loc[y_test[Field.ORDER] == 0, 'label'].apply(
            lambda x: x.get_top(1) if isinstance(x, Bedrock.GeoCode) else x)
        y_test.loc[y_test[Field.ORDER] == 1, 'label'] = y_test.loc[y_test[Field.ORDER] == 1, 'label'].apply(
            lambda x: x.get_bot(1) if isinstance(x, Bedrock.GeoCode) else x)

        encoder = LabelEncoder()
        encoder.fit(['Prairie Du Chien', 'Jordan', 'St Lawrence', 'Tunnel City', 'Wonewoc', 'Eau Claire', 'Mt Simon', None])

        y_train['label'] = encoder.transform(y_train[['label']])
        y_test['label'] = encoder.transform(y_test[['label']])

        print("APPLYING WEIGHTS")

        weights = X_train[Field.INTERPRETATION_METHOD].values.tolist()

        X_train = X_train.drop(columns=[Field.INTERPRETATION_METHOD])
        X_test = X_test.drop(columns=[Field.INTERPRETATION_METHOD])

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            booster='dart',
            n_estimators=n_estimators,
            verbosity=2,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train['label'], sample_weight=weights)

        joblib.dump(model, f'{self.path}.cam.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.cam.fts')
        self.cam_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.cam_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test['label'].tolist(), y_pred))
        print(classification_report(y_test['label'].tolist(), y_pred, zero_division=0))

        report = classification_report(
            y_test['label'].tolist(),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        self.y_pred = y_pred
        self.y_test = y_test
        self.X_test = X_test

        return report['macro avg']['f1-score'], report['accuracy']

    def train_group(self, n_estimators=100, depth=4, max_delta=0, eta=.001, early_stopping=0):
        if self.X_train is None:
            self.load_data()

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

        """Split the testing data into its top and bottom components then recombine"""
        X_train_top = (X_train.drop(columns=[Field.DEPTH_BOT, Field.ELEVATION_BOT])
                       .rename(columns={Field.DEPTH_TOP : Field.DEPTH, Field.ELEVATION_TOP : Field.ELEVATION}))
        X_train_bot = (X_train.drop(columns=[Field.DEPTH_TOP, Field.ELEVATION_TOP])
                       .rename(columns={Field.DEPTH_BOT : Field.DEPTH, Field.ELEVATION_BOT : Field.ELEVATION}))
        X_train = pd.concat([X_train_top, X_train_bot])

        y_train_top = y_train[[Field.GROUP_TOP]].rename(columns={Field.GROUP_TOP: Field.GROUP})
        y_train_bot = y_train[[Field.GROUP_BOT]].rename(columns={Field.GROUP_BOT: Field.GROUP})
        y_train = pd.concat([y_train_top, y_train_bot])

        X_test_top = (X_test.drop(columns=[Field.DEPTH_BOT, Field.ELEVATION_BOT])
                      .rename(columns={Field.DEPTH_TOP : Field.DEPTH, Field.ELEVATION_TOP : Field.ELEVATION}))
        X_test_bot = (X_test.drop(columns=[Field.DEPTH_TOP, Field.ELEVATION_TOP])
                       .rename(columns={Field.DEPTH_BOT : Field.DEPTH, Field.ELEVATION_BOT : Field.ELEVATION}))
        X_test = pd.concat([X_test_top, X_test_bot])

        y_test_top = y_test[[Field.GROUP_TOP]].rename(columns={Field.GROUP_TOP: Field.GROUP})
        y_test_bot = y_test[[Field.GROUP_BOT]].rename(columns={Field.GROUP_BOT: Field.GROUP})
        y_test = pd.concat([y_test_top, y_test_bot])

        """Drop noisy/unnecessary features"""
        X_train = X_train.drop(columns=[Field.PREVIOUS_MEMBER, Field.PREVIOUS_TEXTURE])
        X_test = X_test.drop(columns=[Field.PREVIOUS_MEMBER, Field.PREVIOUS_TEXTURE])

        drop_dict = {
            7: .33,
        }

        encoder, _, _ = Bedrock.init_encoders()

        for group, percentage in drop_dict.items():
            mask = y_train[Field.GROUP] == group
            indices = y_train[mask].index

            np.random.seed(self.random_state)
            kept_indices = np.random.choice(indices, size=int(len(indices) * percentage))

            mask = ~mask | y_train.index.isin(kept_indices)

            X_train = X_train[mask].reset_index(drop=True)
            y_train = y_train[mask].reset_index(drop=True)

        #X_train, y_train = create_bedrock(X_train, y_train, 0, 2000, Field.GROUP)
        #X_train, y_train = create_bedrock(X_train, y_train, 2, 2000, Field.GROUP)
        #X_train, y_train = create_bedrock(X_train, y_train, 5, 2000, Field.GROUP)
        #X_train, y_train = create_bedrock(X_train, y_train, 6, 2000, Field.GROUP)

        #mask = y_train[Field.GROUP] == 0
        #X_train.loc[mask, Field.INTERPRETATION_METHOD] = X_train.loc[mask, Field.INTERPRETATION_METHOD] * 5

        weights = X_train[Field.INTERPRETATION_METHOD].values.tolist()

        X_train = X_train.drop(columns=[Field.INTERPRETATION_METHOD])
        X_test = X_test.drop(columns=[Field.INTERPRETATION_METHOD])

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            objective='multi:softmax',
            booster='dart',
            n_estimators=n_estimators,
            max_depth=depth,
            max_delta_step=max_delta,
            eta=eta,
            early_stopping_rounds=early_stopping,
            verbosity=1,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train[Field.GROUP], sample_weight=weights, eval_set=[(X_test, y_test[Field.GROUP])])

        joblib.dump(model, f'{self.path}.grp.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.grp.fts')
        self.group_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.group_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.GROUP].values.tolist(), y_pred))

        report = classification_report(
            y_test[Field.GROUP].values.tolist(),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        print(classification_report(y_test[Field.GROUP].values.tolist(), y_pred, zero_division=0))

        return report['macro avg']['f1-score'], accuracy_score(y_test[Field.GROUP].values.tolist(), y_pred)

    def train_formation(self, n_estimators=100):
        if self.X_train is None:
            self.load_data()

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

        """Split the testing data into its top and bottom components then recombine"""
        X_train_top = (X_train.drop(columns=[Field.DEPTH_BOT, Field.ELEVATION_BOT])
                       .rename(columns={Field.DEPTH_TOP : Field.DEPTH, Field.ELEVATION_TOP : Field.ELEVATION}))
        X_train_bot = (X_train.drop(columns=[Field.DEPTH_TOP, Field.ELEVATION_TOP])
                       .rename(columns={Field.DEPTH_BOT : Field.DEPTH, Field.ELEVATION_BOT : Field.ELEVATION}))
        X_train = pd.concat([X_train_top, X_train_bot])

        y_train_top = y_train[[Field.FORMATION_TOP]].rename(columns={Field.FORMATION_TOP: Field.FORMATION})
        y_train_bot = y_train[[Field.FORMATION_BOT]].rename(columns={Field.FORMATION_BOT: Field.FORMATION})
        y_train = pd.concat([y_train_top, y_train_bot])

        X_test_top = (X_test.drop(columns=[Field.DEPTH_BOT, Field.ELEVATION_BOT])
                      .rename(columns={Field.DEPTH_TOP : Field.DEPTH, Field.ELEVATION_TOP : Field.ELEVATION}))
        X_test_bot = (X_test.drop(columns=[Field.DEPTH_TOP, Field.ELEVATION_TOP])
                       .rename(columns={Field.DEPTH_BOT : Field.DEPTH, Field.ELEVATION_BOT : Field.ELEVATION}))
        X_test = pd.concat([X_test_top, X_test_bot])

        y_test_top = y_test[[Field.FORMATION_TOP]].rename(columns={Field.FORMATION_TOP: Field.FORMATION})
        y_test_bot = y_test[[Field.FORMATION_BOT]].rename(columns={Field.FORMATION_BOT: Field.FORMATION})
        y_test = pd.concat([y_test_top, y_test_bot])

        """Drop noisy/unnecessary features"""
        X_train = X_train.drop(columns=[Field.PREVIOUS_MEMBER])
        X_test = X_test.drop(columns=[Field.PREVIOUS_MEMBER])

        drop_dict = {
            38: .25,
        }

        _, encoder, _ = Bedrock.init_encoders()

        for formation, percentage in drop_dict.items():
            mask = y_train[Field.FORMATION] == formation
            indices = y_train[mask].index

            np.random.seed(self.random_state)
            kept_indices = np.random.choice(indices, size=int(len(indices) * percentage))

            mask = ~mask | y_train.index.isin(kept_indices)

            X_train = X_train[mask].reset_index(drop=True)
            y_train = y_train[mask].reset_index(drop=True)

        weights = X_train[Field.INTERPRETATION_METHOD].values.tolist()

        X_train = X_train.drop(columns=[Field.INTERPRETATION_METHOD])
        X_test = X_test.drop(columns=[Field.INTERPRETATION_METHOD])

        print('TRAINING MODEL')

        model = xgboost.XGBClassifier(
            objective='multi:softmax',
            booster='dart',
            n_estimators=n_estimators,
            verbosity=2,
            device='cuda',

            rate_drop=.1,
            normalize_type='forest',
            sample_type='weighted',

            tree_method='hist'
        )
        model.fit(X_train, y_train, sample_weight=weights)

        joblib.dump(model, f'{self.path}.frm.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.frm.fts')
        self.group_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.group_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.FORMATION].values.tolist(), y_pred))

        report = classification_report(
            y_test[Field.FORMATION].values.tolist(),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        print(classification_report(y_test[Field.FORMATION].values.tolist(), y_pred, zero_division=0))

        return report['macro avg']['f1-score'], accuracy_score(y_test[Field.FORMATION].values.tolist(), y_pred)

    def train_member(self, n_estimators=100):
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

        """Split the testing data into its top and bottom components then recombine"""
        X_train_top = (X_train.drop(columns=[Field.DEPTH_BOT, Field.ELEVATION_BOT])
                       .rename(columns={Field.DEPTH_TOP : Field.DEPTH, Field.ELEVATION_TOP : Field.ELEVATION}))
        X_train_bot = (X_train.drop(columns=[Field.DEPTH_TOP, Field.ELEVATION_TOP])
                       .rename(columns={Field.DEPTH_BOT : Field.DEPTH, Field.ELEVATION_BOT : Field.ELEVATION}))
        X_train = pd.concat([X_train_top, X_train_bot])

        y_train_top = y_train[Field.MEMBER_TOP].rename(columns={Field.MEMBER_TOP : Field.MEMBER})
        y_train_bot = y_train[Field.MEMBER_BOT].rename(columns={Field.MEMBER_BOT : Field.MEMBER})
        y_train = pd.concat([y_train_top, y_train_bot])

        X_test_top = (X_test.drop(columns=[Field.DEPTH_BOT, Field.ELEVATION_BOT])
                      .rename(columns={Field.DEPTH_TOP : Field.DEPTH, Field.ELEVATION_TOP : Field.ELEVATION}))
        X_test_bot = (X_test.drop(columns=[Field.DEPTH_TOP, Field.ELEVATION_TOP])
                       .rename(columns={Field.DEPTH_BOT : Field.DEPTH, Field.ELEVATION_BOT : Field.ELEVATION}))
        X_test = pd.concat([X_test_top, X_test_bot])

        y_test_top = y_test[Field.MEMBER_TOP].rename(columns={Field.MEMBER_TOP : Field.MEMBER})
        y_test_bot = y_test[Field.MEMBER_BOT].rename(columns={Field.MEMBER_BOT : Field.MEMBER})
        y_test = pd.concat([y_test_top, y_test_bot])

        """Drop noisy/unnecessary features"""
        X_train = X_train.drop(columns=[Field.PREVIOUS_MEMBER])
        X_test = X_test.drop(columns=[Field.PREVIOUS_MEMBER])

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
        model.fit(X_train, y_train)

        joblib.dump(model, f'{self.path}.frm.mdl')
        joblib.dump(X_train.columns.tolist(), f'{self.path}.frm.fts')
        self.group_model = model

        print('EVALUATING CLASSIFIER')

        y_pred = self.group_model.predict(X_test)

        print('Accuracy: ', accuracy_score(y_test[Field.FORMATION].tolist(), y_pred))

        report = classification_report(
            y_test[Field.FORMATION].tolist(),
            y_pred,
            zero_division=0,
            output_dict=True
        )

        print(classification_report(y_test[Field.FORMATION].tolist(), y_pred, zero_division=0))

        return report['macro avg']['f1-score'], accuracy_score(y_test[Field.FORMATION].tolist(), y_pred)