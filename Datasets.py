from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import torch
import joblib
import warnings
import pandas as pd
import random

import Data
from Data import Field
from SignedDistanceFunction import SignedDistanceFunction

class Strat:
    QUATERNARY = 'Quaternary'
    CRETACEOUS = 'Cretaceous'

    PRAIRIE_DU_CHIEN = 'Prairie Du Chien'
    ST_PETER = 'St Peter'
    JORDAN = 'Jordan'
    PLATTEVILLE = 'Platteville'
    GLENWOOD = 'Glenwood'
    DECORAH_SHALE = 'Decorah Shale'
    GALENA = 'Galena'
    ST_LAWRENCE = 'St Lawrence'
    TUNNEL_CITY = 'Tunnel City'
    EAU_CLAIRE = 'Eau Claire'
    WONEWOC = 'Wonewoc'

    CODE_DICT = {
        'OPDC': (PRAIRIE_DU_CHIEN, PRAIRIE_DU_CHIEN),
        'OSTP': (ST_PETER, ST_PETER),
        'CJDN': (JORDAN, JORDAN),
        'OPVL': (PLATTEVILLE, PLATTEVILLE),
        'OGWD': (GLENWOOD, GLENWOOD),
        'ODCR': (DECORAH_SHALE, DECORAH_SHALE),
        'OGCM': (GALENA, GALENA),
        'OPSH': (PRAIRIE_DU_CHIEN, PRAIRIE_DU_CHIEN),
        'OGPC': (GALENA, GALENA),
        'OPOD': (PRAIRIE_DU_CHIEN, PRAIRIE_DU_CHIEN),
        'OPGW': (PLATTEVILLE, GLENWOOD),
        'OGSC': (GALENA, GALENA),
        'OGCD': (GALENA, DECORAH_SHALE),
        'CSTL': (ST_LAWRENCE, ST_LAWRENCE),
        'OGPR': (GALENA, GALENA),
        'OGSV': (GALENA, GALENA),
        'CTLR': (TUNNEL_CITY, TUNNEL_CITY),
        'OGVP': (GALENA, GALENA),
        # 'ODPG' BAD CODE
        'CECR': (EAU_CLAIRE, EAU_CLAIRE),
        'ODGL': (GALENA, GALENA),
        'ODPL': (DECORAH_SHALE, PLATTEVILLE),
        'OPNR': (PRAIRIE_DU_CHIEN, PRAIRIE_DU_CHIEN),
        'CWOC': (WONEWOC, WONEWOC),
        'OPWR': (PRAIRIE_DU_CHIEN, PRAIRIE_DU_CHIEN),
        'CTCG': (TUNNEL_CITY, TUNNEL_CITY),
    }

class StratDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y[Field.STRAT].values).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def condense_layers(df):
    """
    Condenses the layers within a dataframe by combining neighbors of the same value into one
    Args:
        df: Layer dataframe

    Returns: Condensed dataframe
    """
    df = df.sort_values([Field.RELATEID, Field.ELEVATION_TOP, Field.ELEVATION_BOT], ascending=[True, False, True])
    new_df = pd.DataFrame(columns=df.columns)

    for _, hole in df.groupby(Field.RELATEID):
        strat_top = None
        strat_bot = None
        elevation_top = 0
        elevation_bot = 0

        relateid = 0
        utme = 0
        utmn = 0

        for _, row in hole.iterrows():
            if row['strat_top'] == strat_bot and row[Field.ELEVATION_TOP] == elevation_bot:
                elevation_bot = row[Field.ELEVATION_BOT]
                strat_bot = row['strat_bot']
            elif strat_top is None:
                elevation_top = row[Field.ELEVATION_TOP]
                elevation_bot = row[Field.ELEVATION_BOT]

                strat_top = row['strat_top']
                strat_bot = row['strat_bot']

                relateid = row[Field.RELATEID]
                utme = row[Field.UTME]
                utmn = row[Field.UTMN]
            else:
                section = pd.DataFrame({
                    Field.RELATEID: [relateid],
                    Field.UTME: [utme],
                    Field.UTMN: [utmn],
                    Field.ELEVATION_TOP: [elevation_top],
                    Field.ELEVATION_BOT: [elevation_bot],
                    'strat_top': [strat_top],
                    'strat_bot': [strat_bot]
                })
                new_df = pd.concat([new_df, section])

                elevation_top = row[Field.ELEVATION_TOP]
                elevation_bot = row[Field.ELEVATION_BOT]

                strat_top = row['strat_top']
                strat_bot = row['strat_bot']

        section = pd.DataFrame({
            Field.RELATEID: [relateid],
            Field.UTME: [utme],
            Field.UTMN: [utmn],
            Field.ELEVATION_TOP: [elevation_top],
            Field.ELEVATION_BOT: [elevation_bot],
            'strat_top': [strat_top],
            'strat_bot': [strat_bot]
        })
        new_df = pd.concat([new_df, section])

    return new_df

def load_cwi_data(county=(55,), early_return=False):
    warnings.filterwarnings('ignore')

    print('LOADING DATASET')
    df = Data.load('weighted.parquet')
    raw = Data.load_well_raw()

    df[Field.COUNTY] = df[Field.RELATEID].map(raw.set_index(Field.RELATEID)[Field.COUNTY])
    df = df[df[Field.COUNTY].isin(county)]

    df = df.dropna(subset=[Field.DEPTH_TOP, Field.DEPTH_BOT, Field.UTME, Field.UTMN, Field.ELEVATION])

    df[Field.ELEVATION_TOP] = df[Field.ELEVATION] - df[Field.DEPTH_TOP]
    df[Field.ELEVATION_BOT] = df[Field.ELEVATION] - df[Field.DEPTH_BOT]

    df = df[[Field.RELATEID, Field.STRAT, Field.UTME, Field.UTMN, Field.ELEVATION_TOP, Field.ELEVATION_BOT]]

    qdf = df[df[Field.STRAT].astype(str).str[0].isin(['Q', 'R', 'F', 'X', 'Y'])]
    qdf[['strat_top', 'strat_bot']] = (Strat.QUATERNARY, Strat.QUATERNARY)

    kdf = df[df[Field.STRAT].astype(str).str[0].isin(['K'])]
    kdf[['strat_top', 'strat_bot']] = (Strat.CRETACEOUS, Strat.CRETACEOUS)

    df = df[df[Field.STRAT].isin(Strat.CODE_DICT.keys())]
    df['strat_top'] = df[Field.STRAT].apply(lambda x: Strat.CODE_DICT[x][0])
    df['strat_bot'] = df[Field.STRAT].apply(lambda x: Strat.CODE_DICT[x][1])

    df = pd.concat([df, qdf, kdf], ignore_index=True)

    encoder = LabelEncoder()
    encoder.fit(list(set(df['strat_top'].values.tolist()).union(set(df['strat_bot'].values.tolist())))),

    df['strat_top'] = encoder.transform(df['strat_top'])
    df['strat_bot'] = encoder.transform(df['strat_bot'])

    df = df.sort_values([Field.RELATEID, Field.ELEVATION_BOT])

    df['type'] = 0
    bottom_index = df.groupby(Field.RELATEID)[Field.ELEVATION_BOT].idxmin()
    df.loc[bottom_index, 'type'] = 1

    if early_return:
        return df

    point_df = df[df['strat_top'] != df['strat_bot']]
    point_df['type'] = 2

    point_df = point_df.drop(['strat_top', 'strat_bot']).rename()

    sdf = SignedDistanceFunction(df)

    valid_codes = {
        
    }

    qdf = df[df[Field.STRAT].astype(str).str[0].isin(['Q', 'R', 'W', 'F', 'Y', 'X'])]
    qdf[Field.STRAT] = 'Quaternary'

    kdf = df[df[Field.STRAT].astype(str).str[0].isin(['K'])]
    kdf[Field.STRAT] = 'Cretaceous'

    df = df[df[Field.STRAT].isin(list(valid_codes.keys()))]

    df[Field.STRAT] = df[Field.STRAT].replace(valid_codes)

    df = pd.concat([df, qdf, kdf])

    df = condense_layers(df)

    """Get rid of all 0 thickness layers and single layer boreholes"""
    df = df[df[Field.ELEVATION_TOP] - df[Field.ELEVATION_BOT] > 0.0]
    df = df.groupby('relateid').filter(lambda x: len(x) > 1)

    """Drop layers that have very low representation in the dataset"""
    count = df[Field.STRAT].value_counts()
    df = df[df[Field.STRAT].isin(count[count > 100].index)]

    if early_return:
        return df

    encoder = LabelEncoder()
    df[Field.STRAT] = encoder.fit_transform(df[Field.STRAT])
    joblib.dump(encoder, 'nn/strat.enc')

    """Split the dataset by entire wells instead of by individual layers"""
    relateids = list(set(df[Field.RELATEID].values))
    random.shuffle(relateids)

    split = int(len(relateids) * .85)

    train_ids = relateids[:split]
    test_ids = relateids[split:]

    sdf = SignedDistanceFunction(df, len(encoder.classes_))

    utme_scaler = MinMaxScaler()
    df[Field.UTME] = utme_scaler.fit_transform(df[[Field.UTME]].values.tolist())
    joblib.dump(utme_scaler, 'nn/utme.scl')

    utmn_scaler = MinMaxScaler()
    df[Field.UTMN] = utmn_scaler.fit_transform(df[[Field.UTMN]].values.tolist())
    joblib.dump(utmn_scaler, 'nn/utmn.scl')

    elevation_scaler = MinMaxScaler()
    elevation_scaler.fit(df[[Field.ELEVATION_TOP]].values.tolist() + df[[Field.ELEVATION_BOT]].values.tolist())
    df[Field.ELEVATION_TOP] = elevation_scaler.transform(df[[Field.ELEVATION_TOP]].values.tolist())
    df[Field.ELEVATION_BOT] = elevation_scaler.transform(df[[Field.ELEVATION_BOT]].values.tolist())
    joblib.dump(elevation_scaler, 'nn/elevation.scl')

    train_df = df[df[Field.RELATEID].isin(train_ids)]
    test_df = df[df[Field.RELATEID].isin(test_ids)]

    X_train = train_df[[Field.ELEVATION_TOP, Field.ELEVATION_BOT, Field.UTME, Field.UTMN]]
    X_test = test_df[[Field.ELEVATION_TOP, Field.ELEVATION_BOT, Field.UTME, Field.UTMN]]

    y_train = train_df[[Field.STRAT]]
    y_test = train_df[[Field.STRAT]]

    count = y_train[Field.STRAT].value_counts()
    weights = [(1/count[i])**.25 for i in y_train[Field.STRAT].values]

    sampler = WeightedRandomSampler(weights=weights, num_samples=int(len(y_train)), replacement=True)

    train = StratDataset(X_train, y_train)
    test = StratDataset(X_test, y_test)

    train_loader = DataLoader(train, batch_size=512, sampler=sampler)
    test_loader = DataLoader(test, batch_size=512)

    return train_loader, test_loader, sdf

def load_qdi_data(county=27):
    warnings.filterwarnings('ignore')

    print('LOADING DATASET')
    df = Data.load_qdi_raw()

    df = df[df[Field.COUNTY] == county]

    valid_codes = {
        'QNUH' : 'Heiberg',
        'QCMU' : 'Cromwell',
        'QNUT' : 'New Ulm: Twin Cities',
        'QNVT' : 'New Ulm: Villard',
        'QNUU' : 'New Ulm',
        'QOTV' : 'Otter Tail: Villard'
    }

    df = df.dropna(subset=[Field.DEPTH_TOP, Field.DEPTH_BOT, Field.UTME, Field.UTMN, Field.ELEVATION])

    qdf = df[df[Field.STRAT].astype(str).str[0].isin(['Q', 'R', 'W', 'F', 'Y', 'X'])]
    qdf[Field.STRAT] = 'Quaternary'

    kdf = df[df[Field.STRAT].astype(str).str[0].isin(['K'])]
    kdf[Field.STRAT] = 'Cretaceous'

    df = df[df[Field.STRAT].isin(list(valid_codes.keys()))]

    df[Field.STRAT] = df[Field.STRAT].replace(valid_codes)

    df = pd.concat([df, qdf, kdf])

    df[Field.ELEVATION_TOP] = df[Field.ELEVATION] - df[Field.DEPTH_TOP]
    df[Field.ELEVATION_BOT] = df[Field.ELEVATION] - df[Field.DEPTH_BOT]

    df = df[[Field.RELATEID, Field.STRAT, Field.UTME, Field.UTMN, Field.ELEVATION_TOP, Field.ELEVATION_BOT]]

    df = condense_layers(df)

    """Get rid of all 0 thickness layers and single layer boreholes"""
    df = df[df[Field.ELEVATION_TOP] - df[Field.ELEVATION_BOT] > 0.0]
    df = df.groupby('relateid').filter(lambda x: len(x) > 1)

    encoder = LabelEncoder()
    df[Field.STRAT] = encoder.fit_transform(df[Field.STRAT])
    joblib.dump(encoder, 'nn/strat.enc')

    """Split the dataset by entire wells instead of by individual layers"""
    relateids = list(set(df[Field.RELATEID].values))
    random.shuffle(relateids)

    split = int(len(relateids) * .85)

    train_ids = relateids[:split]
    test_ids = relateids[split:]

    sdf = SignedDistanceFunction(df, len(encoder.classes_))

    utme_scaler = MinMaxScaler()
    df[Field.UTME] = utme_scaler.fit_transform(df[[Field.UTME]].values.tolist())
    joblib.dump(utme_scaler, 'nn/utme.scl')

    utmn_scaler = MinMaxScaler()
    df[Field.UTMN] = utmn_scaler.fit_transform(df[[Field.UTMN]].values.tolist())
    joblib.dump(utmn_scaler, 'nn/utmn.scl')

    elevation_scaler = MinMaxScaler()
    elevation_scaler.fit(df[[Field.ELEVATION_TOP]].values.tolist() + df[[Field.ELEVATION_BOT]].values.tolist())
    df[Field.ELEVATION_TOP] = elevation_scaler.transform(df[[Field.ELEVATION_TOP]].values.tolist())
    df[Field.ELEVATION_BOT] = elevation_scaler.transform(df[[Field.ELEVATION_BOT]].values.tolist())
    joblib.dump(elevation_scaler, 'nn/elevation.scl')

    train_df = df[df[Field.RELATEID].isin(train_ids)]
    test_df = df[df[Field.RELATEID].isin(test_ids)]

    X_train = train_df[[Field.ELEVATION_TOP, Field.ELEVATION_BOT, Field.UTME, Field.UTMN]]
    X_test = test_df[[Field.ELEVATION_TOP, Field.ELEVATION_BOT, Field.UTME, Field.UTMN]]

    y_train = train_df[[Field.STRAT]]
    y_test = train_df[[Field.STRAT]]

    count = y_train[Field.STRAT].value_counts()
    weights = [(1/count[i])**.25 for i in y_train[Field.STRAT].values]

    sampler = WeightedRandomSampler(weights=weights, num_samples=int(len(y_train)), replacement=True)

    train = StratDataset(X_train, y_train)
    test = StratDataset(X_test, y_test)

    train_loader = DataLoader(train, batch_size=512, sampler=sampler)
    test_loader = DataLoader(test, batch_size=512)

    return train_loader, test_loader, sdf