import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree

import Data, utils
from Data import Field

class SignedDistanceFunction:

    def __init__(self, df):
        self.field = None
        self.kdtree = None
        self.df = df.copy()
        self.k = 50

        self.meter_const = 3.28

        self.utm_size = 10.0
        self.labels = df[Field.STRAT].value_counts().keys().tolist()

        self.utme_max = df[Field.UTME].max()
        self.utme_min = df[Field.UTME].min()

        self.utmn_max = df[Field.UTMN].max()
        self.utmn_min = df[Field.UTMN].min()

        self.elevation_max = df[Field.ELEVATION_TOP].max()

        self.create_field(df)
        self.create_kdtree(df)

    def create_kdtree(self, df):
        """
        Creates a KDTree to optimize finding the K nearest neighbors to a given borehole using UTM as a metric
        Args:
            df: Dataframe
        Returns:
        """

        self.kdtree = KDTree(df[[Field.UTME, Field.UTMN]])

    def create_field(self, df):
        """
        Creates a field with indices (UTME, UTMN, Depth) that contains data of the form (int, boolean)
        which represents the label and whether it is a boundary respectively
        Args:
            df: Dataframe
        Returns:
        """
        n_utme = int(np.ceil((self.utme_max - self.utme_min) / self.utm_size) + 1)
        n_utmn = int(np.ceil((self.utmn_max - self.utmn_min) / self.utm_size) + 1)
        n_elevation = int(self.elevation_max + 1)

        dtype = np.dtype([('label', np.int8), ('boundary', np.bool_)])

        self.field = np.zeros((n_utme, n_utmn, n_elevation), dtype=dtype)
        self.field['label'] = -1
        self.field['boundary'] = False

        for _, row in df.iterrows():
            utme_idx = int((row[Field.UTME] - self.utme_min) / self.utm_size)
            utmn_idx = int((row[Field.UTMN] - self.utmn_min) / self.utm_size)

            for elevation in range(int(row[Field.ELEVATION_BOT]), int(row[Field.ELEVATION_TOP])):
                self.field[utme_idx, utmn_idx, elevation]['label'] = row[Field.STRAT]

            self.field[utme_idx, utmn_idx, int(row[Field.ELEVATION_TOP])]['boundary'] = True

    def compute_min_distance(self, utme, utmn, elevation, label):
        _, points = self.kdtree.query([utme, utmn], k=self.k)

        df = self.df[(self.df[Field.UTME].isin(points[0])) & (self.df[Field.UTMN].isin(points[1])) & (self.df[Field.STRAT] == label)]

        if df.empty:
            return np.inf

        df['distance'] = np.sqrt((df[Field.UTME] - utme) ** 2 + (df[Field.UTMN] - utmn) ** 2 + ((df[Field.ELEVATION] - elevation)/self.meter_const) ** 2)

        return df['distance'].min()

    def compute_all(self, utme, utmn, elevation, max_label):
        y = []

        for idx in range(max_label):
            y.append(self.compute_signed_distance(utme, utmn, elevation, idx))

        return y