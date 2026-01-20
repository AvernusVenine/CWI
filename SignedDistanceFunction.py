import numpy as np
import pandas as pd
import torch

import Data, utils
from Data import Field

class SignedDistanceFunction:

    def __init__(self, df):
        self.field = None

        self.utm_size = 10.0
        self.labels = df[Field.STRAT].value_counts().keys().tolist()

        self.utme_max = df[Field.UTME].max()
        self.utme_min = df[Field.UTME].min()

        self.utmn_max = df[Field.UTMN].max()
        self.utmn_min = df[Field.UTMN].min()

        self.elevation_max = df[Field.ELEVATION_TOP].max()

        self.create_field(df)

    def create_field(self, df):
        n_utme = int(np.ceil((self.utme_max - self.utme_min) / self.utm_size) + 1)
        n_utmn = int(np.ceil((self.utmn_max - self.utmn_min) / self.utm_size) + 1)
        n_elevation = int(self.elevation_max + 1)

        self.field = np.full((n_utme, n_utmn, n_elevation), -1, dtype=np.int8)

        for _, row in df.iterrows():
            utme_idx = int((row[Field.UTME] - self.utme_min) / self.utm_size)
            utmn_idx = int((row[Field.UTMN] - self.utmn_min) / self.utm_size)

            for elevation in range(int(row[Field.ELEVATION_BOT]), int(row[Field.ELEVATION_TOP])):
                self.field[utme_idx, utmn_idx, elevation] = row[Field.STRAT]

    def compute_signed_distance(self, utme, utmn, elevation, label):
        utme_idx = int((utme - self.utme_min) / self.utm_size)
        utmn_idx = int((utmn - self.utmn_min) / self.utm_size)

        column = self.field[utme_idx, utmn_idx, :]
        label_elevations = np.where(column == label)[0]

        if label_elevations is None:
            return -np.inf

        label_bottom = label_elevations.min()
        label_top = label_elevations.max()

        inside = (label_bottom <= elevation <= label_top + 1)

        bot_dist = abs(elevation - label_bottom)
        top_dist = abs(elevation - (label_top + 1))

        min_dist = min(bot_dist, top_dist)

        return min_dist if inside else -min_dist

    def compute_all(self, utme, utmn, elevation, max_label):
        y = []

        for idx in range(max_label):
            y.append(self.compute_signed_distance(utme, utmn, elevation, idx))

        return y