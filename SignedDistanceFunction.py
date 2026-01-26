import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree

import Data, utils
from Data import Field

class SignedDistanceFunction:

    def __init__(self, df, max_label):
        self.kdtree = None
        self.df = df.copy()
        self.k = 10
        self.max_label = max_label
        self.meter_const = 3.28

        self.extract_boundaries()
        self.create_kdtree()

    def extract_boundaries(self):
        """
        Extracts all known boundary data points
        Returns:
        """
        df_top = self.df.drop(columns=[Field.ELEVATION_BOT]).rename(columns={Field.ELEVATION_TOP: Field.ELEVATION})
        df_bot = self.df.drop(columns=[Field.ELEVATION_TOP]).rename(columns={Field.ELEVATION_BOT: Field.ELEVATION})

        df_bot = df_bot.sort_values([Field.RELATEID, Field.ELEVATION])
        df_bot = df_bot.drop(df_bot.groupby(Field.RELATEID).tail(1).index)

        self.df = pd.concat([df_top, df_bot])

    def create_kdtree(self):
        """
        Creates a KDTree to optimize finding the K nearest neighbors to a given borehole using UTM as its metric
        Args:
        Returns:
        """

        utm = self.df[[Field.UTME, Field.UTMN]].value_counts().index

        self.kdtree = KDTree(utm.values.tolist())

    def compute_min_distance(self, utme, utmn, elevation, label, strat):
        """
        Calculates the minimum signed distance to a known boundary point
        Args:
            utme: UTME
            utmn: UTMN
            elevation: Elevation
            label: Formation Label
            strat: Point Label

        Returns: Minimum signed distance
        """
        _, points = self.kdtree.query([[utme, utmn]], k=self.k)
        points = points[0]

        utme_points = self.df.iloc[points][Field.UTME]
        utmn_points = self.df.iloc[points][Field.UTMN]

        df = self.df[(self.df[Field.UTME].isin(utme_points)) & (self.df[Field.UTMN].isin(utmn_points)) & (self.df[Field.STRAT] == label)]

        if df.empty:
            return -1000

        df['distance'] = np.sqrt((df[Field.UTME] - utme) ** 2 + (df[Field.UTMN] - utmn) ** 2 + ((df[Field.ELEVATION] - elevation)/self.meter_const) ** 2)

        min_dist = df['distance'].min() * self.meter_const

        if min_dist < -1000:
            return -1000

        if strat == label:
            return min_dist
        else:
            return -min_dist

    def compute_all(self, utme, utmn, elevation, strat):
        """
        Computes all the signed distances for a list of points
        Args:
            utme: UTME list
            utmn: UTMN list
            elevation: Elevation list
            strat: Point label list

        Returns: List of signed distances
        """
        dist = []

        for idx in range(len(utme)):
            lst = []

            for label in range(self.max_label):
                lst.append(self.compute_min_distance(utme[idx], utmn[idx], elevation[idx], label, strat[idx]))

            dist.append(lst)

        return dist