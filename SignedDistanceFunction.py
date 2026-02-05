import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder

import Data, utils
from Data import Field

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

class SignedDistanceFunction:

    def __init__(self, df, max_label):
        self.kdtree = None
        self.utm = None
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

        df_top = self.df.drop(columns=[Field.ELEVATION_BOT, Field.STRAT, 'strat_bot']).rename(columns={Field.ELEVATION_TOP: Field.ELEVATION,
                                                                             'strat_top' : Field.STRAT})
        df_bot = self.df.drop(columns=[Field.ELEVATION_TOP, Field.STRAT, 'strat_top']).rename(columns={Field.ELEVATION_BOT: Field.ELEVATION,
                                                                             'strat_bot' : Field.STRAT})

        """Remove the top most reading of a borehole to simplify the model and use actual elevation as a cutoff"""
        df_top = df_top.sort_values([Field.RELATEID, Field.ELEVATION]).reset_index(drop=True)
        df_top = df_top.groupby(Field.RELATEID, group_keys=False).apply(lambda x: x.iloc[1:])

        """Remove the bottom most reading of a borehole due to uncertainty"""
        df_bot = df_bot.sort_values([Field.RELATEID, Field.ELEVATION]).reset_index(drop=True)
        df_bot = df_bot.groupby(Field.RELATEID, group_keys=False).apply(lambda x: x.iloc[:-1])

        df_top = df_top.reset_index(drop=True)
        df_bot = df_bot.reset_index(drop=True)

        self.df = pd.concat([df_top, df_bot])

        encoder = LabelEncoder()
        self.df[Field.STRAT] = encoder.fit_transform(self.df[Field.STRAT])
        self.df = self.df.dropna()

    def create_kdtree(self):
        """
        Creates a KDTree to optimize finding the K nearest neighbors to a given borehole using UTM as its metric
        Args:
        Returns:
        """

        self.utm = self.df[[Field.UTME, Field.UTMN]].drop_duplicates().values

        self.kdtree = KDTree(self.utm)

    def find_nearest_holes(self, utme, utmn, elevation):
        """
        Computes the k-nearest boreholes and the distances to them
        Args:
            utme: UTME
            utmn: UTMN
            elevation: Elevation

        Returns: Nearest boreholes Dataframe with distances
        """
        _, points = self.kdtree.query([[utme, utmn]], k=self.k)
        points = points[0]

        utm = self.utm[points]

        utm_df = pd.DataFrame(utm, columns=[Field.UTME, Field.UTMN])
        utm_df['nearest'] = True

        df = self.df.merge(utm_df, on=[Field.UTME, Field.UTMN], how='inner').copy()
        df.drop(columns=['nearest'], inplace=True)

        df['distance'] = np.sqrt((df[Field.UTME] - utme) ** 2 + (df[Field.UTMN] - utmn) ** 2 + ((df[Field.ELEVATION] - elevation) / self.meter_const) ** 2)

        return df

    def compute_min_distance(self, label, strat, df):
        """
        Calculates the minimum signed distance to a known boundary point
        Args:
            label: Formation Label
            strat: Point Label
            df: Dataframe of nearest boreholes

        Returns: Minimum signed distance
        """

        df = df[df[Field.STRAT] == label]

        if df.empty:
            return -6

        min_dist = np.log(df['distance'].min() + 1)

        if min_dist > 6:
            min_dist = 6

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
            df = self.find_nearest_holes(utme[idx], utmn[idx], elevation[idx])

            for label in range(self.max_label):
                lst.append(self.compute_min_distance(label, strat[idx], df))

            dist.append(lst)

        return dist