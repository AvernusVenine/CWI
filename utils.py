import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import random

SCALED_COLUMNS = ['true_depth_top', 'true_depth_bot', 'elevation', 'utme', 'utmn']
TRUSTED_SOURCES = []

AGE_DROP_COLS = ['relateid', 'age_cat', 'strat', 'color', 'drllr_desc', 'data_src']
QUAT_DROP_COLS = []
BEDROCK_DROP_COLS = []

AGE_CATEGORIES = {
    'A': -1,
    'C': 0,
    'D': 1,
    'F': 2,
    'K': 3,
    'O': 4,
    'P': 5,
    'Q': 6,
    'R': 7,
    'U': 8,
    'X': 9,
    'Y': 10,
    'Z': 11
}
INV_AGE_CATEGORIES = {v: k for k, v in AGE_CATEGORIES.items()}
QUAT_CATEGORIES = {
    'B': 0,
    'C': 1,
    'F': 2,
    'G': 3,
    'H': 4,
    'I': 5,
    'J': 6,
    'L': 7,
    'N': 8,
    'P': 9,
    'S': 10,
    'T': 11,
    'U': 12,
    'W': 13
}
INV_QUAT_CATEGORIES = {v: k for k, v in QUAT_CATEGORIES.items()}
BEDROCK_CATEGORIES = {
    'PUNK' : 0,
    'CAMB' : 1, # Camb Und
    'PUDF' : 2, # Precamb Und
    'CJDN' : 3, # Jordan


}
INV_BEDROCK_CATEGORIES = {v: k for k, v in BEDROCK_CATEGORIES.items()}

BEDROCK_AGES = ['C', 'D', 'K', 'O', 'P', 'U']

PRECAMBRIAN_UNKNOWN = 'PUNK'

BEDROCK_PARENT_MAP = {
    # Jordan Sandstone
    'CJDN' : 'CJDN',
    'CJEC' : 'CJDN',
    'CJDW' : 'CJDN',
    'CJMS' : 'CJDN',
    'CJSL' : 'CJDM',
    'CJTC' : 'CJDN',

    # Mt. Simon
    'CMFL' : 'CMTS',
    'CMRC' : 'CMTS',
    'CMSH' : 'CMTS',
    'CMTS' : 'CMTS',

    # St. Lawrence
    'CSLT' : 'CSTL',
    'CSLW' : 'CSTL',
    'CSTL' : 'CSTL',

    # Tunnel City
    'CTCG' : 'CTCG',
    'CTCM' : 'CTCG',
    'CTCW' : 'CTCG',
    'CTLR' : 'CTCG',
    'CTCE' : 'CTCG',
    'CTMZ' : 'CTCG',

    'CLBK' : 'CTCG',
    'CLRE' : 'CTCG',
    'CLTM' : 'CTCG',

    # Wonewoc
    'CWMS' : 'CWOC',
    'CWOC' : 'CWOC',
    'CWEC' : 'CWOC',

    # Coralville Formation
    'DCRL' : 'DCRL',
    'DCUM' : 'DCRL',
    'DCGZ' : 'DCRL',
    'DCIC' : 'DCRL',
    'DCLC' : 'DCRL',

    # Lower Cedar
    'DCLP' : 'DCVL',
    'DCLS' : 'DCVL',
    'DCOG' : 'DCVL',
    'DCOM' : 'DCVL',
    'DCVL' : 'DCVL',

    # Upper Cedar
    'DCVU' : 'DCVU',

    # Cedar Valley Group
    'DCVA' : 'DCVA',

    # Little Cedar Formation
    'DLBA' : 'DLCD',
    'DLCB' : 'DLCD',
    'DLCD' : 'DLCD',
    'DLCH' : 'DLCD',
    'DLHE' : 'DLCD',

    # Lithograph City Formation
    'DLGH' : 'DLGH',

    # Wapsipinicon Group
    'DWAP' : 'DWPR',
    'DWPR' : 'DWPR',

    'DPOG' : 'DWPR',
    'DPOM' : 'DWPR',

    # Spillville
    'DSOG' : 'DSPL',
    'DSOM' : 'DSPL',
    'DSPL' : 'DSPL',

    # Carlile Shale
    'KCBH' : 'KCRL',
    'KCCD' : 'KCRL',
    'KCFP' : 'KCRL',
    'KCRL' : 'KCRL',

    'KGRN' : 'KCRL',

    'KGRS' : 'KCRL',

    # Coleraine
    'KCLR' : 'KCLR',

    # Dakota Sandstone (Might have to remove)
    'KDKT' : 'KDKT',
    'KDNB' : 'KDKT',
    'KDWB' : 'KDKT',

    # Cretaceous Regolith
    'KREG' : 'KREG',

    # Split Rock Creek
    'KSRC' : 'KRSC',

    # Windrow
    'KWIH' : 'KWND',
    'KWND' : 'KWND',
    'KWOS' : 'KWND',

    # Decorah Shale
}
