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


COLOR_ABBREV_MAP = {
    'BLK': 'BLACK',
    'BLU': 'BLUE',
    'BRN': 'BROWN',
    'GRY': 'GRAY',
    'GRN': 'GREEN',
    'OLV': 'BROWN',
    'ONG': 'ORANGE',
    'PNK': 'PINK',
    'PUR': 'PURPLE',
    'RED': 'RED',
    'TAN': 'BROWN',
    'SLV': 'GRAY',
    'WHT': 'WHITE',
    'YEL': 'YELLOW',
    'VARIED': 'VARIED'
}
COLORS = sorted(set(COLOR_ABBREV_MAP.values()))

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

FINAL_BEDROCK_UNITS = [
    'Ku', 'Kc', 'Ka',
    'Dmu', 'Dm',
    'Ou', 'Omu', 'Ol',
    'Cu', 'Cmu',
    'Mss',
    'Mdb', 'Mbg', 'Mbf', 'Mbt', 'Mlf', 'Mlg', 'Mlt', 'Mlc', 'Mau', 'Mge', 'Mfg', 'Mmd', 'Mmg', 'Mmf',
    'Mcv', 'Mbv', 'Mmv', 'Mfv', 'Msl', 'Mns', 'Mnu', 'Mnr', 'Mnb', 'Mnl', 'Mms', 'Mvu', 'Mld', 'Mmi',
    'PMm', 'Psq', 'Pmi', 'Pmy', 'Pgu', 'Pgr', 'Pgk', 'Pgd', 'Pgp', 'Pga', 'Pgm', 'Pdt', 'Pgt', 'Pdg', 'Pas', 'Pac', 'Pai',
    'Paq', 'Pag', 'Psi', 'Pgs', 'Pml', 'Pls', 'Pm', 'Pif', 'Pvs', 'Pmv' ,'Pmq', 'Pmd', 'Pgn',
    'Aml', 'Ami', 'Apg', 'Apd', 'Apm', 'Apv', 'Agr', 'Agu', 'Asd', 'Agm', 'Agp', 'Aqm', 'Agd', 'Ast', 'Adt', 'Aqp', 'Agl',
    'Agt', 'Agn', 'Ags', 'Aql', 'Aqg', 'Aqt', 'Aqs', 'Aqa', 'Asc', 'Aks', 'Akc', 'Akv', 'Ams', 'Aif', 'Asg', 'Avs', 'Acv',
    'Aag', 'Amv', 'Auv', 'Amm', 'Amg', 'Amt', 'Amd', 'Amn'
]

BEDROCK_PARENT_MAP = {
    # Undifferentiated
    'CAMB' : 'CAMB',
    'PUDF' : 'PUDF',
    'DEVO' : 'DEVO',
    'KRET' : 'KRET',
    'ORDO' : 'ORDO',

    # Cretaceous Regolith
    'KREG' : 'KREG',

    # Greenhorn
    'KGRN' : 'Ku', #

    # Ganeros Shale
    'KGRS' : 'Ku', #

    # Carlile Shale
    'KCBH': 'KCRL',
    'KCCD': 'KCRL',
    'KCFP': 'KCRL',
    'KCRL': 'Ku', #

    # Coleraine
    'KCLR': 'Kc', #

    # Split Rock Creek
    'KSRC': 'Ka', #

    # Windrow
    'KWIH': 'KWND',
    'KWND': 'Ku', #
    'KWOS': 'KWND',

    # Dakota Sandstone
    'KDKT': 'Ka', #
    'KDNB': 'KDKT',
    'KDWB': 'KDKT',


    # Upper Cedar
    'DCVU': 'DCVA', #

    # Lithograph City Formation
    'DLGH': 'DCVU', #

    # Coralville Formation
    'DCRL': 'DCVU', #
    'DCUM': 'DCRL',
    'DCGZ': 'DCRL',
    'DCIC': 'DCRL',
    'DCLC': 'DCRL',

    # Cedar Valley Group
    'DCVA': 'Dmu', #

    # Lower Cedar
    'DCLP': 'DCVL',
    'DCLS': 'DCVL',
    'DCOG': 'DCVL',
    'DCOM': 'DCVL',
    'DCVL': 'DCVA', #

    # Little Cedar Formation
    'DLBA': 'DLCD',
    'DLCB': 'DLCD',
    'DLCD': 'DCVL', #
    'DLCH': 'DLCD',
    'DLHE': 'DLCD',

    # Wapsipinicon Group
    'DWAP': 'Dm', #

    # Pinicon Ridge
    'DWPR': 'DWAP', #
    'DPOG': 'DWPR',
    'DPOM': 'DWPR',

    # Spillville
    'DSOG': 'DSPL',
    'DSOM': 'DSPL',
    'DSPL': 'DWAP', #


    # Red River
    'ORRV': 'Ou', #

    # Winnipeg
    'OWBI': 'OWIN',
    'OWIB': 'OWIN',
    'OWIN': 'Ou', #

    # Maquoketa Formation
    'OMAQ' : 'Ou', #
    'OMQD' : 'OMAQ',
    'OMQG' : 'OMAQ',

    # Galena Group
    'OGGP' : 'Ou', #
    'OGAP' : 'OGGP',
    'OGDP' : 'OGGP',
    'OGPD' : 'OGGP',

    # Dubuque
    'ODGL' : 'ODUB',
    'ODUB' : 'OGGP', #

    # Stewartville
    'OGSC' : 'OGSV',
    'OGSD' : 'OGSV',
    'OGVP' : 'OGSV',
    'OGSV' : 'OGGP', #

    # Prosser
    'OGPC' : 'OGPR',
    'OGPR' : 'OGGP', #

    # Cummingsville
    'OGCM' : 'OGGP', #
    'OGCD' : 'OGCM',

    # Decorah Shale
    'ODCA' : 'ODCR',
    'ODCR' : 'OGGP', #
    'ODPG' : 'ODCR',
    'ODPL' : 'ODCR',
    'ODSP' : 'ODCR',

    # Platteville
    'OPGW' : 'OPVL',
    'OPHF' : 'OPVL',
    'OPMA' : 'OPVL',
    'OPMI' : 'OPVL',
    'OPPE' : 'OPVL',
    'OPSP' : 'OPVL',
    'OPVJ' : 'OPVL',
    'OPVL' : 'Omu', #

    # Glenwood
    'OGSP' : 'OGWD',
    'OGWD' : 'Omu', #

    # St. Peter Sandstone
    'OSCJ' : 'OSTP',
    'OSCS' : 'OSTP',
    'OSPC' : 'OSTP',
    'OSPE' : 'OSTP',
    'OSTN' : 'OSTP',
    'OSTP' : 'Omu', #

    # Praire Du Chien
    'OPCJ' : 'OPDC',
    'OPCM' : 'OPDC',
    'OPCS' : 'OPDC',
    'OPCT' : 'OPDC',
    'OPDC' : 'Ol', #

    # Shakopee
    'OPNR' : 'OPSH',
    'OPWR' : 'OPSH',
    'OPSH' : 'OPDC', #

    # Oneota Dolomite
    'OOCV' : 'OPOD',
    'OOHC' : 'OPOD',
    'OPOD' : 'OPDC', #

    # Stoney Mountain
    'OSTM' : 'ORDO', #


    # Jordan Sandstone
    'CJDN' : 'Cu', #
    'CJEC' : 'CJDN',
    'CJDW' : 'CJDN',
    'CJMS' : 'CJDN',
    'CJSL' : 'CJDM',
    'CJTC' : 'CJDN',

    # St. Lawrence
    'CSLT' : 'Cu', #
    'CSLW' : 'CSTL',
    'CSTL' : 'CSTL',

    # Tunnel City
    'CTCG' : 'Cu', #
    'CTCM' : 'CTCG',
    'CTCW' : 'CTCG',
    'CTCE' : 'CTCG',

    # Mazomanie
    'CTMZ' : 'CSLT', #

    # Lone Rock
    'CTLR' : 'CSLT', #
    'CLBK' : 'CTLR',
    'CLRE' : 'CTLR',
    'CLTM' : 'CTLR',

    # Wonewoc
    'CWMS' : 'CWOC',
    'CWOC' : 'Cmu', #
    'CWEC' : 'CWOC',

    # Eau Claire
    'CECR' : 'Cmu', #
    'CEMS' : 'CECR',

    # Mt. Simon
    'CMFL' : 'CMTS',
    'CMRC' : 'CMTS',
    'CMSH' : 'CMTS',
    'CMTS' : 'Cmu', #


    # Hinckley Sandstone
    'PMHN' : 'Mss', #
    'PMHF' : 'PMHN',

    # Fond Du Lac
    'PMFL' : 'Mss', #

    # Solor Church
    'PSMC' : 'Mss', #
}
