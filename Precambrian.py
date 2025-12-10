import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

from Data import Field

def encode_precambrian(df):
    """
    Determines whether the layer has a precambrian component, and if it does it extracts and encodes it
    :param df: Layer dataframe
    :return: Precambrian one hot encoded dataframe
    """

    for category in CATEGORY_LIST:
        df[category.name] = False

    for lithology in LITHOLOGY_LIST:
        df[lithology.name] = False

    code_cols = {}

    for key, value in PRECAMBRIAN_LITHOLOGY_MAP.items():
        cols = []

        for bedrock in value.bedrocks:
            for unit in bedrock.get_lineage():
                if isinstance(unit, Lithology):
                    cols.append(unit.name)

        code_cols[key] = cols

    for key, value in code_cols.items():
        mask = (df[Field.LITH_PRIM] == key) & (df[Field.STRAT].str.startswith('P', na=False))

        if value:
            df.loc[mask, list(set(value))] = True

    lith_cols = {}

    for key, value in PRECAMBRIAN_MAP.items():
        cols = []

        for lithology in value.bedrocks:
            for unit in lithology.get_lineage():
                if isinstance(unit, Lithology):
                    cols.append(unit.name)

        lith_cols[key] = cols

    for key, value in lith_cols.items():
        mask = df[Field.STRAT].astype(str).str.endswith(key)

        if value:
            df.loc[mask, list(set(value))] = True

    return df


class Precambrian:

    def __init__(self, name : str, parents=None):
        self.name = name

        if parents is None:
            self.parents = []
        elif isinstance(parents, list):
            self.parents = parents
        else:
            self.parents = [parents]

    def get_lineage(self):
        lineage = []
        visited = set()
        stack = [self]

        while stack:
            unit = stack.pop()
            if unit in visited:
                continue
            visited.add(unit)
            lineage.append(unit)

            stack.extend(unit.parents)

        return lineage

class Age(Precambrian):
    pass

class Lithology(Precambrian):
    pass

class PrecambrianCode:

    def __init__(self, bedrocks):
        if isinstance(bedrocks, list):
            self.bedrocks = bedrocks
        else:
            self.bedrocks = [bedrocks]

    def __eq__(self, other):

        if not isinstance(other, PrecambrianCode):
            return False

        ages = self.ages == other.ages
        lithology = self.lithology == other.lithology

        return ages and lithology

    @property
    def ages(self):
        lst = set()

        for bedrock in self.bedrocks:
            lineage = bedrock.get_lineage()

            for item in lineage:
                if isinstance(item, Age):
                    lst.add(item)

        return list(lst)

    @property
    def lithology(self):
        lst = set()

        for bedrock in self.bedrocks:
            lineage = bedrock.get_lineage()

            for item in lineage:
                if isinstance(item, Lithology):
                    lst.add(item)

        return list(lst)

### PRECAMBRIAN AGES ###
Archean = Age('Archean')
Early_Proterozoic = Age('Early Proterozoic')
Mesoproterozoic = Age('Mesoproterozoic')

AGE_LIST = [
    Archean,
    Early_Proterozoic,
    Mesoproterozoic,
]

"""PRECAMBRIAN CATEGORIES"""
Metamorphic = Lithology('Metamorphic')
Sedimentary = Lithology('Sedimentary')
Metasediment = Lithology('Metasediment', parents=Metamorphic)
Volcanic = Lithology('Volcanic')
Intrusive = Lithology('Intrusive')
Felsic = Lithology('Felsic')
Intermediate = Lithology('Intermediate')
Mafic = Lithology('Mafic')
Ultramafic = Lithology('Ultramafic')
Fault = Lithology('Fault')

CATEGORY_LIST = [
    Metamorphic,
    Sedimentary,
    Metasediment,
    Volcanic,
    Intrusive,
    Felsic,
    Intermediate,
    Mafic,
    Ultramafic,
    Fault
]

"""PRECAMBRIAN LITHOLOGY"""
Slate = Lithology('Slate', parents=Metasediment)
Argillite = Lithology('Argillite', parents=Metasediment)
Phyllite = Lithology('Phyllite', parents=Metasediment)
Schist = Lithology('Schist', parents=Metasediment)
Gneiss = Lithology('Gneiss', parents=Metasediment)
Marble = Lithology('Marble', parents=Metasediment)
Quartzite = Lithology('Quartzite', parents=Metasediment)
Metapelite = Lithology('Metapelite', parents=Metasediment)
Metapsammite = Lithology('Metapsammite', parents=Metasediment)
Amphibolite = Lithology('Amphibolite')
Fault_Breccia = Lithology('Fault Breccia', parents=Fault)
Fault_Gouge = Lithology('Fault Gouge', parents=Fault)
Brittle_Fault = Lithology('Brittle Fault', parents=Fault)
Protomylonite = Lithology('Protomylonite')
Mylonite = Lithology('Mylonite')
Ultramylonite = Lithology('Ultramylonite')
Quartz_Vein = Lithology('Quartz Vein')
Pegmatite = Lithology('Pegmatite')
Diabase = Lithology('Diabase')
Aplite = Lithology('Aplite')
Conglomerate = Lithology('Conglomerate', parents=Sedimentary)
Carbonate = Lithology('Carbonate', parents=Sedimentary)
Sandstone = Lithology('Sandstone', parents=Sedimentary)
Siltstone = Lithology('Siltstone', parents=Sedimentary)
Shale = Lithology('Shale', parents=Sedimentary)
Chert = Lithology('Chert', parents=Sedimentary)
Iron_Formation = Lithology('Iron Formation', parents=Sedimentary)

LITHOLOGY_LIST = [
    Slate,
    Argillite,
    Phyllite,
    Schist,
    Gneiss,
    Marble,
    Quartzite,
    Metapelite,
    Metapsammite,
    Amphibolite,
    Fault_Breccia,
    Fault_Gouge,
    Brittle_Fault,
    Protomylonite,
    Mylonite,
    Ultramylonite,
    Iron_Formation,
    Sandstone,
    Diabase,
    Siltstone,
    Conglomerate,
    Chert,
    Quartz_Vein,
    Pegmatite,
    Aplite,
    Carbonate,
    Shale
]

### MAPPING LITHOLOGIES INSTEAD OF CODES ###
PRECAMBRIAN_LITHOLOGY_MAP = {
    'IRFM' : PrecambrianCode(Iron_Formation),
    'BSLT' : PrecambrianCode([Volcanic, Mafic]),
    'SLTE' : PrecambrianCode(Slate),
    'SNDS' : PrecambrianCode(Sandstone),
    'GRAN' : PrecambrianCode([Intrusive, Felsic]),
    'TROC' : PrecambrianCode([Mafic, Intrusive]),
    'GBRO' : PrecambrianCode([Intrusive, Mafic]),
    'QRTZ' : PrecambrianCode(Quartzite),
    'SHLE' : PrecambrianCode(Shale),
    'GRWK' : PrecambrianCode(Sandstone),
    'QZMZ' : PrecambrianCode([Intrusive, Felsic]),
    'DIAB' : PrecambrianCode(Diabase),
    'GNIS' : PrecambrianCode(Gneiss),
    'SHST' : PrecambrianCode(Schist),
    'HNFL' : PrecambrianCode(Metamorphic),
    'VOLU' : PrecambrianCode(Volcanic),
    'THOL' : PrecambrianCode([Volcanic, Mafic]),
    'RHYL' : PrecambrianCode([Volcanic, Felsic]),
    'ANDS' : PrecambrianCode([Volcanic, Intermediate]),
    'AGTR' : PrecambrianCode([Intrusive, Mafic]),
    'PRDT' : PrecambrianCode([Ultramafic, Intrusive]),
    'ANOR' : PrecambrianCode([Intrusive, Felsic]),
    'MSED' : PrecambrianCode(Metasediment),
    'MGMT' : PrecambrianCode([Metamorphic]),
    'ANTR' : PrecambrianCode([Mafic, Intrusive]),
    'TUFF' : PrecambrianCode([Volcanic, Felsic]),
    'ARGI' : PrecambrianCode(Argillite),
    'MGWK' : PrecambrianCode(Metasediment),
    'NORT' : PrecambrianCode([Mafic, Intrusive]),
    'ORGN' : PrecambrianCode(Gneiss),
    'PYTR' : PrecambrianCode([Mafic, Intrusive]),
    'TACT' : PrecambrianCode(Iron_Formation),
    'TONT' : PrecambrianCode([Intrusive, Intermediate]),
    'DIOR' : PrecambrianCode([Intrusive, Intermediate]),
    'BIOS' : PrecambrianCode(Schist),
    'GRDI' : PrecambrianCode([Intrusive, Intermediate, Felsic]),
    'OLGB' : PrecambrianCode([Intrusive, Mafic]),
    'SLSN' : PrecambrianCode(Siltstone),
    'MAFU' : PrecambrianCode(Mafic),
    'OXOG' : PrecambrianCode([Intrusive, Mafic]),
    'MTRO' : PrecambrianCode([Mafic, Intrusive]),
    'ICEL' : PrecambrianCode([Volcanic, Intermediate]),
    'FRDI' : PrecambrianCode([Intrusive, Intermediate]),
    'GRPH' : PrecambrianCode([Felsic, Intrusive]),
    'CNGL' : PrecambrianCode(Conglomerate),
    'AMPH' : PrecambrianCode(Amphibolite),
    'CHRT' : PrecambrianCode(Chert),
    'SYEN' : PrecambrianCode([Intrusive, Mafic]),
    'GBNO' : PrecambrianCode([Intrusive, Mafic]),
    'MGBR' : PrecambrianCode([Intrusive, Mafic]),
    'DACT' : PrecambrianCode([Volcanic, Intermediate, Felsic]),
    'QZAR' : PrecambrianCode(Sandstone),
    'DUNT' : PrecambrianCode([Ultramafic, Intrusive]),
    'TRAN' : PrecambrianCode([Intrusive, Mafic]),
    'MYLN' : PrecambrianCode(Mylonite),
    'FELS' : PrecambrianCode([Volcanic, Felsic]),
    'QUTZ' : PrecambrianCode(Quartz_Vein),
    'MDSN' : PrecambrianCode(Sedimentary),
    'OBTR' : PrecambrianCode([Mafic, Intrusive]),
    'QZPY' : PrecambrianCode([Quartz_Vein]),
    'PICR' : PrecambrianCode([Volcanic, Ultramafic]),
    'INTR' : PrecambrianCode(Intrusive),
    'PYRX' : PrecambrianCode([Ultramafic, Intrusive]),
    'PHYL' : PrecambrianCode(Phyllite),
    'MNZT' : PrecambrianCode([Intrusive, Intermediate]),
    'IRON' : PrecambrianCode(Iron_Formation),
    'FALT' : PrecambrianCode(Fault),
    'VOLM' : PrecambrianCode([Volcanic, Mafic]),
    'MARL' : PrecambrianCode(Sedimentary),
    'MFGN' : PrecambrianCode([Gneiss, Mafic]),
    'UMFU' : PrecambrianCode(Ultramafic),
    'VOLI' : PrecambrianCode([Volcanic, Intermediate]),
    'FEGN' : PrecambrianCode([Felsic, Gneiss]),
    'BXTT' : PrecambrianCode([Fault_Breccia, Metamorphic]),
    'BXVL' : PrecambrianCode([Fault_Breccia, Volcanic]),
    'FFBX' : PrecambrianCode([Fault_Breccia, Volcanic]),
    'BXFT' : PrecambrianCode([Fault_Breccia, Volcanic]),
    'MFBX' : PrecambrianCode([Fault_Breccia, Volcanic, Mafic]),
    'TUBX' : PrecambrianCode([Fault_Breccia, Volcanic, Felsic]),
    'BXSD' : PrecambrianCode([Fault_Breccia, Sedimentary]),
    'BXIN' : PrecambrianCode([Fault_Breccia]),
    'FIBX' : PrecambrianCode([Fault_Breccia, Intrusive, Felsic]),
    'INBX' : PrecambrianCode([Fault_Breccia]),
    'MIBX' : PrecambrianCode([Fault_Breccia, Intrusive, Mafic]),
}

"""CATEGORY/LITHOLOGY MAP"""
PRECAMBRIAN_MAP = {
    'FI' : PrecambrianCode([Felsic, Intrusive]),
    'II' : PrecambrianCode([Intermediate, Intrusive]),
    'IM' : PrecambrianCode([Mafic, Intrusive]),
    'UI' : PrecambrianCode([Ultramafic, Intrusive]),
    'SA' : PrecambrianCode(Slate),
    'RT' : PrecambrianCode(Argillite),
    'PH' : PrecambrianCode(Phyllite),
    'CH' : PrecambrianCode(Schist),
    'GN' : PrecambrianCode(Gneiss),
    'MA' : PrecambrianCode(Marble),
    'QA' : PrecambrianCode(Quartzite),
    'PE' : PrecambrianCode(Metapelite),
    'PS' : PrecambrianCode(Metapsammite),
    'FE' : PrecambrianCode([Felsic, Gneiss]),
    'IG' : PrecambrianCode([Intermediate, Gneiss]),
    'MF' : PrecambrianCode([Mafic, Gneiss]),
    'FS' : PrecambrianCode([Felsic, Schist]),
    'IS' : PrecambrianCode([Intermediate, Schist]),
    'MH' : PrecambrianCode([Mafic, Schist]),
    'AM' : PrecambrianCode(Amphibolite),
    'FX' : PrecambrianCode(Fault_Breccia),
    'FO' : PrecambrianCode(Fault_Gouge),
    'FB' : PrecambrianCode(Brittle_Fault),
    'PM' : PrecambrianCode(Protomylonite),
    'MY' : PrecambrianCode(Mylonite),
    'MM' : PrecambrianCode(Ultramylonite),
    'MS' : PrecambrianCode(Metasediment),
    'QV' : PrecambrianCode(Quartz_Vein),
    'PD' : PrecambrianCode(Pegmatite),
    'DD' : PrecambrianCode(Diabase),
    'AD' : PrecambrianCode(Aplite),
    'UV' : PrecambrianCode([Ultramafic, Volcanic]),
    'VM' : PrecambrianCode([Mafic, Volcanic]),
    'IV' : PrecambrianCode([Intermediate, Volcanic]),
    'VF' : PrecambrianCode([Felsic, Volcanic]),
    'JM' : PrecambrianCode([Mafic, Intermediate, Volcanic]),
    'JF' : PrecambrianCode([Intermediate, Felsic, Volcanic]),
    'CG' : PrecambrianCode(Conglomerate),
    'CR' : PrecambrianCode(Carbonate),
    'DS' : PrecambrianCode(Sandstone),
    'SI' : PrecambrianCode(Siltstone),
    'SE' : PrecambrianCode(Shale),
    'CT' : PrecambrianCode(Chert),
    'IF' : PrecambrianCode(Iron_Formation)
}