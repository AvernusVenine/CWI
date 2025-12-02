import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

from Data import Field
from Bedrock import GeoCode

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
        mask = df[Field.STRAT] == key

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

### CATEGORIES ###
Metamorphic = Lithology('Metamorphic')
Igneous = Lithology('Igneous')
Sedimentary = Lithology('Sedimentary')
Metasediment = Lithology('Metasediment', parents=Metamorphic)
Metaigneous = Lithology('Metaigneous', parents=Metamorphic)
Volcanic = Lithology('Volcanic', parents=Igneous)
Intrusive = Lithology('Intrusive', parents=Igneous)
Felsic = Lithology('Felsic')
Intermediate = Lithology('Intermediate')
Mafic = Lithology('Mafic')
Ultramafic = Lithology('Ultramafic')
Fault = Lithology('Fault')
Dike = Lithology('Dike')

CATEGORY_LIST = [
    Metamorphic,
    Igneous,
    Sedimentary,
    Metasediment,
    Metaigneous,
    Volcanic,
    Intrusive,
    Felsic,
    Intermediate,
    Mafic,
    Ultramafic,
    Fault,
    Dike
]

### PRECAMBRIAN LITHOLOGY ###
Slate = Lithology('Slate', parents=Metasediment)
Argillite = Lithology('Argillite', parents=Metasediment)
Phyllite = Lithology('Phyllite', parents=Metasediment)
Schist = Lithology('Schist', parents=Metasediment)
Gneiss = Lithology('Gneiss', parents=[Metasediment, Metaigneous])
Marble = Lithology('Marble', parents=Metasediment)
Quartzite = Lithology('Quartzite', parents=Metasediment)
Metapelite = Lithology('Metapelite', parents=Metasediment)
Metapsammite = Lithology('Metapsammite', parents=Metasediment)
Felsic_Gneiss = Lithology('Felsic Gneiss', parents=Gneiss)
Intermediate_Gneiss = Lithology('Intermediate Gneiss', parents=Gneiss)
Mafic_Gneiss = Lithology('Mafic Gneiss', parents=Gneiss)
Felsic_Schist = Lithology('Felsic Schist', parents=Schist)
Intermediate_Schist = Lithology('Intermediate Schist', parents=Schist)
Mafic_Schist = Lithology('Mafic Schist', parents=Schist)
Calcium_Silicate_Schist = Lithology('Calcium Silicate Schist', parents=Schist)
Amphibolite = Lithology('Amphibolite', parents=Metaigneous)
Fault_Breccia = Lithology('Fault Breccia', parents=Fault)
Fault_Gouge = Lithology('Fault Gouge', parents=Fault)
Brittle_Fault = Lithology('Brittle Fault', parents=Fault)
Protomylonite = Lithology('Protomylonite', parents=Fault)
Mylonite = Lithology('Mylonite', parents=Fault)
Ultramylonite = Lithology('Ultramylonite', parents=Fault)
Quartz_Vein = Lithology('Quartz Vein')
Pegmatite = Lithology('Pegmatite')
Diabase = Lithology('Diabase', parents=Dike)
Aplite = Lithology('Aplite', parents=Dike)
Conglomerate = Lithology('Conglomerate', parents=Sedimentary)
Carbonate = Lithology('Carbonate', parents=Sedimentary)
Sandstone = Lithology('Sandstone', parents=Sedimentary)
Siltstone = Lithology('Siltstone', parents=Sedimentary)
Shale = Lithology('Shale', parents=Sedimentary)
Chert = Lithology('Chert', parents=Sedimentary)
Iron_Formation = Lithology('Iron Formation', parents=Sedimentary)

LITHOLOGY_LIST = []

### MAPPING LITHOLOGIES INSTEAD OF CODES ###
PRECAMBRIAN_LITHOLOGY_MAP = {
    'IRFM' : GeoCode(Iron_Formation),
    'BSLT' : GeoCode([Volcanic, Mafic]),
    'SLTE' : GeoCode(Slate),
    'SNDS' : GeoCode(Sandstone),
    'GRAN' : GeoCode([Intrusive, Felsic]),
    'TROC' : GeoCode([Mafic, Intrusive]),
    'GBRO' : GeoCode([Intrusive, Mafic]),
    'QRTZ' : GeoCode(Quartzite),
    'SHLE' : GeoCode(Shale),
    'GRWK' : GeoCode(Sandstone),
    'QZMZ' : GeoCode([Intrusive, Felsic]),
    'DIAB' : GeoCode(Diabase),
    'GNIS' : GeoCode(Gneiss),
    'SHST' : GeoCode(Schist),
    'HNFL' : GeoCode(Metamorphic),
    'VOLU' : GeoCode(Volcanic),
    'THOL' : GeoCode([Volcanic, Mafic]),
    'RHYL' : GeoCode([Volcanic, Felsic]),
    'ANDS' : GeoCode([Volcanic, Intermediate]),
    'AGTR' : GeoCode([Intrusive, Mafic]),
    'PRDT' : GeoCode([Ultramafic, Intrusive]),
    'ANOR' : GeoCode([Intrusive, Felsic]),
    'MSED' : GeoCode(Metasediment),
    'MGMT' : GeoCode([Metamorphic, Igneous]),
    'ANTR' : GeoCode([Mafic, Intrusive]),
    'TUFF' : GeoCode([Volcanic, Felsic]),
    'ARGI' : GeoCode(Argillite),
    'MGWK' : GeoCode(Metasediment),
    'NORT' : GeoCode([Mafic, Intrusive]),
    'MBLT' : GeoCode(Metaigneous),
    'ORGN' : GeoCode(Gneiss),
    'PYTR' : GeoCode([Mafic, Intrusive]),
    'TACT' : GeoCode(Iron_Formation),
    'TONT' : GeoCode([Intrusive, Intermediate]),
    'DIOR' : GeoCode([Intrusive, Intermediate]),
    'BIOS' : GeoCode(Schist),
    'GRDI' : GeoCode([Intrusive, Intermediate, Felsic]),
    'OLGB' : GeoCode([Intrusive, Mafic]),
    'SLSN' : GeoCode(Siltstone),
    'MAFU' : GeoCode(Mafic),
    'OXOG' : GeoCode([Intrusive, Mafic]),
    'MTRO' : GeoCode([Mafic, Intrusive]),
    'MVOL' : GeoCode(Metaigneous),
    'ICEL' : GeoCode([Volcanic, Intermediate]),
    'FRDI' : GeoCode([Intrusive, Intermediate]),
    'GRPH' : GeoCode([Felsic, Intrusive]),
    'CNGL' : GeoCode(Conglomerate),
    'AMPH' : GeoCode(Amphibolite),
    'CHRT' : GeoCode(Chert),
    'SYEN' : GeoCode([Intrusive, Mafic]),
    'GBNO' : GeoCode([Intrusive, Mafic]),
    'MGBR' : GeoCode([Intrusive, Mafic]),
    'DACT' : GeoCode([Volcanic, Intermediate, Felsic]),
    'QZAR' : GeoCode(Sandstone),
    'DUNT' : GeoCode([Ultramafic, Intrusive]),
    'TRAN' : GeoCode([Intrusive, Mafic]),
    'MYLN' : GeoCode(Mylonite),
    'FELS' : GeoCode([Volcanic, Felsic]),
    'QUTZ' : GeoCode(Quartz_Vein),
    'MDSN' : GeoCode(Sedimentary),
    'OBTR' : GeoCode([Mafic, Intrusive]),
    'QZPY' : GeoCode([Igneous, Quartz_Vein]),
    'PICR' : GeoCode([Volcanic, Ultramafic]),
    'INTR' : GeoCode(Intrusive),
    'PYRX' : GeoCode([Ultramafic, Intrusive]),
    'PHYL' : GeoCode(Phyllite),
    'MNZT' : GeoCode([Intrusive, Intermediate]),
    'IRON' : GeoCode(Iron_Formation),
    'FALT' : GeoCode(Fault),
    'MDIR' : GeoCode(Metaigneous),
    'VOLM' : GeoCode([Volcanic, Mafic]),
    'MARL' : GeoCode(Sedimentary),
    'MFGN' : GeoCode([Gneiss, Mafic]),
    'UMFU' : GeoCode(Ultramafic),
    'VOLI' : GeoCode([Volcanic, Intermediate]),
    'FEGN' : GeoCode([Felsic, Gneiss]),
    'BXTT' : GeoCode([Fault_Breccia, Metamorphic]),
    'BXVL' : GeoCode([Fault_Breccia, Volcanic]),
    'FFBX' : GeoCode([Fault_Breccia, Volcanic]),
    'BXFT' : GeoCode([Fault_Breccia, Volcanic]),
    'MFBX' : GeoCode([Fault_Breccia, Volcanic, Mafic]),
    'TUBX' : GeoCode([Fault_Breccia, Volcanic, Felsic]),
    'BXSD' : GeoCode([Fault_Breccia, Sedimentary]),
    'BXIN' : GeoCode([Fault_Breccia, Igneous]),
    'FIBX' : GeoCode([Fault_Breccia, Intrusive, Felsic]),
    'INBX' : GeoCode([Fault_Breccia, Igneous]),
    'MIBX' : GeoCode([Fault_Breccia, Intrusive, Mafic]),
}

### NEW PRECAMBRIAN CODES ###

PRECAMBRIAN_CODE_MAP = {
    'PAFI' : GeoCode([Felsic, Intrusive, Archean]),
    'PEFI' : GeoCode([Felsic, Intrusive, Early_Proterozoic]),
    'PMFI' : GeoCode([Felsic, Intrusive, Mesoproterozoic]),
    'PUFI' : GeoCode([Felsic, Intrusive, ]),
    'PAII' : GeoCode([Intermediate, Intrusive, Archean]),
    'PEII' : GeoCode([Intermediate, Intrusive, Early_Proterozoic]),
    'PMII' : GeoCode([Intermediate, Intrusive, Mesoproterozoic]),
    'PUII' : GeoCode([Intermediate, Intrusive, ]),
    'PAIM' : GeoCode([Mafic, Intrusive, Archean]),
    'PEIM' : GeoCode([Mafic, Intrusive, Early_Proterozoic]),
    'PMIM' : GeoCode([Mafic, Intrusive, Mesoproterozoic]),
    'PUIM' : GeoCode([Mafic, Intrusive, ]),
    'PAUI' : GeoCode([Ultramafic, Intrusive, Archean]),
    'PEUI' : GeoCode([Ultramafic, Intrusive, Early_Proterozoic]),
    'PMUI' : GeoCode([Ultramafic, Intrusive, Mesoproterozoic]),
    'PUUI' : GeoCode([Ultramafic, Intrusive, ]),
    'PASA' : GeoCode([Slate, Archean]),
    'PESA' : GeoCode([Slate, Early_Proterozoic]),
    'PMSA' : GeoCode([Slate, Mesoproterozoic]),
    'PUSA' : GeoCode([Slate, ]),
    'PART' : GeoCode([Argillite, Archean]),
    'PERT' : GeoCode([Argillite, Early_Proterozoic]),
    'PMRT' : GeoCode([Argillite, Mesoproterozoic]),
    'PURT' : GeoCode([Argillite, ]),
    'PAPH' : GeoCode([Phyllite, Archean]),
    'PEPH' : GeoCode([Phyllite, Early_Proterozoic]),
    'PMPH' : GeoCode([Phyllite, Mesoproterozoic]),
    'PUPH' : GeoCode([Phyllite, ]),
    'PACH' : GeoCode([Schist, Archean]),
    'PECH' : GeoCode([Schist, Early_Proterozoic]),
    'PMCH' : GeoCode([Schist, Mesoproterozoic]),
    'PUCH' : GeoCode([Schist, ]),
    'PAGN' : GeoCode([Gneiss, Archean]),
    'PEGN' : GeoCode([Gneiss, Early_Proterozoic]),
    'PMGN' : GeoCode([Gneiss, Mesoproterozoic]),
    'PUGN' : GeoCode([Gneiss, ]),
    'PAMA' : GeoCode([Marble, Archean]),
    'PEMA' : GeoCode([Marble, Early_Proterozoic]),
    'PMMA' : GeoCode([Marble, Mesoproterozoic]),
    'PUMA' : GeoCode([Marble, ]),
    'PAQA' : GeoCode([Quartzite, Archean]),
    'PEQA' : GeoCode([Quartzite, Early_Proterozoic]),
    'PMQA' : GeoCode([Quartzite, Mesoproterozoic]),
    'PUQA' : GeoCode([Quartzite, ]),
    'PAPE' : GeoCode([Metapelite, Archean]),
    'PEPE' : GeoCode([Metapelite, Early_Proterozoic]),
    'PMPE' : GeoCode([Metapelite, Mesoproterozoic]),
    'PUPE' : GeoCode([Metapelite, ]),
    'PAPS' : GeoCode([Metapsammite, Archean]),
    'PEPS' : GeoCode([Metapsammite, Early_Proterozoic]),
    'PMPS' : GeoCode([Metapsammite, Mesoproterozoic]),
    'PUPS' : GeoCode([Metapsammite, ]),
    'PAFE' : GeoCode([Felsic_Gneiss, Archean]),
    'PEFE' : GeoCode([Felsic_Gneiss, Early_Proterozoic]),
    'PMFE' : GeoCode([Felsic_Gneiss, Mesoproterozoic]),
    'PUFE' : GeoCode([Felsic_Gneiss, ]),
    'PAIG' : GeoCode([Intermediate_Gneiss, Archean]),
    'PEIG' : GeoCode([Intermediate_Gneiss, Early_Proterozoic]),
    'PMIG' : GeoCode([Intermediate_Gneiss, Mesoproterozoic]),
    'PUIG' : GeoCode([Intermediate_Gneiss, ]),
    'PAMF' : GeoCode([Mafic_Gneiss, Archean]),
    'PEMF' : GeoCode([Mafic_Gneiss, Early_Proterozoic]),
    'PMMF' : GeoCode([Mafic_Gneiss, Mesoproterozoic]),
    'PUMF' : GeoCode([Mafic_Gneiss, ]),
    'PAFS' : GeoCode([Felsic_Schist, Archean]),
    'PEFS' : GeoCode([Felsic_Schist, Early_Proterozoic]),
    'PMFS' : GeoCode([Felsic_Schist, Mesoproterozoic]),
    'PUFS' : GeoCode([Felsic_Schist, ]),
    'PAIS' : GeoCode([Intermediate_Schist, Archean]),
    'PEIS' : GeoCode([Intermediate_Schist, Early_Proterozoic]),
    'PMIS' : GeoCode([Intermediate_Schist, Mesoproterozoic]),
    'PUIS' : GeoCode([Intermediate_Schist, ]),
    'PAMH' : GeoCode([Mafic_Schist, Archean]),
    'PEMH' : GeoCode([Mafic_Schist, Early_Proterozoic]),
    'PMMH' : GeoCode([Mafic_Schist, Mesoproterozoic]),
    'PUMH' : GeoCode([Mafic_Schist, ]),
    'PACC' : GeoCode([Calcium_Silicate_Schist, Archean]),
    'PECC' : GeoCode([Calcium_Silicate_Schist, Early_Proterozoic]),
    'PMCC' : GeoCode([Calcium_Silicate_Schist, Mesoproterozoic]),
    'PUCC' : GeoCode([Calcium_Silicate_Schist, ]),
    'PAAM' : GeoCode([Amphibolite, Archean]),
    'PEAM' : GeoCode([Amphibolite, Early_Proterozoic]),
    'PMAM' : GeoCode([Amphibolite, Mesoproterozoic]),
    'PUAM' : GeoCode([Amphibolite, ]),
    'PAFX' : GeoCode([Fault_Breccia, Archean]),
    'PEFX' : GeoCode([Fault_Breccia, Early_Proterozoic]),
    'PMFX' : GeoCode([Fault_Breccia, Mesoproterozoic]),
    'PUFX' : GeoCode([Fault_Breccia, ]),
    'PAFO' : GeoCode([Fault_Gouge, Archean]),
    'PEFO' : GeoCode([Fault_Gouge, Early_Proterozoic]),
    'PMFO' : GeoCode([Fault_Gouge, Mesoproterozoic]),
    'PUFO' : GeoCode([Fault_Gouge, ]),
    'PAFB' : GeoCode([Brittle_Fault, Archean]),
    'PEFB' : GeoCode([Brittle_Fault, Early_Proterozoic]),
    'PMFB' : GeoCode([Brittle_Fault, Mesoproterozoic]),
    'PUFB' : GeoCode([Brittle_Fault, ]),
    'PAPM' : GeoCode([Protomylonite, Archean]),
    'PEPM' : GeoCode([Protomylonite, Early_Proterozoic]),
    'PMPM' : GeoCode([Protomylonite, Mesoproterozoic]),
    'PUPM' : GeoCode([Protomylonite, ]),
    'PAMY' : GeoCode([Mylonite, Archean]),
    'PEMY' : GeoCode([Mylonite, Early_Proterozoic]),
    'PMMY' : GeoCode([Mylonite, Mesoproterozoic]),
    'PUMY' : GeoCode([Mylonite, ]),
    'PAMM' : GeoCode([Ultramylonite, Archean]),
    'PEMM' : GeoCode([Ultramylonite, Early_Proterozoic]),
    'PMMM' : GeoCode([Ultramylonite, Mesoproterozoic]),
    'PUMM' : GeoCode([Ultramylonite, ]),
    'PAMS' : GeoCode([Metasediment, Archean]),
    'PEMS' : GeoCode([Metasediment, Early_Proterozoic]),
    'PMSS' : GeoCode([Metasediment, Mesoproterozoic]),
    'PUSS' : GeoCode([Metasediment, ]),
    'PAQV' : GeoCode([Quartz_Vein, Archean]),
    'PEQV' : GeoCode([Quartz_Vein, Early_Proterozoic]),
    'PMQV' : GeoCode([Quartz_Vein, Mesoproterozoic]),
    'PUQV' : GeoCode([Quartz_Vein, ]),
    'PAPD' : GeoCode([Pegmatite, Archean]),
    'PEPD' : GeoCode([Pegmatite, Early_Proterozoic]),
    'PMPD' : GeoCode([Pegmatite, Mesoproterozoic]),
    'PUPD' : GeoCode([Pegmatite, ]),
    'PADD' : GeoCode([Diabase, Archean]),
    'PEDD' : GeoCode([Diabase, Early_Proterozoic]),
    'PMDD' : GeoCode([Diabase, Mesoproterozoic]),
    'PUDD' : GeoCode([Diabase, ]),
    'PAAD' : GeoCode([Aplite, Archean]),
    'PEAD' : GeoCode([Aplite, Early_Proterozoic]),
    'PMAD' : GeoCode([Aplite, Mesoproterozoic]),
    'PUAD' : GeoCode([Aplite, ]),
    'PAUV' : GeoCode([Ultramafic, Volcanic, Archean]),
    'PEUV' : GeoCode([Ultramafic, Volcanic, Early_Proterozoic]),
    'PMUV' : GeoCode([Ultramafic, Volcanic, Mesoproterozoic]),
    'PUUV' : GeoCode([Ultramafic, Volcanic, ]),
    'PAVM' : GeoCode([Mafic, Volcanic, Archean]),
    'PEVM' : GeoCode([Mafic, Volcanic, Early_Proterozoic]),
    'PMVM' : GeoCode([Mafic, Volcanic, Mesoproterozoic]),
    'PUVM' : GeoCode([Mafic, Volcanic, ]),
    'PAIV' : GeoCode([Intermediate, Volcanic, Archean]),
    'PEIV' : GeoCode([Intermediate, Volcanic, Early_Proterozoic]),
    'PMIV' : GeoCode([Intermediate, Volcanic, Mesoproterozoic]),
    'PUIV' : GeoCode([Intermediate, Volcanic, ]),
    'PAVF' : GeoCode([Felsic, Volcanic, Archean]),
    'PEVF' : GeoCode([Felsic, Volcanic, Early_Proterozoic]),
    'PMVF' : GeoCode([Felsic, Volcanic, Mesoproterozoic]),
    'PUVF' : GeoCode([Felsic, Volcanic, ]),
    'PAJM' : GeoCode([Mafic, Intermediate, Volcanic, Archean]),
    'PEJM' : GeoCode([Mafic, Intermediate, Volcanic, Early_Proterozoic]),
    'PMJM' : GeoCode([Mafic, Intermediate, Volcanic, Mesoproterozoic]),
    'PUJM' : GeoCode([Mafic, Intermediate, Volcanic, ]),
    'PAJF' : GeoCode([Felsic, Intermediate, Volcanic, Archean]),
    'PEJF' : GeoCode([Felsic, Intermediate, Volcanic, Early_Proterozoic]),
    'PMJF' : GeoCode([Felsic, Intermediate, Volcanic, Mesoproterozoic]),
    'PUJF' : GeoCode([Felsic, Intermediate, Volcanic, ]),
    'PACG' : GeoCode([Conglomerate, Archean]),
    'PECG' : GeoCode([Conglomerate, Early_Proterozoic]),
    'PMCG' : GeoCode([Conglomerate, Mesoproterozoic]),
    'PUCG' : GeoCode([Conglomerate, ]),
    'PACR' : GeoCode([Carbonate, Archean]),
    'PECR' : GeoCode([Carbonate, Early_Proterozoic]),
    'PMCR' : GeoCode([Carbonate, Mesoproterozoic]),
    'PUCR' : GeoCode([Carbonate, ]),
    'PADS' : GeoCode([Sandstone, Archean]),
    'PEDS' : GeoCode([Sandstone, Early_Proterozoic]),
    'PMDS' : GeoCode([Sandstone, Mesoproterozoic]),
    'PUDS' : GeoCode([Sandstone, ]),
    'PASI' : GeoCode([Siltstone, Archean]),
    'PESI' : GeoCode([Siltstone, Early_Proterozoic]),
    'PMSI' : GeoCode([Siltstone, Mesoproterozoic]),
    'PUSI' : GeoCode([Siltstone, ]),
    'PASE' : GeoCode([Shale, Archean]),
    'PESE' : GeoCode([Shale, Early_Proterozoic]),
    'PMSE' : GeoCode([Shale, Mesoproterozoic]),
    'PUSE' : GeoCode([Shale, ]),
    'PACT' : GeoCode([Chert, Archean]),
    'PECT' : GeoCode([Chert, Early_Proterozoic]),
    'PMCT' : GeoCode([Chert, Mesoproterozoic]),
    'PUCT' : GeoCode([Chert, ]),
    'PAIF' : GeoCode([Iron_Formation, Archean]),
    'PEIF' : GeoCode([Iron_Formation, Early_Proterozoic]),
    'PMIF' : GeoCode([Iron_Formation, Mesoproterozoic]),
    'PUIF' : GeoCode([Iron_Formation, ]),

    ### OLD CODES ###
    'PAAI' : GeoCode(Intrusive),
    'PAAR' : GeoCode([Archean, Gneiss]), # Description says Gneiss, lith in db says a bunch of igneous/meta
    'PAAU' : GeoCode([Archean, Felsic, Intermediate, Intrusive]),
    'PABD' : GeoCode([Archean, Felsic, Intermediate, Intrusive]),
    'PABG' : GeoCode([Archean, Gneiss, Felsic, Intrusive]), # This is both Gneiss and Granite?
    'PABK' : GeoCode(Archean),
    'PABL' : GeoCode([Archean, Felsic, Intrusive]),
    'PABR' : GeoCode([Archean, Gneiss]),
    'PADL' : GeoCode([Archean, Mafic, Intrusive]),
    'PAEF' : GeoCode([Archean, Felsic, Volcanic]),
    'PAES' : GeoCode([Archean, Iron_Formation]),
    'PAEY' : GeoCode(Archean),
    'PAFL' : GeoCode(Archean),
    'PAFR' : GeoCode([Archean, Felsic, Intrusive]),
    'PAFV' : GeoCode([Archean, Felsic, Intermediate, Volcanic]),
    'PAGF' : GeoCode([Archean, Gneiss]),
    'PAGM' : GeoCode([Archean, Metasediment]),
    'PAGR' : GeoCode([Archean, Felsic, Intrusive]),
    'PAJL' : GeoCode(Archean),
    'PAKG' : GeoCode(Archean),
    'PALC' : GeoCode([Archean, Conglomerate]),
    'PALG' : GeoCode(Archean),
    'PALK' : GeoCode(Archean),
    'PALL' : GeoCode([Archean, Felsic, Intrusive]),
    'PALP' : GeoCode(Archean),
    'PALS' : GeoCode(Archean),
    'PALT' : GeoCode(Archean),
    'PALV' : GeoCode(Archean),
    'PAMB' : GeoCode([Archean, Mafic, Volcanic]),
    'PAMC' : GeoCode([Archean, Gneiss]),
    'PAMD' : GeoCode(Archean),
    'PAMG' : GeoCode([Archean, Gneiss]),
    'PAMI' : GeoCode([Archean, Mafic, Intermediate, Volcanic]),
    'PAML' : GeoCode(Archean),
    'PAMP' : GeoCode(Archean),
    'PAMR' : GeoCode([Archean, Gneiss]),
    'PAMT' : GeoCode(Archean),
    'PAMU' : GeoCode([Archean, Mafic, Intrusive, Ultramafic, Intrusive]),
    'PAMV' : GeoCode([Archean, Gneiss]),
    'PANB' : GeoCode([Archean, Mafic, Volcanic]),
    'PANL' : GeoCode(Archean),
    'PANS' : GeoCode(Archean),
    'PANT' : GeoCode(Archean),
    'PANV' : GeoCode(Archean),
    'PANY' : GeoCode([Archean, Slate]),
    'PAOD' : GeoCode([Archean, Felsic, Intrusive]),
    'PAOG' : GeoCode([Archean, Gneiss]),
    'PAOK' : GeoCode([Archean]),
    'PAOR' : GeoCode([Archean, Felsic, Intrusive]),
    'PAPG' : GeoCode([Archean, Gneiss]),
    'PAPL' : GeoCode(Archean),
    'PAPP' : GeoCode(Archean),
    'PAQF' : GeoCode([Archean, Felsic, Intrusive]),
    'PASB' : GeoCode([Archean, Felsic, Intrusive]),
    'PASF' : GeoCode([Archean, Gneiss]),
    'PASG' : GeoCode(Archean),
    'PASH' : GeoCode([Archean, Felsic, Intrusive]),
    'PASL' : GeoCode(Archean),
    'PASM' : GeoCode([Archean, Schist]),
    'PASN' : GeoCode([Archean, Felsic, Intrusive]),
    'PASR' : GeoCode([Archean, Gneiss]),
    'PAST' : GeoCode([Archean, Gneiss]),
    'PASV' : GeoCode(Archean),
    'PATL' : GeoCode(Archean),
    'PAUD' : GeoCode(Archean),
    'PAVC' : GeoCode([Archean, Felsic, Intrusive]),
    'PAVP' : GeoCode([Archean, Felsic, Intrusive]),
    'PAWB' : GeoCode([Archean, Intermediate, Intrusive]),
    'PAWL' : GeoCode(Archean),
    'PCMU' : GeoCode([Mafic, Intrusive, Ultramafic, Intrusive]),
    'PEAG' : GeoCode(Early_Proterozoic),
    'PEAL' : GeoCode([Early_Proterozoic, Intermediate, Intrusive]),
    'PEBC' : GeoCode([Early_Proterozoic, Intermediate, Intrusive]),
    'PEBI' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PECM' : GeoCode(Early_Proterozoic),
    'PECS' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PEDN' : GeoCode(Early_Proterozoic),
    'PEDQ' : GeoCode([Early_Proterozoic, Quartzite]),
    'PEFG' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PEFH' : GeoCode([Early_Proterozoic, Intermediate, Intrusive]),
    'PEFM' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PEGD' : GeoCode([Early_Proterozoic, Intermediate, Intrusive]),
    'PEGI' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PEGP' : GeoCode([Early_Proterozoic, Intermediate, Intrusive]),
    'PEGT' : GeoCode(Early_Proterozoic),
    'PEGU' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PEHL' : GeoCode([Early_Proterozoic, Intermediate, Intrusive]),
    'PEIL' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PELC' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PELF' : GeoCode([Early_Proterozoic, Metasediment]),
    'PELR' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PELS' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PEMG' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PEML' : GeoCode(Early_Proterozoic),
    'PEMN' : GeoCode(Early_Proterozoic),
    'PEMU' : GeoCode([Early_Proterozoic, Mafic, Intrusive]),
    'PEPG' : GeoCode([Early_Proterozoic, Mafic, Intermediate, Volcanic]),
    'PEPK' : GeoCode([Early_Proterozoic, Quartzite]),
    'PEPP' : GeoCode([Early_Proterozoic, Mafic, Intrusive]),
    'PEPZ' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PERB' : GeoCode(Early_Proterozoic),
    'PERD' : GeoCode(Early_Proterozoic),
    'PERE' : GeoCode(Early_Proterozoic),
    'PERF' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PERG' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PERK' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PERL' : GeoCode(Early_Proterozoic),
    'PERU' : GeoCode(Early_Proterozoic),
    'PERV' : GeoCode(Early_Proterozoic),
    'PESC' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PESG' : GeoCode([Early_Proterozoic, Schist]),
    'PEST' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PESW' : GeoCode([Early_Proterozoic, Mafic, Intrusive]),
    'PESX' : GeoCode([Early_Proterozoic, Quartzite]),
    'PETL' : GeoCode(Early_Proterozoic),
    'PETR' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PEUC' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PEUD' : GeoCode(Early_Proterozoic),
    'PEUM' : GeoCode([Early_Proterozoic, Ultramafic, Intrusive]),
    'PEUS' : GeoCode([Early_Proterozoic, Iron_Formation]),
    'PEVT' : GeoCode(Early_Proterozoic),
    'PEWR' : GeoCode([Early_Proterozoic, Felsic, Intrusive]),
    'PEWS' : GeoCode([Early_Proterozoic, Metasediment]),
    'PEWT' : GeoCode([Early_Proterozoic, Intermediate, Intrusive]),
    'PEWV' : GeoCode(Early_Proterozoic),
    'PEYV' : GeoCode([Early_Proterozoic, Mafic, Volcanic]),
    'PMAG' : GeoCode([Mesoproterozoic, Diabase]),
    'PMBB' : GeoCode(Mesoproterozoic),
    'PMBC' : GeoCode([Mesoproterozoic, Mafic, Intrusive]),
    'PMBE' : GeoCode(Mesoproterozoic),
    'PMBF' : GeoCode([Mesoproterozoic, Felsic, Intrusive]),
    'PMBI' : GeoCode([Mesoproterozoic, Intermediate, Intrusive]),
    'PMBL' : GeoCode([Mesoproterozoic, Diabase]),
    'PMBM' : GeoCode([Mesoproterozoic, Mafic, Intrusive]),
    'PMBO' : GeoCode(Mesoproterozoic),
    'PMBP' : GeoCode([Mesoproterozoic, Intermediate, Intrusive]),
    'PMBR' : GeoCode([Mesoproterozoic, Diabase]),
    'PMBS' : GeoCode(Mesoproterozoic),
    'PMBT' : GeoCode(Mesoproterozoic),
    'PMCF' : GeoCode(Mesoproterozoic),
    'PMCV' : GeoCode([Mesoproterozoic, Mafic, Intermediate, Volcanic]),
    'PMCX' : GeoCode(Mesoproterozoic),
    'PMDA' : GeoCode(Mesoproterozoic),
    'PMDC' : GeoCode(Mesoproterozoic),
    'PMDE' : GeoCode([Mesoproterozoic, Mafic, Intrusive]),
    'PMDF' : GeoCode([Mesoproterozoic, Felsic, Intrusive]),
    'PMDL' : GeoCode(Mesoproterozoic),
    'PMEP' : GeoCode([Mesoproterozoic, Mafic, Volcanic]),
    'PMES' : GeoCode(Mesoproterozoic),
    'PMGI' : GeoCode([Mesoproterozoic, Mafic, Intrusive]),
    'PMGL' : GeoCode(Mesoproterozoic),
    'PMHR' : GeoCode(Mesoproterozoic),
    'PMIU' : GeoCode(Mesoproterozoic),
    'PMKL' : GeoCode(Mesoproterozoic),
    'PMLB' : GeoCode([Mesoproterozoic, Diabase]),
    'PMLD' : GeoCode(Mesoproterozoic),
    'PMLG' : GeoCode(Mesoproterozoic),
    'PMLS' : GeoCode(Mesoproterozoic),
    'PMMI' : GeoCode(Mesoproterozoic),
    'PMMU' : GeoCode([Mesoproterozoic, Mafic, Intrusive]),
    'PMMV' : GeoCode(Mesoproterozoic),
    'PMNB' : GeoCode(Mesoproterozoic),
    'PMND' : GeoCode([Mesoproterozoic, Sandstone]),
    'PMFL' : GeoCode(Mesoproterozoic),
    'PMHN' : GeoCode([Mesoproterozoic, Sandstone]),
    'PMHF' : GeoCode(Mesoproterozoic),
    'PMNF' : GeoCode([Mesoproterozoic, Felsic, Volcanic]),
    'PMNI' : GeoCode([Mesoproterozoic, Intermediate, Volcanic]),
    'PMNL' : GeoCode(Mesoproterozoic),
    'PMNM' : GeoCode([Mesoproterozoic, Mafic, Volcanic]),
    'PMNP' : GeoCode([Mesoproterozoic, Sandstone]),
    'PMNS' : GeoCode(Mesoproterozoic),
    'PMOI' : GeoCode([Mesoproterozoic, Ultramafic, Intrusive]),
    'PMPA' : GeoCode([Mesoproterozoic, Mafic, Intrusive]),
    'PMPK' : GeoCode(Mesoproterozoic),
    'PMPR' : GeoCode(Mesoproterozoic),
    'PMRC' : GeoCode(Mesoproterozoic),
    'PMSC' : GeoCode(Mesoproterozoic),
    'PMSK' : GeoCode(Mesoproterozoic),
    'PMSL' : GeoCode(Mesoproterozoic),
    'PMSU' : GeoCode(Mesoproterozoic),
    'PMSW' : GeoCode(Mesoproterozoic),
    'PMTH' : GeoCode(Mesoproterozoic),
    'PMTI' : GeoCode(Mesoproterozoic),
    'PMUD' : GeoCode(Mesoproterozoic),
    'PMUS' : GeoCode(Mesoproterozoic),
    'PMVU' : GeoCode(Mesoproterozoic),
    'PMWD' : GeoCode(Mesoproterozoic),
    'PMWL' : GeoCode([Mesoproterozoic, Felsic, Intrusive]),
    'PMWM' : GeoCode(Mesoproterozoic)
}