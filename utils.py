import pandas as pd

from comparison_test import bedrock


class Bedrock:
    C_UNDIFF = 'CAMB'
    P_UNDIFF = 'PUDF'
    D_UNDIFF = 'DEVO'
    K_UNDIFF = 'KRET'
    O_UNDIFF = 'ORDO'

    CARLILE = 'KCRL'
    COLERAINE = 'KCLR'
    DAKOTA = 'KDWB'
    K_REGOLITH = 'KREG'
    SPLIT_ROCK_CREEK = 'KSRC'
    WINDROW = 'KWND'

    UPPER_CEDAR = 'DCVU'
    LITTLE_CEDAR = 'DLCD'
    CEDAR_VALLEY = 'DCVA'
    LOWER_CEDAR = 'DCVL'
    PINICON_RIDGE = 'DWPR'
    WAPSIPINICON = 'DWAP'
    SPILLVILLE = 'DSPL'

    MAQUOKETA = 'OMAQ'
    GALENA = 'OGGP'
    DUBUQUE = 'ODUB'
    ST_PETER = 'OSTP'
    DECORAH = 'ODCR'
    PLATTEVILLE = 'OPVL'
    PROSSER = 'OGPR'
    CUMMINGSVILLE = 'OGCM'
    STEWARTVILLE = 'OGSV'
    GLENWOOD = 'OGWD'
    PRARIE_DU_CHIEN = 'OPDC'
    SHAKOPEE = 'OPSH'
    RICHMOND = 'Richmond'
    WILLOW_RIVER = 'Willow River'
    ONEOTA = 'OPOD'
    WINNIPEG = 'OWIN'

    JORDAN = 'CJDN'
    ST_LAWRENCE = 'CSTL'
    TUNNEL_CITY = 'CTCG'
    MAZOMANIE = 'CTMZ'
    LONE_ROCK = 'CTLR'
    WONEWOC = 'CWOC'
    EAU_CLAIRE = 'CECR'
    MT_SIMON = 'CMTS'

    FOND_DU_LAC = 'PMFL'
    HINCKLEY = 'PMHN'
    SOLOR_CHURCH = 'PMSC'

class Field:
    UTME = 'utme'
    UTMN = 'utmn'
    ELEVATION = 'elevation'
    STRAT = 'strat'
    RELATEID = 'relateid'
    AGE = 'age'
    AGE_CATEGORY = 'age_cat'
    DATA_SOURCE = 'data_src'
    FIRST_BEDROCK_CATEGORY = 'first_bdrk_cat'
    DEPTH_TO_BEDROCK = 'depth_to_bdrk'
    TOP_DEPTH_TO_BEDROCK = 'top_depth_to_bdrk'
    BOT_DEPTH_TO_BEDROCK = 'bot_depth_to_bdrk'
    WEIGHT = 'weight'
    DEPTH_TOP = 'depth_top'
    DEPTH_BOT = 'depth_bot'
    COLOR = 'color'
    DRILLER_DESCRIPTION = 'drllr_desc'
    PREVIOUS_AGE_CATEGORY = 'prev_age_cat'
    CORE = 'core'
    CUTTINGS = 'cuttings'
    INTERPRETATION_METHOD = 'strat_mc'

MIN_LABEL_COUNT = 50

AGE_RAND_STATE = 1

LAYER_FEATURE_COLS = [
    Field.UTME, Field.UTMN, Field.ELEVATION, Field.STRAT, Field.RELATEID, Field.AGE_CATEGORY, Field.DATA_SOURCE,
    Field.WEIGHT, Field.DEPTH_TOP, Field.DEPTH_BOT, Field.COLOR, Field.DRILLER_DESCRIPTION, Field.CORE,
    Field.CUTTINGS, Field.INTERPRETATION_METHOD
]

SCALED_COLUMNS = [Field.DEPTH_TOP, Field.DEPTH_BOT, Field.ELEVATION, Field.UTME, Field.UTMN, Field.DEPTH_TO_BEDROCK]
TRUSTED_SOURCES = []

SHAPEFILE_PATHS = ['ju_pg', 'ka_pg', 'kc_pg', 'ku_pg', 'pz_pg', 'S21_pcpg']

#TODO: I actually havent added first_bedrock_age yet...
GENERAL_DROP_COLS = [Field.RELATEID, Field.DRILLER_DESCRIPTION, Field.COLOR, Field.WEIGHT, Field.STRAT, Field.DATA_SOURCE,
                     Field.INTERPRETATION_METHOD]
AGE_DROP_COLS = [Field.AGE_CATEGORY, Field.AGE]
QUAT_DROP_COLS = [Field.DEPTH_TOP, Field.DEPTH_BOT, Field.UTME, Field.UTMN, Field.ELEVATION, Field.DEPTH_TO_BEDROCK,
                  Field.TOP_DEPTH_TO_BEDROCK, Field.BOT_DEPTH_TO_BEDROCK]
#TODO: Need to hyperparameter train this
BEDROCK_DROP_COLS = [Field.BOT_DEPTH_TO_BEDROCK, Field.AGE_CATEGORY]

QUAT_COLOR_MAP = {
    'BROWN': 'B', 'DK. BRN': 'B', 'LT. BRN': 'B', 'TAN': 'B',
    'GRAY': 'G',  'DK. GRY': 'G', 'LT. GRY': 'G', 'BLU/GRY' : 'G',
    'BLUE': 'G', 'DK. BLU': 'G', 'LT. BLU': 'G',
    'BLACK': 'K',
    'RED': 'R',
    'GREEN': 'L',
    'ORANGE': 'O',
    'Other/Varied': 'U',
    'WHITE': 'W',
    'YELLOW': 'Y'
}

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

FIRST_BEDROCK_CATEGORIES = {
    'Ku' : 1,
    'Kc' : 2,
    'Ka' : 3,
    'Ju' : 4,
    'Dmu' : 5,
    'Dm' : 6,
    'Ou' : 7,
    'Omu' : 8,
    'Ol' : 9,
    'Cu' : 10,
    'Cmu' : 11,
    'M' : 12,
    'P' : 13,
    'A' : 14,
}
INV_FIRST_BEDROCK_CATEGORIES = {v: k for k, v in FIRST_BEDROCK_CATEGORIES.items()}

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
BEDROCK_CATEGORIES = {}
INV_BEDROCK_CATEGORIES = {}

BEDROCK_AGES = ('C', 'D', 'K', 'O')
SORTED_PRECAMBRIAN =  ('PMHN', 'PMFL', 'PMSC', 'PMHF')

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

BEDROCK_EXCEPTIONS = [
    'CAMB', 'PUDF', 'DEVO', 'KRET', 'ORDO', 'KREG', 'UREG', 'DWAP'
]

#TODO: This is where underrepresented classes are mapped to their parents!
BEDROCK_UNDERREP_MAP = {
    'KCBH' : Bedrock.CARLILE,
    'KCCD' : Bedrock.CARLILE,
    'KCFP' : Bedrock.CARLILE,

    'KDNB' : Bedrock.K_UNDIFF,
    'KDWB' : Bedrock.K_UNDIFF,
    'KDKT' : Bedrock.K_UNDIFF,

    'KGRN' : Bedrock.K_UNDIFF,
    'KGRS' : Bedrock.K_UNDIFF,

    'KWIH' : Bedrock.WINDROW,
    'KWOS' : Bedrock.WINDROW,

    'DLGH' : Bedrock.UPPER_CEDAR,

    'DCRL' : Bedrock.UPPER_CEDAR,
    'DCUM' : Bedrock.UPPER_CEDAR,
    'DCGZ' : Bedrock.UPPER_CEDAR,
    'DCIC' : Bedrock.UPPER_CEDAR,

    'DLBA' : Bedrock.LITTLE_CEDAR,
    'DLCB' : Bedrock.LITTLE_CEDAR,
    'DLCH' : Bedrock.LITTLE_CEDAR,
    'DLHE' : Bedrock.LITTLE_CEDAR,

    'ODCA' : Bedrock.DECORAH,

    'OSPE' : Bedrock.ST_PETER,
    'OSTN' : Bedrock.ST_PETER,

    'OPHF' : Bedrock.PLATTEVILLE,
    'OPMA' : Bedrock.PLATTEVILLE,
    'OPMI' : Bedrock.PLATTEVILLE,
    'OPPE' : Bedrock.PLATTEVILLE,

    'OGDP' : 'ODPL', #TODO: Need to ask but I am pretty sure these represent the exact same thing

    'OOCV' : Bedrock.ONEOTA,
    'OOHC' : Bedrock.ONEOTA,

    'ORRV' : Bedrock.O_UNDIFF,

    'OSTM' : Bedrock.O_UNDIFF,

    'OWBI' : Bedrock.WINNIPEG,
    'OWIB' : Bedrock.WINNIPEG,

    'CLBK' : Bedrock.LONE_ROCK,
    'CLRE' : Bedrock.LONE_ROCK,
    'CLTM' : Bedrock.LONE_ROCK,

    'CMRC' : Bedrock.MT_SIMON,
}
# Giant list of every possible (well layered) code and which age(s) they belong to
BEDROCK_AGE_MAP = {
    'CAMB': frozenset([Bedrock.C_UNDIFF]),
    'PUDF': frozenset([Bedrock.P_UNDIFF]),
    'DEVO': frozenset([Bedrock.D_UNDIFF]),
    'KRET': frozenset([Bedrock.K_UNDIFF]),
    'ORDO': frozenset([Bedrock.O_UNDIFF]),

    'KCRL' : frozenset([Bedrock.K_UNDIFF]),
    'KCLR' : frozenset([Bedrock.K_UNDIFF]),
    'KREG' : frozenset([Bedrock.K_UNDIFF]),
    'KSRC' : frozenset([Bedrock.K_UNDIFF]),
    'KWND' : frozenset([Bedrock.K_UNDIFF]),

    'DCVU' : frozenset([Bedrock.D_UNDIFF]),
    'DCLC' : frozenset([Bedrock.D_UNDIFF]),
    'DCVA' : frozenset([Bedrock.D_UNDIFF]),
    'DLCD' : frozenset([Bedrock.D_UNDIFF]),
    'DCLP' : frozenset([Bedrock.D_UNDIFF]),
    'DCLS' : frozenset([Bedrock.D_UNDIFF]),
    'DCOG' : frozenset([Bedrock.D_UNDIFF]),
    'DCOM' : frozenset([Bedrock.D_UNDIFF]),
    'DCVL' : frozenset([Bedrock.D_UNDIFF]),
    'DWAP' : frozenset([Bedrock.D_UNDIFF]),
    'DWPR' : frozenset([Bedrock.D_UNDIFF]),
    'DPOG' : frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DPOM' : frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DSOG' : frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DSOM' : frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),

    'DSPL' : frozenset([Bedrock.D_UNDIFF]),
    'OMAQ' : frozenset([Bedrock.O_UNDIFF]),
    'OMQD' : frozenset([Bedrock.O_UNDIFF]),
    'OMQG' : frozenset([Bedrock.O_UNDIFF]),
    'OGAP' : frozenset([Bedrock.O_UNDIFF]),
    'OGGP' : frozenset([Bedrock.O_UNDIFF]),
    'OGPD' : frozenset([Bedrock.O_UNDIFF]),
    'ODGL' : frozenset([Bedrock.O_UNDIFF]),
    'ODUB' : frozenset([Bedrock.O_UNDIFF]),
    'OGSC' : frozenset([Bedrock.O_UNDIFF]),
    'OGSD' : frozenset([Bedrock.O_UNDIFF]),
    'OGVP' : frozenset([Bedrock.O_UNDIFF]),
    'OGSV' : frozenset([Bedrock.O_UNDIFF]),
    'OGPC' : frozenset([Bedrock.O_UNDIFF]),
    'OGPR' : frozenset([Bedrock.O_UNDIFF]),
    'OGCM' : frozenset([Bedrock.O_UNDIFF]),
    'OGCD' : frozenset([Bedrock.O_UNDIFF]),
    'ODCR' : frozenset([Bedrock.O_UNDIFF]),
    'OGDP' : frozenset([Bedrock.O_UNDIFF]),
    'ODPG' : frozenset([Bedrock.O_UNDIFF]),
    'ODSP' : frozenset([Bedrock.O_UNDIFF]),
    'OPGW' : frozenset([Bedrock.O_UNDIFF]),
    'OPSP' : frozenset([Bedrock.O_UNDIFF]),
    'OPVJ' : frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPVL' : frozenset([Bedrock.O_UNDIFF]),
    'OGSP' : frozenset([Bedrock.O_UNDIFF]),
    'OGWD' : frozenset([Bedrock.O_UNDIFF]),
    'OSCJ' : frozenset([Bedrock.O_UNDIFF]),
    'OSCS' : frozenset([Bedrock.O_UNDIFF]),
    'OSPC' : frozenset([Bedrock.O_UNDIFF]),
    'OSTP' : frozenset([Bedrock.O_UNDIFF]),
    'OPCJ' : frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPCM' : frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPCS' : frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPCT' : frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPDC' : frozenset([Bedrock.O_UNDIFF]),
    'OPNR' : frozenset([Bedrock.O_UNDIFF]),
    'OPWR' : frozenset([Bedrock.O_UNDIFF]),
    'OPSH' : frozenset([Bedrock.O_UNDIFF]),
    'OPOD' : frozenset([Bedrock.O_UNDIFF]),
    'OWIN' : frozenset([Bedrock.O_UNDIFF]),

    'CJDN' : frozenset([Bedrock.C_UNDIFF]),
    'CJMS' : frozenset([Bedrock.C_UNDIFF]),
    'CJEC' : frozenset([Bedrock.C_UNDIFF]),
    'CJDW' : frozenset([Bedrock.C_UNDIFF]),
    'CJSL' : frozenset([Bedrock.C_UNDIFF]),
    'CJTC' : frozenset([Bedrock.C_UNDIFF]),
    'CSLT' : frozenset([Bedrock.C_UNDIFF]),
    'CSLW' : frozenset([Bedrock.C_UNDIFF]),
    'CSTL' : frozenset([Bedrock.C_UNDIFF]),
    'CTCG' : frozenset([Bedrock.C_UNDIFF]),
    'CTCM' : frozenset([Bedrock.C_UNDIFF]),
    'CTCW' : frozenset([Bedrock.C_UNDIFF]),
    'CTCE' : frozenset([Bedrock.C_UNDIFF]),
    'CTMZ' : frozenset([Bedrock.C_UNDIFF]),
    'CWMS' : frozenset([Bedrock.C_UNDIFF]),
    'CWOC' : frozenset([Bedrock.C_UNDIFF]),
    'CWEC' : frozenset([Bedrock.C_UNDIFF]),
    'CECR' : frozenset([Bedrock.C_UNDIFF]),
    'CEMS' : frozenset([Bedrock.C_UNDIFF]),
    'CMFL' : frozenset([Bedrock.C_UNDIFF, Bedrock.P_UNDIFF]),
    'CMSH' : frozenset([Bedrock.C_UNDIFF, Bedrock.P_UNDIFF]),
    'CMTS' : frozenset([Bedrock.C_UNDIFF]),

    'PMHN' : frozenset([Bedrock.P_UNDIFF]),
    'PMHF' : frozenset([Bedrock.P_UNDIFF]),
    'PMFL' : frozenset([Bedrock.P_UNDIFF]),
    'PMSC' : frozenset([Bedrock.P_UNDIFF]),
}

BEDROCK_GROUP_MAP = {
    'CAMB': frozenset([Bedrock.C_UNDIFF]),
    'PUDF': frozenset([Bedrock.P_UNDIFF]),
    'DEVO': frozenset([Bedrock.D_UNDIFF]),
    'KRET': frozenset([Bedrock.K_UNDIFF]),
    'ORDO': frozenset([Bedrock.O_UNDIFF]),

    'KCRL': frozenset([Bedrock.K_UNDIFF]),
    'KCLR': frozenset([Bedrock.K_UNDIFF]),
    'KREG': frozenset([Bedrock.K_UNDIFF]),
    'KSRC': frozenset([Bedrock.K_UNDIFF]),
    'KWND': frozenset([Bedrock.K_UNDIFF]),

    'DCVU': frozenset([Bedrock.D_UNDIFF]),
    'DCLC': frozenset([Bedrock.D_UNDIFF]),
    'DCVA': frozenset([Bedrock.D_UNDIFF]),
    'DLCD': frozenset([Bedrock.D_UNDIFF]),
    'DCLP': frozenset([Bedrock.D_UNDIFF]),
    'DCLS': frozenset([Bedrock.D_UNDIFF]),
    'DCOG': frozenset([Bedrock.D_UNDIFF]),
    'DCOM': frozenset([Bedrock.D_UNDIFF]),
    'DCVL': frozenset([Bedrock.D_UNDIFF]),
    'DWAP': frozenset([Bedrock.D_UNDIFF]),
    'DWPR': frozenset([Bedrock.D_UNDIFF]),
    'DPOG': frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DPOM': frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DSOG': frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DSOM': frozenset([Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),

    'DSPL': frozenset([Bedrock.D_UNDIFF]),
    'OMAQ': frozenset([Bedrock.O_UNDIFF]),
    'OMQD': frozenset([Bedrock.O_UNDIFF]),
    'OMQG': frozenset([Bedrock.O_UNDIFF]),
    'OGAP': frozenset([Bedrock.O_UNDIFF]),
    'OGGP': frozenset([Bedrock.O_UNDIFF]),
    'OGPD': frozenset([Bedrock.O_UNDIFF]),
    'ODGL': frozenset([Bedrock.O_UNDIFF]),
    'ODUB': frozenset([Bedrock.O_UNDIFF]),
    'OGSC': frozenset([Bedrock.O_UNDIFF]),
    'OGSD': frozenset([Bedrock.O_UNDIFF]),
    'OGVP': frozenset([Bedrock.O_UNDIFF]),
    'OGSV': frozenset([Bedrock.O_UNDIFF]),
    'OGPC': frozenset([Bedrock.O_UNDIFF]),
    'OGPR': frozenset([Bedrock.O_UNDIFF]),
    'OGCM': frozenset([Bedrock.O_UNDIFF]),
    'OGCD': frozenset([Bedrock.O_UNDIFF]),
    'ODCR': frozenset([Bedrock.O_UNDIFF]),
    'OGDP': frozenset([Bedrock.O_UNDIFF]),
    'ODPG': frozenset([Bedrock.O_UNDIFF]),
    'ODSP': frozenset([Bedrock.O_UNDIFF]),
    'OPGW': frozenset([Bedrock.O_UNDIFF]),
    'OPSP': frozenset([Bedrock.O_UNDIFF]),
    'OPVJ': frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPVL': frozenset([Bedrock.O_UNDIFF]),
    'OGSP': frozenset([Bedrock.O_UNDIFF]),
    'OGWD': frozenset([Bedrock.O_UNDIFF]),
    'OSCJ': frozenset([Bedrock.O_UNDIFF]),
    'OSCS': frozenset([Bedrock.O_UNDIFF]),
    'OSPC': frozenset([Bedrock.O_UNDIFF]),
    'OSTP': frozenset([Bedrock.O_UNDIFF]),
    'OPCJ': frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPCM': frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPCS': frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPCT': frozenset([Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPDC': frozenset([Bedrock.O_UNDIFF]),
    'OPNR': frozenset([Bedrock.O_UNDIFF]),
    'OPWR': frozenset([Bedrock.O_UNDIFF]),
    'OPSH': frozenset([Bedrock.O_UNDIFF]),
    'OPOD': frozenset([Bedrock.O_UNDIFF]),
    'OWIN': frozenset([Bedrock.O_UNDIFF]),

    'CJDN': frozenset([Bedrock.C_UNDIFF]),
    'CJMS': frozenset([Bedrock.C_UNDIFF]),
    'CJEC': frozenset([Bedrock.C_UNDIFF]),
    'CJDW': frozenset([Bedrock.C_UNDIFF]),
    'CJSL': frozenset([Bedrock.C_UNDIFF]),
    'CJTC': frozenset([Bedrock.C_UNDIFF]),
    'CSLT': frozenset([Bedrock.C_UNDIFF]),
    'CSLW': frozenset([Bedrock.C_UNDIFF]),
    'CSTL': frozenset([Bedrock.C_UNDIFF]),
    'CTCG': frozenset([Bedrock.C_UNDIFF]),
    'CTCM': frozenset([Bedrock.C_UNDIFF]),
    'CTCW': frozenset([Bedrock.C_UNDIFF]),
    'CTCE': frozenset([Bedrock.C_UNDIFF]),
    'CTMZ': frozenset([Bedrock.C_UNDIFF]),
    'CWMS': frozenset([Bedrock.C_UNDIFF]),
    'CWOC': frozenset([Bedrock.C_UNDIFF]),
    'CWEC': frozenset([Bedrock.C_UNDIFF]),
    'CECR': frozenset([Bedrock.C_UNDIFF]),
    'CEMS': frozenset([Bedrock.C_UNDIFF]),
    'CMFL': frozenset([Bedrock.C_UNDIFF, Bedrock.P_UNDIFF]),
    'CMSH': frozenset([Bedrock.C_UNDIFF, Bedrock.P_UNDIFF]),
    'CMTS': frozenset([Bedrock.C_UNDIFF]),

    'PMHN': frozenset([Bedrock.P_UNDIFF]),
    'PMHF': frozenset([Bedrock.P_UNDIFF]),
    'PMFL': frozenset([Bedrock.P_UNDIFF]),
    'PMSC': frozenset([Bedrock.P_UNDIFF]),
}

#TODO: Sometimes a layer is two classifications at once, so we need to assign both as valid predictions
BEDROCK_SET_MAP = {
    'CAMB' : frozenset([Bedrock.C_UNDIFF]),
    'PUDF' : frozenset([Bedrock.P_UNDIFF]),
    'DEVO' : frozenset([Bedrock.D_UNDIFF]),
    'KRET' : frozenset([Bedrock.K_UNDIFF]),
    'ORDO' : frozenset([Bedrock.O_UNDIFF]),

    'KCRL' : frozenset([Bedrock.CARLILE,
                        Bedrock.K_UNDIFF]),

    'KCLR' : frozenset([Bedrock.COLERAINE,
                        Bedrock.K_UNDIFF]),

    'KREG' : frozenset([Bedrock.K_REGOLITH,
                        Bedrock.K_UNDIFF]),

    'KSRC' : frozenset([Bedrock.SPLIT_ROCK_CREEK,
                        Bedrock.K_UNDIFF]),

    'KWND' : frozenset([Bedrock.WINDROW,
                        Bedrock.K_UNDIFF]),

    'DCVU' : frozenset([Bedrock.UPPER_CEDAR,
                        Bedrock.CEDAR_VALLEY,
                        Bedrock.D_UNDIFF]),
    'DCLC' : frozenset([Bedrock.UPPER_CEDAR, Bedrock.LITTLE_CEDAR,
                        Bedrock.CEDAR_VALLEY,
                        Bedrock.D_UNDIFF]),

    'DCVA' : frozenset([Bedrock.CEDAR_VALLEY,
                        Bedrock.D_UNDIFF]),

    'DLCD' : frozenset([Bedrock.LITTLE_CEDAR,
                        Bedrock.CEDAR_VALLEY,
                        Bedrock.D_UNDIFF]),

    'DCLP' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.PINICON_RIDGE,
                        Bedrock.CEDAR_VALLEY, Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF]),
    'DCLS' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.SPILLVILLE,
                        Bedrock.CEDAR_VALLEY, Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF]),
    'DCOG' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.GALENA,
                        Bedrock.CEDAR_VALLEY,
                        Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DCOM' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.MAQUOKETA,
                        Bedrock.CEDAR_VALLEY,
                        Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DCVL' : frozenset([Bedrock.LOWER_CEDAR,
                        Bedrock.CEDAR_VALLEY,
                        Bedrock.D_UNDIFF]),

    'DWAP' : frozenset([Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF]),
    'DWPR' : frozenset([Bedrock.WAPSIPINICON, Bedrock.PINICON_RIDGE,
                        Bedrock.D_UNDIFF]),

    'DPOG' : frozenset([Bedrock.PINICON_RIDGE, Bedrock.GALENA,
                        Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DPOM' : frozenset([Bedrock.PINICON_RIDGE, Bedrock.MAQUOKETA,
                        Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),

    'DSOG' : frozenset([Bedrock.SPILLVILLE, Bedrock.GALENA,
                        Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DSOM' : frozenset([Bedrock.SPILLVILLE, Bedrock.MAQUOKETA,
                        Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF, Bedrock.O_UNDIFF]),
    'DSPL' : frozenset([Bedrock.SPILLVILLE,
                        Bedrock.WAPSIPINICON,
                        Bedrock.D_UNDIFF]),

    'OMAQ' : frozenset([Bedrock.MAQUOKETA,
                        Bedrock.O_UNDIFF]),
    'OMQD' : frozenset([Bedrock.MAQUOKETA, Bedrock.DUBUQUE,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OMQG' : frozenset([Bedrock.MAQUOKETA, Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),

    'OGAP' : frozenset([Bedrock.GALENA, Bedrock.ST_PETER,
                        Bedrock.O_UNDIFF]),
    'OGGP' : frozenset([Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OGPD' : frozenset([Bedrock.GALENA, Bedrock.PROSSER,
                        Bedrock.O_UNDIFF]),

    'ODGL' : frozenset([Bedrock.DUBUQUE, Bedrock.CUMMINGSVILLE,
                        Bedrock.STEWARTVILLE, Bedrock.PROSSER, #Technically it goes through these two, consult geologist
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'ODUB' : frozenset([Bedrock.DUBUQUE,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),

    'OGSC' : frozenset([Bedrock.STEWARTVILLE, Bedrock.CUMMINGSVILLE,
                        Bedrock.PROSSER, #Goes through, consult
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OGSD' : frozenset([Bedrock.STEWARTVILLE, Bedrock.DECORAH,
                        Bedrock.PROSSER, Bedrock.CUMMINGSVILLE, #Goes through, consult
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OGVP' : frozenset([Bedrock.STEWARTVILLE, Bedrock.PROSSER,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OGSV' : frozenset([Bedrock.STEWARTVILLE,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),

    'OGPC' : frozenset([Bedrock.PROSSER, Bedrock.CUMMINGSVILLE,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OGPR' : frozenset([Bedrock.PROSSER,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),

    'OGCM' : frozenset([Bedrock.CUMMINGSVILLE,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OGCD' : frozenset([Bedrock.CUMMINGSVILLE, Bedrock.DECORAH,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),

    'ODCR' : frozenset([Bedrock.DECORAH,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'OGDP' : frozenset([Bedrock.DECORAH, Bedrock.PLATTEVILLE,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'ODPG' : frozenset([Bedrock.DECORAH, Bedrock.PLATTEVILLE, Bedrock.GLENWOOD,
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),
    'ODSP' : frozenset([Bedrock.DECORAH, Bedrock.ST_PETER,
                        Bedrock.PLATTEVILLE, bedrock.GLENWOOD, #Goes through, consult
                        Bedrock.GALENA,
                        Bedrock.O_UNDIFF]),

    'OPGW' : frozenset([Bedrock.PLATTEVILLE, Bedrock.GLENWOOD,
                        Bedrock.O_UNDIFF]),
    'OPSP' : frozenset([Bedrock.PLATTEVILLE, Bedrock.ST_PETER,
                        Bedrock.GLENWOOD, #Goes through, consult
                        Bedrock.O_UNDIFF]),
    'OPVJ' : frozenset([Bedrock.PLATTEVILLE, Bedrock.JORDAN,
                        Bedrock.ST_PETER, Bedrock.PRARIE_DU_CHIEN, Bedrock.SHAKOPEE, Bedrock.ONEOTA, #Goes through, consult
                        Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OPVL' : frozenset([Bedrock.PLATTEVILLE,
                        Bedrock.O_UNDIFF]),

    'OGSP' : frozenset([Bedrock.GLENWOOD, Bedrock.ST_PETER,
                        Bedrock.O_UNDIFF]),
    'OGWD' : frozenset([Bedrock.GLENWOOD,
                        Bedrock.O_UNDIFF]),

    'OSCJ' : frozenset([Bedrock.ST_PETER, Bedrock.JORDAN,
                        Bedrock.SHAKOPEE, Bedrock.ONEOTA, Bedrock.PRARIE_DU_CHIEN,
                        Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OSCS' : frozenset([Bedrock.ST_PETER, Bedrock.ST_LAWRENCE,
                        Bedrock.SHAKOPEE, Bedrock.ONEOTA, Bedrock.PRARIE_DU_CHIEN, Bedrock.JORDAN,
                        Bedrock.O_UNDIFF, Bedrock.C_UNDIFF]),
    'OSPC' : frozenset([Bedrock.ST_PETER, Bedrock.PRARIE_DU_CHIEN,
                        Bedrock.]),
    'OSTP' : frozenset([Bedrock.ST_PETER]),

    'OPCJ' : frozenset([Bedrock.PRARIE_DU_CHIEN, Bedrock.JORDAN]),
    'OPCM' : frozenset([Bedrock.PRARIE_DU_CHIEN, Bedrock.MT_SIMON]),
    'OPCS' : frozenset([Bedrock.PRARIE_DU_CHIEN, Bedrock.ST_LAWRENCE]),
    'OPCT' : frozenset([Bedrock.PRARIE_DU_CHIEN, Bedrock.TUNNEL_CITY]),
    'OPDC' : frozenset([Bedrock.PRARIE_DU_CHIEN]),

    'OPNR' : frozenset([Bedrock.SHAKOPEE, Bedrock.RICHMOND]),
    'OPWR' : frozenset([Bedrock.SHAKOPEE, Bedrock.WILLOW_RIVER]),
    'OPSH' : frozenset([Bedrock.SHAKOPEE]),

    'OPOD' : frozenset([Bedrock.ONEOTA]),

    'OWIN' : frozenset([Bedrock.WINNIPEG]),

    'CJDN' : frozenset([Bedrock.JORDAN]),
    'CJMS' : frozenset([Bedrock.JORDAN, Bedrock.MT_SIMON]),
    'CJEC' : frozenset([Bedrock.JORDAN, Bedrock.EAU_CLAIRE]),
    'CJDW' : frozenset([Bedrock.JORDAN, Bedrock.WONEWOC]),
    'CJSL' : frozenset([Bedrock.JORDAN, Bedrock.ST_LAWRENCE]),
    'CJTC' : frozenset([Bedrock.JORDAN, Bedrock.TUNNEL_CITY]),

    'CSLT' : frozenset([Bedrock.ST_LAWRENCE, Bedrock.TUNNEL_CITY]),
    'CSLW' : frozenset([Bedrock.ST_LAWRENCE, Bedrock.WONEWOC]),
    'CSTL' : frozenset([Bedrock.ST_LAWRENCE]),

    'CTCG' : frozenset([Bedrock.TUNNEL_CITY]),
    'CTCM' : frozenset([Bedrock.TUNNEL_CITY, Bedrock.MT_SIMON]),
    'CTCW' : frozenset([Bedrock.TUNNEL_CITY, Bedrock.WONEWOC]),
    'CTCE' : frozenset([Bedrock.TUNNEL_CITY, Bedrock.EAU_CLAIRE]),

    'CTMZ' : frozenset([Bedrock.MAZOMANIE]),

    'CTLR' : frozenset([Bedrock.LONE_ROCK]),

    'CWMS' : frozenset([Bedrock.WONEWOC, Bedrock.MT_SIMON]),
    'CWOC' : frozenset([Bedrock.WONEWOC]),
    'CWEC' : frozenset([Bedrock.WONEWOC, Bedrock.EAU_CLAIRE]),

    'CECR' : frozenset([Bedrock.EAU_CLAIRE]),
    'CEMS' : frozenset([Bedrock.EAU_CLAIRE, Bedrock.MT_SIMON]),

    'CMFL' : frozenset([Bedrock.MT_SIMON, Bedrock.FOND_DU_LAC]),
    'CMSH' : frozenset([Bedrock.MT_SIMON, Bedrock.HINCKLEY]),
    'CMTS' : frozenset([Bedrock.MT_SIMON]),

    'PMHN' : frozenset([Bedrock.HINCKLEY]),
    'PMHF'  : frozenset([Bedrock.HINCKLEY, Bedrock.FOND_DU_LAC]),

    'PMFL' : frozenset([Bedrock.FOND_DU_LAC]),

    'PMSC' : frozenset([Bedrock.SOLOR_CHURCH]),
}

def load_bedrock_categories(df : pd.DataFrame):
    global BEDROCK_CATEGORIES, INV_BEDROCK_CATEGORIES

    BEDROCK_CATEGORIES = {val: i for i, val in enumerate(df['strat'].unique())}
    INV_BEDROCK_CATEGORIES = {v: k for k, v in BEDROCK_CATEGORIES.items()}