import pandas as pd

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

BEDROCK_AGES = ('C', 'D', 'K', 'O', 'P')

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
BEDROCK_PARENT_MAP_TEMP = {
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

#TODO: Sometimes a layer is two classifications at once, so we need to assign both as valid predictions
BEDROCK_SET_MAP = {
    'CAMB' : frozenset([Bedrock.C_UNDIFF]),
    'PUDF' : frozenset([Bedrock.P_UNDIFF]),
    'DEVO' : frozenset([Bedrock.D_UNDIFF]),
    'KRET' : frozenset([Bedrock.K_UNDIFF]),
    'ORDO' : frozenset([Bedrock.O_UNDIFF]),

    'KCRL' : frozenset([Bedrock.CARLILE]),

    'KCLR' : frozenset([Bedrock.COLERAINE]),

    'KREG' : frozenset([Bedrock.K_REGOLITH]),

    'KSRC' : frozenset([Bedrock.SPLIT_ROCK_CREEK]),

    'KWND' : frozenset([Bedrock.WINDROW]),

    'DCVU' : frozenset([Bedrock.UPPER_CEDAR]),
    'DCLC' : frozenset([Bedrock.UPPER_CEDAR, Bedrock.LITTLE_CEDAR]),

    'DCVA' : frozenset([Bedrock.CEDAR_VALLEY]),

    'DLCD' : frozenset([Bedrock.LITTLE_CEDAR]),

    'DCLP' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.PINICON_RIDGE]),
    'DCLS' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.SPILLVILLE]),
    'DCOG' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.GALENA]),
    'DCOM' : frozenset([Bedrock.LOWER_CEDAR, Bedrock.MAQUOKETA]),
    'DCVL' : frozenset([Bedrock.LOWER_CEDAR]),

    'DWAP' : frozenset([Bedrock.WAPSIPINICON]),
    'DWPR' : frozenset([Bedrock.WAPSIPINICON, Bedrock.PINICON_RIDGE]),

    'DPOG' : frozenset([Bedrock.PINICON_RIDGE, Bedrock.GALENA]),
    'DPOM' : frozenset([Bedrock.PINICON_RIDGE, Bedrock.MAQUOKETA]),

    'DSOG' : frozenset([Bedrock.SPILLVILLE, Bedrock.GALENA]),
    'DSOM' : frozenset([Bedrock.SPILLVILLE, Bedrock.MAQUOKETA]),
    'DSPL' : frozenset([Bedrock.SPILLVILLE]),

    'OMAQ' : frozenset([Bedrock.MAQUOKETA]),
    'OMQD' : frozenset([Bedrock.MAQUOKETA, Bedrock.DUBUQUE]),
    'OMQG' : frozenset([Bedrock.MAQUOKETA, Bedrock.GALENA]),

    'OGAP' : frozenset([Bedrock.GALENA, Bedrock.ST_PETER]),
    'OGGP' : frozenset([Bedrock.GALENA]),
    'OGPD' : frozenset([Bedrock.GALENA, Bedrock.PROSSER]),

    'ODGL' : frozenset([Bedrock.DUBUQUE, Bedrock.CUMMINGSVILLE]),
    'ODUB' : frozenset([Bedrock.DUBUQUE]),

    'OGSC' : frozenset([Bedrock.STEWARTVILLE, Bedrock.CUMMINGSVILLE]),
    'OGSD' : frozenset([Bedrock.STEWARTVILLE, Bedrock.DECORAH]),
    'OGVP' : frozenset([Bedrock.STEWARTVILLE, Bedrock.PROSSER]),
    'OGSV' : frozenset([Bedrock.STEWARTVILLE]),

    'OGPC' : frozenset([Bedrock.PROSSER, Bedrock.CUMMINGSVILLE]),
    'OGPR' : frozenset([Bedrock.PROSSER]),

    'OGCM' : frozenset([Bedrock.CUMMINGSVILLE]),
    'OGCD' : frozenset([Bedrock.CUMMINGSVILLE, Bedrock.DECORAH]),

    'ODCR' : frozenset([Bedrock.DECORAH]),
    'OGDP' : frozenset([Bedrock.DECORAH, Bedrock.PLATTEVILLE, Bedrock.GALENA]),
    'ODPG' : frozenset([Bedrock.DECORAH, Bedrock.PLATTEVILLE, Bedrock.GLENWOOD]),
    'ODPL' : frozenset([Bedrock.DECORAH, Bedrock.PLATTEVILLE]),
    'ODSP' : frozenset([Bedrock.DECORAH, Bedrock.ST_PETER]),

    'OPGW' : frozenset([Bedrock.PLATTEVILLE, Bedrock.GLENWOOD]),
    'OPSP' : frozenset([Bedrock.PLATTEVILLE, Bedrock.ST_PETER]),
    'OPVJ' : frozenset([Bedrock.PLATTEVILLE, Bedrock.JORDAN]),
    'OPVL' : frozenset([Bedrock.PLATTEVILLE]),

    'OGSP' : frozenset([Bedrock.GLENWOOD, Bedrock.ST_PETER]),
    'OGWD' : frozenset([Bedrock.GLENWOOD]),

    'OSCJ' : frozenset([Bedrock.ST_PETER, Bedrock.JORDAN]),
    'OSCS' : frozenset([Bedrock.ST_PETER, Bedrock.ST_LAWRENCE]),
    'OSPC' : frozenset([Bedrock.ST_PETER, Bedrock.PRARIE_DU_CHIEN]),
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
}

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
    'KGRN' : 'KRET', #

    # Ganeros Shale
    'KGRS' : 'KRET', #

    # Carlile Shale
    'KCBH': 'KCRL',
    'KCCD': 'KCRL',
    'KCFP': 'KCRL',
    'KCRL': 'KRET', #

    # Coleraine
    'KCLR': 'KRET', #

    # Split Rock Creek
    'KSRC': 'KRET', #

    # Windrow
    'KWIH': 'KWND',
    'KWND': 'KRET', #
    'KWOS': 'KWND',

    # Dakota Sandstone
    'KDKT': 'KRET', #
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
    'DCVA': 'DCVA', #

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
    'DWAP': 'DWAP', #

    # Pinicon Ridge
    'DWPR': 'DWAP', #
    'DPOG': 'DWPR',
    'DPOM': 'DWPR',

    # Spillville
    'DSOG': 'DSPL',
    'DSOM': 'DSPL',
    'DSPL': 'DWAP', #


    # Red River
    'ORRV': 'ORDO', #

    # Winnipeg
    'OWBI': 'OWIN',
    'OWIB': 'OWIN',
    'OWIN': 'OWIN', #

    # Maquoketa Formation
    'OMAQ' : 'OMAQ', #
    'OMQD' : 'OMAQ',
    'OMQG' : 'OMAQ',

    # Galena Group
    'OGGP' : 'OGGP', #
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
    'OPVL' : 'OPVL', #

    # Glenwood
    'OGSP' : 'OGWD',
    'OGWD' : 'OGWD', #

    # St. Peter Sandstone
    'OSCJ' : 'OSTP',
    'OSCS' : 'OSTP',
    'OSPC' : 'OSTP',
    'OSPE' : 'OSTP',
    'OSTN' : 'OSTP',
    'OSTP' : 'OSTP', #

    # Praire Du Chien
    'OPCJ' : 'OPDC',
    'OPCM' : 'OPDC',
    'OPCS' : 'OPDC',
    'OPCT' : 'OPDC',
    'OPDC' : 'OPDC', #

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
    'CJDN' : 'CJDN', #
    'CJEC' : 'CJDN',
    'CJDW' : 'CJDN',
    'CJMS' : 'CJDN',
    'CJSL' : 'CJDM',
    'CJTC' : 'CJDN',

    # St. Lawrence
    'CSLT' : 'CSLT', #
    'CSLW' : 'CSTL',
    'CSTL' : 'CSTL',

    # Tunnel City
    'CTCG' : 'CTCG', #
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
    'CWOC' : 'CWOC', #
    'CWEC' : 'CWOC',

    # Eau Claire
    'CECR' : 'CECR', #
    'CEMS' : 'CECR',

    # Mt. Simon
    'CMFL' : 'CMTS',
    'CMRC' : 'CMTS',
    'CMSH' : 'CMTS',
    'CMTS' : 'CMTS', #

    # Hinckley Sandstone
    'PMHN' : 'PMHN', #
    'PMHF' : 'PMHN',
}


def load_bedrock_categories(df : pd.DataFrame):
    global BEDROCK_CATEGORIES, INV_BEDROCK_CATEGORIES

    BEDROCK_CATEGORIES = {val: i for i, val in enumerate(df['strat'].unique())}
    INV_BEDROCK_CATEGORIES = {v: k for k, v in BEDROCK_CATEGORIES.items()}