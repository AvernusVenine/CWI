import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

from Data import Field

"""def init_encoders():

    group_encoder = LabelEncoder()
    group_encoder.fit([group.name for group in GROUP_LIST] + [None])

    formation_encoder = LabelEncoder()
    formation_encoder.fit([formation.name for formation in FORMATION_LIST] + [None])

    member_encoder = LabelEncoder()
    member_encoder.fit([member.name for member in MEMBER_LIST] + [None])

    return group_encoder, formation_encoder, member_encoder"""

"""def encode_bedrock(df):

    group_encoder = LabelEncoder()
    group_encoder.fit([None] + [group.name for group in GROUP_LIST])

    formation_encoder = LabelEncoder()
    formation_encoder.fit([formation.name for formation in FORMATION_LIST] + [None])

    member_encoder = LabelEncoder()
    member_encoder.fit([member.name for member in MEMBER_LIST] + [None])

    df[Field.GROUP_TOP] = group_encoder.transform([None])[0]
    df[Field.GROUP_BOT] = group_encoder.transform([None])[0]

    df[Field.FORMATION_TOP] = formation_encoder.transform([None])[0]
    df[Field.FORMATION_BOT] = formation_encoder.transform([None])[0]

    df[Field.MEMBER_TOP] = member_encoder.transform([None])[0]
    df[Field.MEMBER_BOT] = member_encoder.transform([None])[0]

    for key, value in BEDROCK_CODE_MAP.items():
        mask = df[Field.STRAT] == key

        df.loc[mask, Field.GROUP_TOP] = group_encoder.transform([value.top_group.name if value.top_group else None])[0]
        df.loc[mask, Field.GROUP_BOT] = group_encoder.transform([value.bot_group.name if value.bot_group else None])[0]

        df.loc[mask, Field.FORMATION_TOP] = formation_encoder.transform([value.top_formation.name if value.top_formation else None])[0]
        df.loc[mask, Field.FORMATION_BOT] = formation_encoder.transform([value.bot_formation.name if value.bot_formation else None])[0]

        df.loc[mask, Field.MEMBER_TOP] = member_encoder.transform([value.top_member.name if value.top_member else None])[0]
        df.loc[mask, Field.MEMBER_BOT] = member_encoder.transform([value.bot_member.name if value.bot_member else None])[0]

    return df"""

class Bedrock:

    def __init__(self, name : str, parent=None):
        self.name = name
        self.parent = parent

    def get_lineage(self):
        lineage = []
        unit = self

        while unit:
            lineage.append(unit)
            unit = unit.parent

        return lineage

class GeoCode:

    def __init__(self, bedrock):
        if isinstance(bedrock, list):
            self.top = bedrock[0]
            self.bot = bedrock[1]
        else:
            self.top = bedrock
            self.bot = bedrock

        self.top_lineage = [bdrk.name for bdrk in self.top.get_lineage()]
        self.top_lineage.reverse()
        self.bot_lineage = [bdrk.name for bdrk in self.bot.get_lineage()]
        self.bot_lineage.reverse()

    def __eq__(self, other):
        if not isinstance(other, GeoCode):
            return False

        return self.top == other.top and self.bot == other.bot

    def __contains__(self, other):
        if not isinstance(other, GeoCode):
            return False

        return set(other.top_lineage).issubset(set(self.top_lineage)) and set(other.bot_lineage).issubset(set(self.bot_lineage))

    def get_top(self, idx):
        if idx >= len(self.top_lineage):
            return None

        return self.top_lineage[idx]

    def get_bot(self, idx):
        if idx >= len(self.bot_lineage):
            return None

        return self.bot_lineage[idx]

"""BEDROCKS"""
Cretaceous = Bedrock('K')
Devonian = Bedrock('D')
Ordovician = Bedrock('O')
Cambrian = Bedrock('C')
Precambrian = Bedrock('G')

Cretaceous_Regolith = Bedrock('Cretaceous Regolith', parent=Cretaceous)
Carlile_Shale = Bedrock('Carlile Shale', parent=Cretaceous)
Dakota = Bedrock('Dakota', parent=Cretaceous)
Split_Rock = Bedrock('Split Rock', parent=Cretaceous)
Windrow = Bedrock('Windrow', parent=Cretaceous)

Cedar_Valley = Bedrock('Cedar Valley', parent=Devonian)
Upper_Cedar = Bedrock('Upper Cedar Valley', parent=Cedar_Valley)
Lower_Cedar = Bedrock('Lower Cedar Valley', parent=Cedar_Valley)
Little_Cedar = Bedrock('Little Cedar', parent=Lower_Cedar)
Coralville = Bedrock('Coralville', parent=Upper_Cedar)
Lithograph_City = Bedrock('Lithograph City', parent=Upper_Cedar)
Wapsipinicon = Bedrock('Wapsipinicon', parent=Devonian)
Pinicon_Ridge = Bedrock('Pinicon Ridge', parent=Wapsipinicon)
Spillville = Bedrock('Spillville', parent=Wapsipinicon)

Maquoketa = Bedrock('Maquoketa', parent=Ordovician)
Galena = Bedrock('Galena', parent=Ordovician)
Cummingsville = Bedrock('Cummingsville', parent=Galena)
Prosser = Bedrock('Prosser', parent=Galena)
Stewartville = Bedrock('Stewartville', parent=Galena)
Dubuque = Bedrock('Dubuque', parent=Galena)
Decorah_Shale = Bedrock('Decorah Shale', parent=Ordovician)
Platteville = Bedrock('Platteville', parent=Ordovician)
Glenwood = Bedrock('Glenwood', parent=Ordovician)
St_Peter = Bedrock('St Peter', parent=Ordovician)
Winnipeg = Bedrock('Winnipeg', parent=Ordovician)

Prairie_Du_Chien = Bedrock('Prairie Du Chien', parent=Cambrian)
Oneota = Bedrock('Oneota', parent=Prairie_Du_Chien)
Shakopee = Bedrock('Shakopee', parent=Prairie_Du_Chien)
New_Richmond = Bedrock('New Richmond', parent=Shakopee)
Willow_River = Bedrock('Willow River', parent=Shakopee)
Jordan = Bedrock('Jordan', parent=Cambrian)
St_Lawrence = Bedrock('St Lawrence', parent=Cambrian)
Tunnel_City = Bedrock('Tunnel City', parent=Cambrian)
Lone_Rock = Bedrock('Lone Rock', parent=Tunnel_City)
Mazomanie = Bedrock('Mazomanie', parent=Tunnel_City)
Wonewoc = Bedrock('Wonewoc', parent=Cambrian)
Eau_Claire = Bedrock('Eau Claire', parent=Cambrian)
Mt_Simon = Bedrock('Mt Simon', parent=Cambrian)

Hinckley = Bedrock('Hinckley', parent=Precambrian)
Fond_Du_Lac = Bedrock('Fond Du Lac', parent=Precambrian)
Solor_Church = Bedrock('Solor Church', parent=Precambrian)

"""CODE MAPPINGS"""
BEDROCK_CODE_MAP = {
    'CAMB' : GeoCode(Cambrian),
    'DEVO' : GeoCode(Devonian),
    'KRET' : GeoCode(Cretaceous),
    'ORDO' : GeoCode(Ordovician),

    'KCBH' : GeoCode(Carlile_Shale),
    'KCCD' : GeoCode(Carlile_Shale),
    'KCFP' : GeoCode(Carlile_Shale),
    'KCRL' : GeoCode(Carlile_Shale),
    'KDKT' : GeoCode(Dakota),
    'KDNB' : GeoCode(Dakota),
    'KDWB' : GeoCode(Dakota),
    'KREG' : GeoCode(Cretaceous_Regolith),
    'KSRC' : GeoCode(Split_Rock),
    'KWIH' : GeoCode(Windrow),
    'KWND' : GeoCode(Windrow),
    'KWOS' : GeoCode(Windrow),

    'DCVU' : GeoCode(Upper_Cedar),
    'DLGH' : GeoCode(Lithograph_City),
    'DCRL' : GeoCode(Coralville),
    'DCUM' : GeoCode([Coralville, Little_Cedar]),
    'DCGZ' : GeoCode(Coralville),
    'DCIC' : GeoCode(Coralville),
    'DCLC' : GeoCode([Coralville, Little_Cedar]),
    'DCVA' : GeoCode(Cedar_Valley),
    'DLBA' : GeoCode(Little_Cedar),
    'DLCB' : GeoCode(Little_Cedar),
    'DLCD' : GeoCode(Little_Cedar),
    'DLCH' : GeoCode(Little_Cedar),
    'DLHE' : GeoCode(Little_Cedar),
    'DCLP' : GeoCode([Little_Cedar, Pinicon_Ridge]),
    'DCLS' : GeoCode([Little_Cedar, Spillville]),
    'DCOG' : GeoCode([Cedar_Valley, Galena]),
    'DCOM' : GeoCode([Cedar_Valley, Maquoketa]),
    'DCVL' : GeoCode(Lower_Cedar),
    'DWAP' : GeoCode(Wapsipinicon),
    'DWPR' : GeoCode(Pinicon_Ridge),
    'DPOG' : GeoCode([Pinicon_Ridge, Galena]),
    'DPOM' : GeoCode([Pinicon_Ridge, Maquoketa]),
    'DSOG' : GeoCode([Spillville, Galena]),
    'DSOM' : GeoCode([Spillville, Maquoketa]),
    'DSPL' : GeoCode(Spillville),

    'OMAQ' : GeoCode(Maquoketa),
    'OMQD' : GeoCode([Maquoketa, Dubuque]),
    'OMQG' : GeoCode([Maquoketa, Galena]),
    'OGAP' : GeoCode([Galena, St_Peter]),
    'OGDP' : GeoCode([Galena, Platteville]),
    'OGGP' : GeoCode(Galena),
    'OGPD' : GeoCode([Prosser, Decorah_Shale]),
    'ODGL' : GeoCode([Dubuque, Cummingsville]),
    'ODUB' : GeoCode(Dubuque),
    'OGSC' : GeoCode([Stewartville, Cummingsville]),
    'OGSD' : GeoCode([Stewartville, Decorah_Shale]),
    'OGVP' : GeoCode([Stewartville, Prosser]),
    'OGSV' : GeoCode(Stewartville),
    'OGPC' : GeoCode([Prosser, Cummingsville]),
    'OGPR' : GeoCode(Prosser),
    'OGCM' : GeoCode(Cummingsville),
    'OGCD' : GeoCode([Cummingsville, Decorah_Shale]),
    'ODCA' : GeoCode(Decorah_Shale),
    'ODCR' : GeoCode(Decorah_Shale),
    'ODPG' : GeoCode([Decorah_Shale, Glenwood]),
    'ODPL' : GeoCode([Decorah_Shale, Platteville]),
    'ODSP' : GeoCode([Decorah_Shale, St_Peter]),
    'OPGW' : GeoCode([Platteville, Glenwood]),
    'OPHF' : GeoCode(Platteville),
    'OPMA' : GeoCode(Platteville),
    'OPMI' : GeoCode(Platteville),
    'OPPE' : GeoCode(Platteville),
    'OPSP' : GeoCode([Platteville, St_Peter]),
    'OPVJ' : GeoCode([Platteville, Jordan]),
    'OPVL' : GeoCode(Platteville),
    'OGSP' : GeoCode([Glenwood, St_Peter]),
    'OGWD' : GeoCode(Glenwood),
    'OSCJ' : GeoCode([St_Peter, Jordan]),
    'OSCS' : GeoCode([St_Peter, St_Lawrence]),
    'OSPC' : GeoCode([St_Peter, Prairie_Du_Chien]),
    'OSPE' : GeoCode(St_Peter),
    'OSTN' : GeoCode(St_Peter),
    'OSTP' : GeoCode(St_Peter),
    'OPCJ' : GeoCode([Prairie_Du_Chien, Jordan]),
    'OPCM' : GeoCode([Prairie_Du_Chien, Mt_Simon]),
    'OPCS' : GeoCode([Prairie_Du_Chien, St_Lawrence]),
    'OPCT' : GeoCode([Prairie_Du_Chien, Tunnel_City]),
    'OPDC' : GeoCode(Prairie_Du_Chien),
    'OPNR' : GeoCode(New_Richmond),
    'OPWR' : GeoCode(Willow_River),
    'OPSH' : GeoCode(Shakopee),
    'OOCV' : GeoCode(Oneota),
    'OOHC' : GeoCode(Oneota),
    'OPOD' : GeoCode(Oneota),
    #'ORRV' : GeoCode(Red_River),
    #'OSTM' : GeoCode(Stoney_Mountain),
    'OWBI' : GeoCode(Winnipeg),
    'OWIB' : GeoCode(Winnipeg),
    'OWIN' : GeoCode(Winnipeg),

    'CJDN' : GeoCode(Jordan),
    'CJEC' : GeoCode([Jordan, Eau_Claire]),
    'CJDW' : GeoCode([Jordan, Wonewoc]),
    'CJMS' : GeoCode([Jordan, Mt_Simon]),
    'CJSL' : GeoCode([Jordan, St_Lawrence]),
    'CJTC' : GeoCode([Jordan, Tunnel_City]),
    'CSLT' : GeoCode([St_Lawrence, Tunnel_City]),
    'CSLW' : GeoCode([St_Lawrence, Wonewoc]),
    'CSTL' : GeoCode(St_Lawrence),
    'CTCG' : GeoCode(Tunnel_City),
    'CTCM' : GeoCode([Tunnel_City, Mt_Simon]),
    'CTCW' : GeoCode([Tunnel_City, Wonewoc]),
    'CTCE' : GeoCode([Tunnel_City, Eau_Claire]),
    'CTMZ' : GeoCode(Mazomanie),
    'CTLR' : GeoCode(Lone_Rock),
    'CLBK' : GeoCode(Lone_Rock),
    'CLRE' : GeoCode(Lone_Rock),
    'CWMS' : GeoCode([Wonewoc, Mt_Simon]),
    'CWOC' : GeoCode(Wonewoc),
    'CWEC' : GeoCode([Wonewoc, Eau_Claire]),
    'CECR' : GeoCode(Eau_Claire),
    'CEMS' : GeoCode([Eau_Claire, Mt_Simon]),
    'CMFL' : GeoCode([Mt_Simon, Fond_Du_Lac]),
    'CMRC' : GeoCode(Mt_Simon),
    'CMSH' : GeoCode([Mt_Simon, Hinckley]),
    'CMTS' : GeoCode(Mt_Simon),

    'PMHN' : GeoCode(Hinckley),
    'PMHF' : GeoCode([Hinckley, Fond_Du_Lac]),
    'PMFL' : GeoCode(Fond_Du_Lac),
    'PMSC' : GeoCode(Solor_Church),
}