import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re
import warnings

from Data import Field

def init_encoders():
    """
    Initializes and fits encoders for group, formation, and member
    :return: Group encoder, formation encoder, member encoder
    """

    group_encoder = LabelEncoder()
    group_encoder.fit([group.name for group in GROUP_LIST] + [None])

    formation_encoder = LabelEncoder()
    formation_encoder.fit([formation.name for formation in FORMATION_LIST] + [None])

    member_encoder = LabelEncoder()
    member_encoder.fit([member.name for member in MEMBER_LIST] + [None])

    return group_encoder, formation_encoder, member_encoder

def encode_bedrock(df):
    """
    Determines whether the layer has a bedrock component, and if it does it extracts and encodes it
    :param df: Layer dataframe
    :return: Bedrock encoded dataframe
    """

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

    return df

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

class Primary(Bedrock):
    pass

class Secondary(Bedrock):
    pass

class Tertiary(Bedrock):
    pass

class Age(Bedrock):
    pass

class Group(Bedrock):
    pass

class Formation(Bedrock):
    pass

class Member(Bedrock):
    pass


class GeoCode:

    def __init__(self, primary=None, secondary=None, tertiary=None):
        if isinstance(primary, list):
            self.primary_top = primary[0]
            self.primary_bot = primary[1]
        else:
            self.primary_top = primary
            self.primary_bot = primary

        if isinstance(secondary, list):
            self.secondary_top = secondary[0]
            self.primary_bot = secondary[1]
        else:
            self.secondary_top = secondary
            self.secondary_bot = secondary

        if isinstance(tertiary, list):
            self.tertiary_top = tertiary[0]
            self.tertiary_bot = tertiary[1]
        else:
            self.tertiary_top = tertiary
            self.tertiary_bot = tertiary


    def __eq__(self, other):
        if not isinstance(other, GeoCode):
            return False

        groups = (self.top_group == other.top_group) & (self.bot_group == other.bot_group)
        formations = (self.top_formation == other.top_formation) & (self.bot_formation == other.bot_formation)
        members = (self.top_member == other.top_member) & (self.bot_member == other.bot_member)

        return members and formations and groups

    @property
    def top_group(self):
        for item in self.top.get_lineage():
            if isinstance(item, Group):
                return item

        return None

    @property
    def top_formation(self):
        for item in self.top.get_lineage():
            if isinstance(item, Formation):
                return item

        return None

    @property
    def top_member(self):
        for item in self.top.get_lineage():
            if isinstance(item, Member):
                return item

        return None

    @property
    def bot_group(self):
        for item in self.bot.get_lineage():
            if isinstance(item, Group):
                return item

        return None

    @property
    def bot_formation(self):
        for item in self.bot.get_lineage():
            if isinstance(item, Formation):
                return item

        return None

    @property
    def bot_member(self):
        for item in self.bot.get_lineage():
            if isinstance(item, Member):
                return item

        return None

"""BEDROCK PRIMARY"""
Cretaceous_Regolith = Primary('Cretaceous Regolith')
Carlile_Shale = Primary('Carlile Shale')
Dakota = Primary('Dakota')
Split_Rock = Primary('Split Rock')
Windrow = Primary('Windrow')
Upper_Cedar = Primary('Upper Cedar Valley')
Cedar_Valley = Primary('Cedar Valley')
Lower_Cedar = Primary('Lower Cedar Valley')
Wapsipinicon = Primary('Wapsipinicon')
Maquoketa = Primary('Maquoketa')
Galena = Primary('Galena')
Decorah_Shale = Primary('Decorah Shale')
Platteville = Primary('Platteville')
Glenwood = Primary('Glenwood')
St_Peter = Primary('St Peter')
Prairie_Du_Chien = Primary('Prairie Du Chien')
Jordan = Primary('Jordan')
St_Lawrence = Primary('St Lawrence')
Tunnel_City = Primary('Tunnel City')
Wonewoc = Primary('Wonewoc')
Eau_Claire = Primary('Eau Claire')
Mt_Simon = Primary('Mt Simon')
Hinckley = Primary('Hinckley')
Fond_Du_Lac = Primary('Fond Du Lac')
Solor_Church = Primary('Solor Church')

"""BEDROCK SECONDARY"""
Lithograph_City = Secondary('Lithograph City')
Coralville = Secondary('Coralville')
Little_Cedar = Secondary('Little Cedar')
Pinicon_Ridge = Secondary('Pinicon Ridge')
Spillville = Secondary('Spillville')
Dubuque = Secondary('Dubuque')
Stewartville = Secondary('Stewartville')
Prosser = Secondary('Prosser')
Cummingsville = Secondary('Cummingsville')
Shakopee = Secondary('Shakopee')
Oneota = Secondary('Oneota')
Lone_Rock = Secondary('Lone Rock')
Mazomanie = Secondary('Mazomanie')

"""BEDROCK TERTIARY"""
Willow_River = Tertiary('Willow River')
New_Richmond = Tertiary('New Richmond')

"""CODE MAPPINGS"""
BEDROCK_CODE_MAP = {
    'KCBH' : GeoCode(),
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
    'DLGH' : GeoCode(Upper_Cedar, Lithograph_City),
    'DCRL' : GeoCode(Cedar_Valley, Coralville),
    'DCUM' : GeoCode([Cedar_Valley, Lower_Cedar], [Coralville, Little_Cedar]),
    'DCGZ' : GeoCode(Cedar_Valley, Coralville),
    'DCIC' : GeoCode(Cedar_Valley, Coralville),
    'DCLC' : GeoCode([Cedar_Valley, Lower_Cedar], [Coralville, Little_Cedar]),
    'DCVA' : GeoCode(Cedar_Valley),
    'DLBA' : GeoCode(Lower_Cedar, Little_Cedar),
    'DLCB' : GeoCode(Lower_Cedar, Little_Cedar),
    'DLCD' : GeoCode(Lower_Cedar, Little_Cedar),
    'DLCH' : GeoCode(Lower_Cedar, Little_Cedar),
    'DLHE' : GeoCode(Lower_Cedar, Little_Cedar),
    'DCLP' : GeoCode([Lower_Cedar, Wapsipinicon], [Little_Cedar, Pinicon_Ridge]),
    'DCLS' : GeoCode([Lower_Cedar, Wapsipinicon], [Little_Cedar, Spillville]),
    'DCOG' : GeoCode([Cedar_Valley, Galena]),
    'DCOM' : GeoCode([Cedar_Valley, Maquoketa]),
    'DCVL' : GeoCode(Lower_Cedar),
    'DWAP' : GeoCode(Wapsipinicon),
    'DWPR' : GeoCode(Wapsipinicon, Pinicon_Ridge),
    'DPOG' : GeoCode([Wapsipinicon, Galena], [Pinicon_Ridge, None]),
    'DPOM' : GeoCode([Wapsipinicon, Maquoketa], [Pinicon_Ridge, None]),
    'DSOG' : GeoCode([Wapsipinicon, Galena], [Spillville, None]),
    'DSOM' : GeoCode([Wapsipinicon, Maquoketa], [Spillville, None]),
    'DSPL' : GeoCode(Wapsipinicon, Spillville),

    'OMAQ' : GeoCode(Maquoketa),
    'OMQD' : GeoCode([Maquoketa, Dubuque]),
    'OMQG' : GeoCode([Maquoketa, Galena]),
    'OGAP' : GeoCode([Galena, St_Peter_Sandstone]),
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
    'ODCA' : GeoCode(Carimona),
    'ODCR' : GeoCode(Decorah_Shale),
    'ODPG' : GeoCode([Decorah_Shale, Glenwood]),
    'ODPL' : GeoCode([Decorah_Shale, Platteville]),
    'ODSP' : GeoCode([Decorah_Shale, St_Peter_Sandstone]),
    'OPGW' : GeoCode([Platteville, Glenwood]),
    'OPHF' : GeoCode(Hidden_Falls),
    'OPMA' : GeoCode(Magnolia),
    'OPMI' : GeoCode(Mifflin),
    'OPPE' : GeoCode(Pecatonica),
    'OPSP' : GeoCode([Platteville, St_Peter_Sandstone]),
    'OPVJ' : GeoCode([Platteville, Jordan_Sandstone]),
    'OPVL' : GeoCode(Platteville),
    'OGSP' : GeoCode([Glenwood, St_Peter_Sandstone]),
    'OGWD' : GeoCode(Glenwood),
    'OSCJ' : GeoCode([St_Peter_Sandstone, Jordan_Sandstone]),
    'OSCS' : GeoCode([St_Peter_Sandstone, St_Lawrence]),
    'OSPC' : GeoCode([St_Peter_Sandstone, Prairie_Du_Chien]),
    'OSPE' : GeoCode(Pigs_Eye),
    'OSTN' : GeoCode(Tonti),
    'OSTP' : GeoCode(St_Peter_Sandstone),
    'OPCJ' : GeoCode([Prairie_Du_Chien, Jordan_Sandstone]),
    'OPCM' : GeoCode([Prairie_Du_Chien, Mt_Simon_Sandstone]),
    'OPCS' : GeoCode([Prairie_Du_Chien, St_Lawrence]),
    'OPCT' : GeoCode([Prairie_Du_Chien, Tunnel_City]),
    'OPDC' : GeoCode(Prairie_Du_Chien),
    'OPNR' : GeoCode(New_Richmond),
    'OPWR' : GeoCode(Willow_River),
    'OPSH' : GeoCode(Shakopee),
    'OOCV' : GeoCode(Coon_Valley),
    'OOHC' : GeoCode(Hager_City),
    'OPOD' : GeoCode(Oneota),
    'ORRV' : GeoCode(Red_River),
    #'OSTM' : GeoCode(Stoney_Mountain),
    'OWBI' : GeoCode(Black_Island),
    'OWIB' : GeoCode(Icebox),
    'OWIN' : GeoCode(Winnipeg),

    'CJDN' : GeoCode(Jordan_Sandstone),
    'CJEC' : GeoCode([Jordan_Sandstone, Eau_Claire]),
    'CJDW' : GeoCode([Jordan_Sandstone, Wonewoc_Sandstone]),
    'CJMS' : GeoCode([Jordan_Sandstone, Mt_Simon_Sandstone]),
    'CJSL' : GeoCode([Jordan_Sandstone, St_Lawrence]),
    'CJTC' : GeoCode([Jordan_Sandstone, Tunnel_City]),
    'CSLT' : GeoCode([St_Lawrence, Tunnel_City]),
    'CSLW' : GeoCode([St_Lawrence, Wonewoc_Sandstone]),
    'CSTL' : GeoCode(St_Lawrence),
    'CTCG' : GeoCode(Tunnel_City),
    'CTCM' : GeoCode([Tunnel_City, Mt_Simon_Sandstone]),
    'CTCW' : GeoCode([Tunnel_City, Wonewoc_Sandstone]),
    'CTCE' : GeoCode([Tunnel_City, Eau_Claire]),
    'CTMZ' : GeoCode(Mazomanie),
    'CTLR' : GeoCode(Lone_Rock),
    'CLBK' : GeoCode(Birkmose),
    'CLRE' : GeoCode(Reno),
    'CWMS' : GeoCode([Wonewoc_Sandstone, Mt_Simon_Sandstone]),
    'CWOC' : GeoCode(Wonewoc_Sandstone),
    'CWEC' : GeoCode([Wonewoc_Sandstone, Eau_Claire]),
    'CECR' : GeoCode(Eau_Claire),
    'CEMS' : GeoCode([Eau_Claire, Mt_Simon_Sandstone]),
    'CMFL' : GeoCode([Mt_Simon_Sandstone, Fond_Du_Lac]),
    'CMRC' : GeoCode(Red_Clastics),
    'CMSH' : GeoCode([Mt_Simon_Sandstone, Hinckley_Sandstone]),
    'CMTS' : GeoCode(Mt_Simon_Sandstone),
    'PMHN' : GeoCode(Hinckley_Sandstone),
    'PMHF' : GeoCode([Hinckley_Sandstone, Fond_Du_Lac]),
    'PMFL' : GeoCode(Fond_Du_Lac),
    'PMSC' : GeoCode(Solor_Church),
}

"""BEDROCK AGES"""

Devonian = Age('Devonian')
Ordovician = Age('Ordovician')
Cambrian = Age('Cambrian')
Precambrian = Age('Precambrian')
Cretaceous = Age('Cretaceous')

"""BEDROCK GROUPS"""
Cedar_Valley = Group('Cedar Valley', parent=Devonian)
Upper_Cedar = Group('Upper Cedar Valley', parent=Devonian)
Lower_Cedar = Group('Lower Cedar Valley', parent=Devonian)
Wapsipinicon = Group('Wapsipinicon', parent=Devonian)
Galena = Group('Galena', parent=Ordovician)
Prairie_Du_Chien = Group('Prairie Du Chien', parent=Ordovician)
Tunnel_City = Group('Tunnel City', parent=Cambrian)

GROUP_LIST = [
    Cedar_Valley,
    Upper_Cedar,
    Lower_Cedar,
    Wapsipinicon,
    Galena,
    Prairie_Du_Chien,
    Tunnel_City
]
GROUP_DICT = {group.name: group for group in GROUP_LIST}

"""BEDROCK FORMATIONS"""
Cretaceous_Regolith = Formation('Cretaceous Regolith', parent=Cretaceous)
Carlile_Shale = Formation('Carlile Shale', parent=Cretaceous)
#Greenhorn_Limestone = Formation('Greenhorn Limestone', parent=Cretaceous)
#Ganeros_Shale = Formation('Ganeros Shale', parent=Cretaceous)
Dakota_Sandstone = Formation('Dakota Sandstone', parent=Cretaceous)
Split_Rock = Formation('Split Rock', parent=Cretaceous)
Windrow = Formation('Windrow', parent=Cretaceous)
Lithograph_City = Formation('Lithograph City', parent=Upper_Cedar)
Coralville = Formation('Coralville', parent=Cedar_Valley)
Little_Cedar = Formation('Little Cedar', parent=Lower_Cedar)
Pinicon_Ridge = Formation('Pinicon Ridge', parent=Wapsipinicon)
Spillville = Formation('Spillville', parent=Wapsipinicon)
Red_River = Formation('Red River', parent=Ordovician)
Winnipeg = Formation('Winnipeg', parent=Ordovician)
Maquoketa = Formation('Maquoketa', parent=Ordovician)
Dubuque = Formation('Dubuque', parent=Galena)
Stewartville = Formation('Stewartville', parent=Galena)
Prosser = Formation('Prosser', parent=Galena)
Cummingsville = Formation('Cummingsville', parent=Galena)
Decorah_Shale = Formation('Decorah Shale', parent=Ordovician)
Platteville = Formation('Platteville', parent=Ordovician)
Glenwood = Formation('Glenwood', parent=Ordovician)
St_Peter_Sandstone = Formation('St Peter Sandstone', parent=Ordovician)
Shakopee = Formation('Shakopee', parent=Prairie_Du_Chien)
Oneota = Formation('Oneota', parent=Prairie_Du_Chien)
#Stoney_Mountain = Formation('Stoney Mountain', parent=Ordovician)
Jordan_Sandstone = Formation('Jordan Sandstone', parent=Cambrian)
St_Lawrence = Formation('St Lawrence', parent=Cambrian)
Lone_Rock = Formation('Lone Rock', parent=Tunnel_City)
Mazomanie = Formation('Mazomanie', parent=Tunnel_City)
#Davis = Formation('Davis', parent=Tunnel_City)
Wonewoc_Sandstone = Formation('Wonewoc Sandstone', parent=Cambrian)
Eau_Claire = Formation('Eau Claire', parent=Cambrian)
#Bonneterre = Formation('Bonneterre', parent=Cambrian)
Mt_Simon_Sandstone = Formation('Mt Simon Sandstone', parent=Cambrian)
Hinckley_Sandstone = Formation('Hinckley Sandstone', parent=Precambrian)
Fond_Du_Lac = Formation('Fond Du Lac', parent=Precambrian)
Solor_Church = Formation('Solor Church', parent=Precambrian)

FORMATION_LIST = [
    Cretaceous_Regolith,
    Split_Rock,
    Windrow,
    Carlile_Shale,
    #Greenhorn_Limestone,
    #Ganeros_Shale,
    Dakota_Sandstone,
    Lithograph_City,
    Coralville,
    Little_Cedar,
    Pinicon_Ridge,
    Spillville,
    Red_River,
    Winnipeg,
    #Stoney_Mountain,
    Maquoketa,
    Dubuque,
    Stewartville,
    Prosser,
    Cummingsville,
    Decorah_Shale,
    Platteville,
    Glenwood,
    St_Peter_Sandstone,
    Shakopee,
    Oneota,
    Jordan_Sandstone,
    St_Lawrence,
    Lone_Rock,
    Mazomanie,
    #Davis,
    Wonewoc_Sandstone,
    Eau_Claire,
    #Bonneterre,
    Mt_Simon_Sandstone,
    Hinckley_Sandstone,
    Fond_Du_Lac,
    Solor_Church
]
FORMATION_DICT = {formation.name: formation for formation in FORMATION_LIST}

"""BEDROCK MEMBERS"""
Blue_Hill = Member('Blue Hill', parent=Carlile_Shale)
Codell_Sandstone = Member('Codell Sandstone', parent=Carlile_Shale)
Fairport = Member('Fairport', parent=Carlile_Shale)
Nishnabotna = Member('Nishnabotna', parent=Dakota_Sandstone)
Woodbury = Member('Woodbury', parent=Dakota_Sandstone)
Iron_Hill = Member('Iron Hill', parent=Windrow)
Ostrander = Member('Ostrander', parent=Windrow)
Hinckle = Member('Hinckle', parent=Little_Cedar)
Gizzard_Creek = Member('Gizzard Creek', parent=Coralville)
Iowa_City = Member('Iowa City', parent=Coralville)
Bassett = Member('Bassett', parent=Little_Cedar)
Chickasaw_Shale = Member('Chickasaw Shale', parent=Little_Cedar)
Eagle_Center = Member('Eagle Center', parent=Little_Cedar)
Carimona = Member('Carimona', parent=Decorah_Shale)
Hidden_Falls = Member('Hidden Falls', parent=Platteville)
Magnolia = Member('Magnolia', parent=Platteville)
Mifflin = Member('Mifflin', parent=Platteville)
Pecatonica = Member('Pecatonica', parent=Platteville)
Pigs_Eye = Member('Pigs Eye', parent=St_Peter_Sandstone)
Tonti = Member('Tonti', parent=St_Peter_Sandstone)
New_Richmond = Member('New Richmond', parent=Shakopee)
Willow_River = Member('Willow River', parent=Shakopee)
Coon_Valley = Member('Coon Valley', parent=Oneota)
Hager_City = Member('Hager City', parent=Oneota)
Black_Island = Member('Black Island', parent=Winnipeg)
Icebox = Member('Icebox', parent=Winnipeg)
Birkmose = Member('Birkmose', parent=Lone_Rock)
Reno = Member('Reno', parent=Lone_Rock)
Red_Clastics = Member('Red Clastics', parent=Mt_Simon_Sandstone)

MEMBER_LIST = [
    Blue_Hill,
    Codell_Sandstone,
    Fairport,
    Nishnabotna,
    Woodbury,
    Iron_Hill,
    Ostrander,
    Hinckle,
    Gizzard_Creek,
    Iowa_City,
    Bassett,
    Chickasaw_Shale,
    Eagle_Center,
    Carimona,
    Hidden_Falls,
    Magnolia,
    Mifflin,
    Pecatonica,
    Pigs_Eye,
    Tonti,
    New_Richmond,
    Willow_River,
    Coon_Valley,
    Hager_City,
    Black_Island,
    Icebox,
    Birkmose,
    Reno,
    Red_Clastics,
]
MEMBER_DICT = {member.name: member for member in MEMBER_LIST}

### BEDROCK UNIT CODES ###

BEDROCK_CODE_MAP = {
    'CAMB' : GeoCode(Cambrian),
    'DEVO' : GeoCode(Devonian),
    'KRET' : GeoCode(Cretaceous),
    'ORDO' : GeoCode(Ordovician),

    'KCBH' : GeoCode(Blue_Hill),
    'KCCD' : GeoCode(Codell_Sandstone),
    'KCFP' : GeoCode(Fairport),
    'KCRL' : GeoCode(Carlile_Shale),
    'KDKT' : GeoCode(Dakota_Sandstone),
    'KDNB' : GeoCode(Nishnabotna),
    'KDWB' : GeoCode(Woodbury),
    #'KGRN' : GeoCode(Greenhorn_Limestone),
    #'KGRS' : GeoCode(Ganeros_Shale),
    'KREG' : GeoCode(Cretaceous_Regolith),
    'KSRC' : GeoCode(Split_Rock),
    'KWIH' : GeoCode(Iron_Hill),
    'KWND' : GeoCode(Windrow),
    'KWOS' : GeoCode(Ostrander),

    'DCVU' : GeoCode(Upper_Cedar),
    'DLGH' : GeoCode(Lithograph_City),
    'DCRL' : GeoCode(Coralville),
    'DCUM' : GeoCode([Hinckle, Coralville]),
    'DCGZ' : GeoCode(Gizzard_Creek),
    'DCIC' : GeoCode(Iowa_City),
    'DCLC' : GeoCode([Coralville, Little_Cedar]),
    'DCVA' : GeoCode(Cedar_Valley),
    'DLBA' : GeoCode(Bassett),
    'DLCB' : GeoCode([Chickasaw_Shale, Bassett]),
    'DLCD' : GeoCode(Little_Cedar),
    'DLCH' : GeoCode(Chickasaw_Shale),
    'DLHE' : GeoCode([Eagle_Center, Hinckle]),
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
    'OGAP' : GeoCode([Galena, St_Peter_Sandstone]),
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
    'ODCA' : GeoCode(Carimona),
    'ODCR' : GeoCode(Decorah_Shale),
    'ODPG' : GeoCode([Decorah_Shale, Glenwood]),
    'ODPL' : GeoCode([Decorah_Shale, Platteville]),
    'ODSP' : GeoCode([Decorah_Shale, St_Peter_Sandstone]),
    'OPGW' : GeoCode([Platteville, Glenwood]),
    'OPHF' : GeoCode(Hidden_Falls),
    'OPMA' : GeoCode(Magnolia),
    'OPMI' : GeoCode(Mifflin),
    'OPPE' : GeoCode(Pecatonica),
    'OPSP' : GeoCode([Platteville, St_Peter_Sandstone]),
    'OPVJ' : GeoCode([Platteville, Jordan_Sandstone]),
    'OPVL' : GeoCode(Platteville),
    'OGSP' : GeoCode([Glenwood, St_Peter_Sandstone]),
    'OGWD' : GeoCode(Glenwood),
    'OSCJ' : GeoCode([St_Peter_Sandstone, Jordan_Sandstone]),
    'OSCS' : GeoCode([St_Peter_Sandstone, St_Lawrence]),
    'OSPC' : GeoCode([St_Peter_Sandstone, Prairie_Du_Chien]),
    'OSPE' : GeoCode(Pigs_Eye),
    'OSTN' : GeoCode(Tonti),
    'OSTP' : GeoCode(St_Peter_Sandstone),
    'OPCJ' : GeoCode([Prairie_Du_Chien, Jordan_Sandstone]),
    'OPCM' : GeoCode([Prairie_Du_Chien, Mt_Simon_Sandstone]),
    'OPCS' : GeoCode([Prairie_Du_Chien, St_Lawrence]),
    'OPCT' : GeoCode([Prairie_Du_Chien, Tunnel_City]),
    'OPDC' : GeoCode(Prairie_Du_Chien),
    'OPNR' : GeoCode(New_Richmond),
    'OPWR' : GeoCode(Willow_River),
    'OPSH' : GeoCode(Shakopee),
    'OOCV' : GeoCode(Coon_Valley),
    'OOHC' : GeoCode(Hager_City),
    'OPOD' : GeoCode(Oneota),
    'ORRV' : GeoCode(Red_River),
    #'OSTM' : GeoCode(Stoney_Mountain),
    'OWBI' : GeoCode(Black_Island),
    'OWIB' : GeoCode(Icebox),
    'OWIN' : GeoCode(Winnipeg),

    'CJDN' : GeoCode(Jordan_Sandstone),
    'CJEC' : GeoCode([Jordan_Sandstone, Eau_Claire]),
    'CJDW' : GeoCode([Jordan_Sandstone, Wonewoc_Sandstone]),
    'CJMS' : GeoCode([Jordan_Sandstone, Mt_Simon_Sandstone]),
    'CJSL' : GeoCode([Jordan_Sandstone, St_Lawrence]),
    'CJTC' : GeoCode([Jordan_Sandstone, Tunnel_City]),
    'CSLT' : GeoCode([St_Lawrence, Tunnel_City]),
    'CSLW' : GeoCode([St_Lawrence, Wonewoc_Sandstone]),
    'CSTL' : GeoCode(St_Lawrence),
    'CTCG' : GeoCode(Tunnel_City),
    'CTCM' : GeoCode([Tunnel_City, Mt_Simon_Sandstone]),
    'CTCW' : GeoCode([Tunnel_City, Wonewoc_Sandstone]),
    'CTCE' : GeoCode([Tunnel_City, Eau_Claire]),
    'CTMZ' : GeoCode(Mazomanie),
    'CTLR' : GeoCode(Lone_Rock),
    'CLBK' : GeoCode(Birkmose),
    'CLRE' : GeoCode(Reno),
    'CWMS' : GeoCode([Wonewoc_Sandstone, Mt_Simon_Sandstone]),
    'CWOC' : GeoCode(Wonewoc_Sandstone),
    'CWEC' : GeoCode([Wonewoc_Sandstone, Eau_Claire]),
    'CECR' : GeoCode(Eau_Claire),
    'CEMS' : GeoCode([Eau_Claire, Mt_Simon_Sandstone]),
    'CMFL' : GeoCode([Mt_Simon_Sandstone, Fond_Du_Lac]),
    'CMRC' : GeoCode(Red_Clastics),
    'CMSH' : GeoCode([Mt_Simon_Sandstone, Hinckley_Sandstone]),
    'CMTS' : GeoCode(Mt_Simon_Sandstone),
    'PMHN' : GeoCode(Hinckley_Sandstone),
    'PMHF' : GeoCode([Hinckley_Sandstone, Fond_Du_Lac]),
    'PMFL' : GeoCode(Fond_Du_Lac),
    'PMSC' : GeoCode(Solor_Church),
}