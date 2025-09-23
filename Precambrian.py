from Bedrock import GeoCode


class Precambrian:

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
Unknown = Age('Unknown')

AGE_LIST = [
    Archean,
    Early_Proterozoic,
    Mesoproterozoic,
    Unknown,
]

### PRECAMBRIAN LITHOLOGY ###
Felsic_Intrusive = Lithology('Felsic Intrusive')
Intermediate_Intrusive = Lithology('Intermediate Intrusive')
Mafic_Intrusive = Lithology('Mafic Intrusive')
Ultramafic_Intrusive = Lithology('Ultramafic Intrusive')
Slate = Lithology('Slate')
Argillite = Lithology('Argillite')
Phyllite = Lithology('Phyllite')
Schist = Lithology('Schist')
Gneiss = Lithology('Gneiss')
Marble = Lithology('Marble')
Quartzite = Lithology('Quartzite')
Metapelite = Lithology('Metapelite')
Metapsammite = Lithology('Metapsammite')
Felsic_Gneiss = Lithology('Felsic Gneiss')
Intermediate_Gneiss = Lithology('Intermediate Gneiss')
Mafic_Gneiss = Lithology('Mafic Gneiss')
Felsic_Schist = Lithology('Felsic Schist')
Intermediate_Schist = Lithology('Intermediate Schist')
Mafic_Schist = Lithology('Mafic Schist')
Calcium_Silicate_Schist = Lithology('Calcium Silicate Schist')
Amphibolite = Lithology('Amphibolite')
Fault_Breccia = Lithology('Fault Breccia')
Fault_Gouge = Lithology('Fault Gouge')
Brittle_Fault = Lithology('Brittle Fault')
Protomylonite = Lithology('Protomylonite')
Mylonite = Lithology('Mylonite')
Ultramylonite = Lithology('Ultramylonite')
Metasediment = Lithology('Metasediment')
Quartz_Vein = Lithology('Quartz Vein')
Pegmatite = Lithology('Pegmatite')
Diabase = Lithology('Diabase')
Aplite = Lithology('Aplite')
Ultramafic_Volcanic = Lithology('Ultramafic Volcanic')
Mafic_Volcanic = Lithology('Mafic Volcanic')
Intermediate_Volcanic = Lithology('Intermediate Volcanic')
Felsic_Volcanic = Lithology('Felsic Volcanic')
Mafic_To_Intermediate = Lithology('Mafic To Intermediate')
Intermediate_To_Felsic = Lithology('Intermediate To Felsic')
Conglomerate = Lithology('Conglomerate')
Carbonate = Lithology('Carbonate')
Sandstone = Lithology('Sandstone')
Siltstone = Lithology('Siltstone')
Shale = Lithology('Shale')
Chert = Lithology('Chert')
Iron_Formation = Lithology('Iron Formation')

### NEW PRECAMBRIAN CODES ###

PRECAMBRIAN_CODE_MAP = {
    'PAFI' : GeoCode([Felsic_Intrusive, Archean]),
    'PEFI' : GeoCode([Felsic_Intrusive, Early_Proterozoic]),
    'PMFI' : GeoCode([Felsic_Intrusive, Mesoproterozoic]),
    'PUFI' : GeoCode([Felsic_Intrusive, Unknown]),
    'PAII' : GeoCode([Intermediate_Intrusive, Archean]),
    'PEII' : GeoCode([Intermediate_Intrusive, Early_Proterozoic]),
    'PMII' : GeoCode([Intermediate_Intrusive, Mesoproterozoic]),
    'PUII' : GeoCode([Intermediate_Intrusive, Unknown]),
    'PAIM' : GeoCode([Mafic_Intrusive, Archean]),
    'PEIM' : GeoCode([Mafic_Intrusive, Early_Proterozoic]),
    'PMIM' : GeoCode([Mafic_Intrusive, Mesoproterozoic]),
    'PUIM' : GeoCode([Mafic_Intrusive, Unknown]),
}