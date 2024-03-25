"""NucML Constants."""
from enum import Enum, unique


@unique
class AMEDatasetURLs(str, Enum):
    """Pointers to the AME dataset URLs."""

    MASS = 'https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt'
    RCT1 = 'https://www-nds.iaea.org/amdc/ame2020/rct1.mas20.txt'
    RCT2 = 'https://www-nds.iaea.org/amdc/ame2020/rct2_1.mas20.txt'


EVALUATION_DATASET_URL = 'https://storage.googleapis.com/original_nuclear_data/evaluations.zip'

RIPL_DATASET_URL = "https://www-nds.iaea.org/RIPL-3/levels/levels.zip"

MAGIC_NUMBERS = [2, 8, 20, 28, 40, 50, 82, 126, 184]

EXFOR_MODES = ["neutrons", "protons", "alphas", "deuterons", "gammas", "helions"]
EXFOR_DATASET_URL = 'https://www-nds.iaea.org/x4toc4-master/C4-2021-11-10.zip'

EXFOR_MATERIALS = [
    'n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os,', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Ns', 'Hs', 'Mt', '??'
]

EXFOR_ISOTOPES = [
    '1H', '2H', '3H', '3He', '4He', '6Li', '7Li', '7Be', '8Li', '9Be', '10Be', '11B', '10B', '12C', '13C', '14C',
    '14N', '15N', '16O', '17O', '18O', '19F', '20Ne', '21Ne', '22Na', '22Ne', '23Na', '24Mg', '25Mg', '26Mg', '27Al',
    '27Mg', '26Al', '28Si', '29Si', '30Si', '31P', '31Si', '32Si', '32S', '33S', '34S', '35Cl', '36S', '36Cl', '37Cl',
    '40Ar', '36Ar', '37Ar', '38Ar', '39Ar', '39K', '41Ar', '40K', '41K', '40Ca', '41Ca', '42Ca', '43Ca', '44Ca',
    '45Ca', '45Sc', '46Ca', '48Ca', '46Sc', '48Ti', '44Ti', '46Ti', '47Ti', '49Ti', '50Ti', '51V', '50V', '52Cr',
    '50Cr', '51Cr', '53Cr', '54Cr', '53Mn', '54Mn', '55Mn', '56Fe', '54Fe', '55Fe', '57Fe', '57Co', '58Co', '58Fe',
    '59Co', '59Fe', '60Fe', '59Ni', '60Co', '58Ni', '60Ni', '61Ni', '62Ni', '63Ni', '64Ni', '64Cu', '65Ni', '63Cu',
    '65Cu', '64Zn', '65Zn', '66Cu', '66Zn', '67Zn', '68Zn', '70Zn', '70Ga', '69Ga', '71Ga', '73Ge', '68Ge', '70Ge',
    '72Ge', '74Ge', '75As', '76Ge', '79Se', '74Se', '76Se', '77Se', '78Se', '80Br', '80Se', '82Se', '76Br', '78Kr',
    '79Br', '79Kr', '80Kr', '81Br', '81Kr', '82Kr', '83Kr', '84Kr', '85Kr', '86Kr', '85Rb', '84Rb', '86Rb', '87Rb',
    '84Sr', '86Sr', '88Rb', '88Sr', '87Sr', '88Y', '89Sr', '89Y', '90Sr', '92Sr', '94Sr', '96Sr', '90Y', '91Y', '91Zr',
    '88Zr', '90Zr', '92Zr', '93Nb', '93Zr', '94Zr', '95Zr', '96Zr', '94Nb', '95Nb', '96Mo', '92Mo', '94Mo', '95Mo',
    '97Mo', '98Mo', '100Mo', '100Ru', '101Ru', '96Ru', '98Ru', '98Tc', '99Mo', '99Ru', '99Tc', '102Ru', '103Rh',
    '103Ru', '104Ru', '105Ru', '106Ru', '102Pd', '104Pd', '104Rh', '105Pd', '105Rh', '106Pd', '107Pd', '108Ag',
    '108Pd', '110Pd', '106Cd', '107Ag', '109Ag', '110Ag', '111Ag', '112Cd', '108Cd', '109Cd', '110Cd', '111Cd',
    '113Cd', '113In', '114Cd', '114In', '115Cd', '115In', '116Cd', '119Sn', '112Sn', '113Sn', '114Sn', '115Sn',
    '116Sn', '117Sn', '118Sn', '120Sn', '121Sn', '122Sn', '124Sn', '122Sb', '125Sn', '126Sn', '121Sb', '123Sb',
    '124Sb', '125Sb', '128Te', '120Te', '122Te', '123Te', '124Te', '125I', '125Te', '126I', '126Te', '127I', '130Te',
    '129I', '130I', '131I', '131Xe', '124Xe', '125Xe', '126Xe', '127Xe', '128Xe', '129Xe', '130Xe', '132Cs', '132Xe',
    '133Cs', '133Xe', '134Xe', '135Xe', '136Xe', '134Cs', '135Cs', '137Ba', '137Cs', '130Ba', '132Ba', '133Ba',
    '134Ba', '135Ba', '136Ba', '138Ba', '138La', '139Ba', '139La', '140Ba', '140Ce', '140La', '132Ce', '133Ce', '134Ce',
    '135Ce', '136Ce', '137Ce', '138Ce', '139Ce', '141Ce', '142Ce', '141Pr', '143Ce', '144Ce', '142Nd', '142Pr',
    '143Pr', '144Nd', '143Nd', '145Nd', '146Nd', '147Nd', '148Nd', '150Nd', '146Pm', '147Pm', '148Pm', '149Pm',
    '150Sm', '151Pm', '144Sm', '145Sm', '147Sm', '148Sm', '149Sm', '151Eu', '151Sm', '152Eu', '152Sm', '153Sm',
    '154Sm', '153Eu', '154Eu', '155Eu', '157Gd', '148Gd', '152Gd', '153Gd', '154Gd', '155Gd', '156Gd', '158Gd',
    '159Tb', '160Gd', '161Gd', '156Dy', '158Dy', '160Dy', '160Tb', '161Dy', '162Dy', '163Dy', '163Ho', '164Dy',
    '165Dy', '165Ho', '166Ho', '167Er', '162Er', '164Er', '166Er', '168Er', '169Er', '169Tm', '170Er', '171Er',
    '168Yb', '169Yb', '170Tm', '170Yb', '171Tm', '171Yb', '172Yb', '173Yb', '174Yb', '175Lu', '175Yb', '176Lu',
    '176Yb', '177Lu', '178Hf', '174Hf', '176Hf', '177Hf', '179Hf', '180Hf', '181Hf', '181Ta', '182Hf', '179Ta',
    '180Ta', '182Ta', '184W', '180W', '181W', '182W', '183W', '185W', '186W', '184Os', '184Re', '185Re', '186Os',
    '186Re', '187Os', '187Re', '187W', '188Os', '188Re', '188W', '189Os', '190Os', '191Ir', '191Os', '192Ir', '192Os',
    '193Ir', '193Os', '194Ir', '195Pt', '190Pt', '192Pt', '193Pt', '194Pt', '196Pt', '197Au', '198Pt', '199Pt',
    '198Au', '199Au', '201Hg', '194Hg', '196Hg', '198Hg', '199Hg', '200Hg', '202Hg', '203Hg', '204Hg', '204Tl',
    '203Tl', '205Tl', '207Pb', '204Pb', '205Pb', '206Pb', '208Pb', '209Bi', '210Pb', '210Bi', '210Po', '220Rn',
    '222Rn', '223Ra', '224Ra', '226Ra', '227Ac', '227Th', '228Ra', '228Th', '229Th', '230Th', '231Th', '232Th',
    '229Pa', '230Pa', '231Pa', '233Th', '234Th', '232Pa', '233Pa', '234Pa', '238U', '230U', '231U', '232U', '233U',
    '234U', '235U', '236U', '237U', '232Np', '233Np', '234Np', '235Np', '236Np', '237Np', '239U', '238Np', '236Pu',
    '237Pu', '238Pu', '239Np', '240Np', '239Pu', '240Pu', '241Pu', '242Pu', '243Pu', '244Pu', '238Am', '239Am',
    '240Am', '241Am', '245Pu', '242Am', '243Am', '240Cm', '241Cm', '242Cm', '243Cm', '244Am', '244Cm', '245Cm',
    '246Cm', '247Cm', '248Cm', '244Bk', '245Bk', '246Bk', '247Bk', '248Bk', '249Bk', '249Cm', '249Cf', '250Bk',
    '248Es', '249Es', '250Cf', '250Es', '251Cf', '252Cf', '253Cf', '253Es', '254Cf', '254Es', '255Es', '25Ne'
]
