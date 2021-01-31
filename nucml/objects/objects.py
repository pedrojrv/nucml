import pickle
import os
import sys

sys.path.append("..")
sys.path.append("../..")

from nucml import general_utilities  # pylint: disable=import-error

dirname = os.path.dirname(__file__)

def load_zan(type="EXFOR"):
    """Loads a python dictionary containing proton, mass number, and neutron number mapping 
    to isotopes in either EXFOR or AME format (i.e. {'912':'C12'}).

    If type is EXFOR the isotope format is ELAAA (i.e. C12). If AME the format is AAAEL (i.e. 12C).

    Args:
        type (str, optional): Dictionary values format. Can be EXFOR or AME. Defaults to "EXFOR".

    Returns:
        dict
    """    
    if type.upper() == "EXFOR":
        filename = os.path.join(dirname, 'key_ZAN_EXFOR_value_el_dict.pkl')
    elif type.upper() == "AME":
        filename = os.path.join(dirname, 'key_ZAN_AME_value_el_dict.pkl')
    zan_dict = general_utilities.load_obj(filename)
    return zan_dict

def get_element(zan_dict, Z, A, N):
    """Gets the element identifier given the number of protons, mass number, and neutrons using
    the dictionary obtained from load_zan().

    Args:
        zan_dict (dict): Dictionary obtained from the load_zan() function.
        Z (int): Number of protons.
        A (int): Mass number.
        N (int): Number of neutrons.

    Returns:
        str: Element tag.
    """    
    ZAN = str(Z) + str(A) + str(N)
    if ZAN == "12019":
        return "Water"
    else:
        return zan_dict[ZAN]["Element"]

def get_isotope(zan_dict, Z, A, N):
    """Gets the isotope identifier given the number of protons, mass number, and neutrons using
    the dictionary obtained from load_zan().

    Args:
        zan_dict (dict): Dictionary obtained from the load_zan() function.
        Z (int): Number of protons.
        A (int): Mass number.
        N (int): Number of neutrons.

    Returns:
        str: Element tag.
    """    
    ZAN = str(Z) + str(A) + str(N)
    if ZAN == "12019":
        return "Water"
    else:
        return zan_dict[ZAN]["Isotope"]
