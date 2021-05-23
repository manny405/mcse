


"""
Scripts for performing operations on StructDict
"""

import numpy as np
import pandas as pd

get_implemented = ["property", "prop"]

def get(struct_dict, dtype="property", options=["energy"],
        astype=""):
    """
    Gets data from a dictionary of structures. 
    Returns the data as a pandas dataframe. 

    Arguments
    ---------
    dtype: str
        Determines what data type to get from the StructDict. 
        Currently implemented options are: "property"
        To be implemented: "geo"
    options: dict
        Options to pass into the dtype function. Behavior is determined
        by the specific dtype.
    astype: float
        
          
    """
    if dtype not in get_implemented:
        raise Exception("Getting dtype {} from StructDcit is not implemented"
                        .format(dtype))
    if dtype == "property" or dtype == "prop":
        dtype = "prop"

    get_func = eval("get_{}".format(dtype))        

    return get_func(struct_dict, options)


def get_prop(struct_dict, options=["energy"]):
    """
    Returns the property of the structure for each entry in options. 
    
    Arguments
    ---------
    options: str or list
        Options can be a list of proeprties to obtain or a comma 
        separated string of properties.
    """
    if type(options) == "str":
        options = options.split(",")
    
    result = pd.DataFrame(columns=options,
                          index=[x for x in struct_dict.keys()])
   
    for struct_id,struct in struct_dict.items():
        for option in options:
            if option == "lattice_vectors":
                result[option][struct_id] = np.vstack(struct.get_lattice_vectors())
            elif option == "lattice_norms" or option == "norm":
                result[option][struct_id] = np.linalg.norm(
                        np.vstack(struct.get_lattice_vectors()), axis=-1)
            else:
                result[option][struct_id] = struct.get_property(option)
                

    return result


if __name__ == "__main__":
    pass
    
