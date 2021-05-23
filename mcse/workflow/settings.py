

import os,time,json
from importlib import import_module

import numpy as np

from mcse import Structure


def check_settings_file(obj, settings_path):
    """
    Check that the current settings of the object match the settings defined 
    in the settings_path file. 
    
    """
    ### Make sure this is safe in case multiple workflows are started at the 
    ### same time and all trying to write the settings file. 
    ### IMPLEMENT LATER IF NECESSARY
    # if not os.path.exists(settings_path):
    #     try: 
    #         with open("{}.lock".format(settings_path), "w") as lockfile:
    #             lockfile.write()
    #     except:
    current_settings = get_settings(obj)
        
    if not os.path.exists(settings_path):
        with open(settings_path, "w") as f:
            f.write(json.dumps(current_settings))   
        return 
    
    with open(settings_path, "r") as f:
        saved_settings = json.load(f)
    
    if current_settings != saved_settings:
        raise Exception("Current settings for {} does not match "
                    .format(type(obj).__name__)+
                    "the settings stored in {}"
                    .format(settings_path))
    else:
        return
        
        
# def check_settings(obj, settings):
#     """
#     Dictionary instead of file
    
#     """
    

            

def get_settings(obj):
    """
    Obtains a JSON file safe set of settings for the given object with respect
    to the mcse API. Note that this is not a general method that can be 
    used for all objects. 
    
    Does not work for objects without names such as integers or floats. This 
    is expected behavior because that is not the purpose of this function. 
    
    """
    if type(obj).__name__ == "function":
        obj_name = obj.__name__
    else:
        obj_name = type(obj).__name__
    
    json_safe_dict = {
        "Name": obj_name,
        "Module": obj.__module__,
        "Arguments": {},
        ### Decides if it's a function or object
        "Function": type(obj).__name__ == "function",
        ### This is a bool list for if the argument should be loaded and 
        ### possibly instantiated from the given arguments
        "Module_Arguments": []
        }
    
    ### Check if obejct is a driver or workflow. If this is the case, then
    ### we're only interested in arguments.
    if obj.__class__.__bases__[0].__name__ == "BaseDriver_" or \
        obj.__class__.__name__ == "Workflow":
        ### If driver, we're only interested in the settings used to initialize
        ### the driver
        interested_variables = obj.__init__.__code__.co_varnames
        obj_dict = {}
        for key,value in obj.__dict__.items():
            if key not in interested_variables:
                continue
            else:
                obj_dict[key] = value
            
    else:
        obj_dict = obj.__dict__
    
    ### Get the arguments for the object
    for key,value in obj_dict.items():
        module_argument = False
        
        if key == "comm":
            continue
        elif key == "size":
            continue
        elif key == "rank":
            continue
        # elif key == "drivers":
            # continue    
        
        ### Alternatively, don't need to skip driver list and can just get 
        ### settings from here automatically. 
        ### Specific implementation for Workflow class. 
        if key == "drivers":
            module_argument = True
            final_value = []
            for entry in value:
                final_value.append(get_settings(entry))
            value = final_value
        
        if type(value).__module__ == np.__name__:
            value = value.tolist()
        
        if type(value).__name__ == "function":
            module_argument = True
            ### Recursion is our friend here
            value = get_settings(value)
        
        elif value.__class__.__bases__[0].__name__ == "BaseDriver_":
            module_argument = True
            value = get_settings(value)
        
        if type(value) == Structure:
            value = value.document()
            
        json_safe_dict["Arguments"][key] = value
        json_safe_dict["Module_Arguments"].append(module_argument)
        
    
    return json_safe_dict


def from_settings_file(settings_file_path):
    with open(settings_file_path) as f:
        settings = json.load(f)
    return from_settings(settings)
    
            
def from_settings(settings):
    """
    Instantiate object from a settings file. 
    
    """
    if type(settings) == list:
        obj_list = []
        for entry in settings:
            ### Recursion
            obj_list.append(from_settings(entry))
        return obj_list
            
    obj = getattr(import_module(settings["Module"]), settings["Name"])
    
    ### If the settings were for a function, then importing it is all that 
    ### can and should possibly be done. Can now be returned. 
    if settings["Function"]:
        return obj
    
    arguments = settings["Arguments"]
    module_arguments = settings["Module_Arguments"]
    kwargs = {}
    for idx,key in enumerate(arguments):
        value = arguments[key]
        
        ### Check if this argument also needs to be imported
        temp_module_argument = module_arguments[idx]
        if temp_module_argument:
            ### Pog recursion, no need to write anything else
            value = from_settings(value)
        
        ### Check if value is a Structure object as a dictionary
        if type(value) == dict:
            if "struct_id" in value and "geometry" in value and "properties" in value:
                value = Structure.from_dict(value)
            
        kwargs[key] = value
    
    return obj(**kwargs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
        
    