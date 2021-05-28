
import os
import json
import h5py
import numpy as np

## torch is optional
try: 
    import torch
except:
    pass

import ase
from ase.io.formats import all_formats as ase_all_formats
from ase.data import vdw_radii,atomic_numbers,covalent_radii,chemical_symbols

from mcse import Structure
from mcse.io.check import mcse_ext,extension2format,format2extension

# Acceptable file formats listed here
ase_file_formats = [x for x in ase_all_formats.keys()]
mcse_file_formats = ['json','geometry', 'geo', 'aims', "MBD", "mbd"]


def write(path, struct_obj, file_format='json', combined=False, overwrite=False):
    """
    Wrapper function for writing structures. Interprets operation based on the 
    data type of the struct_obj arguement. 
    
    Arguments
    ---------
    path: str
        If struct_obj is a StructureDictionary, path is the directory to output
        all the structures.
        If struct_obj is a Structure, path is a file name
    struct_obj: Structure, StructureDictionary, dict
        Any of these objects are acceptable inputs. 
    file_format: str or list of str
        Any of the mcse_file_formats or ase_file_formats are acceptable
    overwrite: bool
        True: Structure files will be overwritten in the dir_path
        False: Exception will be raised if file already exists
        
    """
    if type(path) != str:
        raise Exception("First arugment is a file path and must be a string." +
            "User Input argument was of type {}.".format(type(path)))
    
    if type(struct_obj) == Structure:
        check_parent_dir(path)
        if type(file_format) == str:
            wrapper_write_struct(path, struct_obj, file_format=file_format, 
                                 overwrite=overwrite)

    else:
        if type(file_format) == str:
            output_struct_dict(path, struct_obj, file_format=file_format,
                               combined=combined,
                               overwrite=overwrite)
        else:
            output_struct_dict_multi_format(path, struct_obj, 
                                            file_format_list=file_format,
                                            combined=combined,
                                            overwrite=overwrite)
                           

def output_struct_dict(dir_path, 
                       struct_dict, 
                       file_format='json', 
                       combined=False,
                       overwrite=False):
    """
    Writes a file for each structure in the StructureDictionary to the 
    specified directory path.
    
    Arguments
    ---------
    file_format: str
        Can be any of the support file formats or an ase file format
    overwite: bool
        True: Structure files will be overwritten in the dir_path
        False: Exception will be raised if file already exists
        
    """
    
    if type(dir_path) != str:
        raise Exception('Directory path was not a string. '+
                   'Check that arguments are of the form: '+
                   'write(dir_path, struct_dict).')
    
    
    if combined == True:
        if file_format == "cif":
            raise Exception("Not Implemented")
        elif file_format == "json":
            check_overwrite(dir_path, overwrite=overwrite)
            write_dict = {}
            for struct_id,struct in struct_dict.items():
                write_dict[struct_id] = struct.document()
            with open(dir_path, "w") as f:
                f.write(json.dumps(write_dict, indent=4))
        elif file_format == "torch":
            check_overwrite(dir_path, overwrite=overwrite)
            torch.save(struct_dict, dir_path)
        else:
            raise Exception("Combined not implemented for {}"
                            .format(file_format))
        return
    
    check_dir(dir_path)
    
    for struct_id,struct in struct_dict.items():
        # The correct file extension for the file path will be added 
        # in the specific function responsible for outputing the file
        file_path = os.path.join(dir_path,struct_id)
        
        # Call wrapper function for each structure
        wrapper_write_struct(file_path, struct, file_format=file_format,
                             overwrite=overwrite)


def output_struct_dict_multi_format(dir_path, struct_dict, 
                                    file_format_list=["json", "geo"],
                                    combined=False,
                                    overwrite=False):
    """
    Will output structures as multiple file formats into separate folders for
    each file format. 
    
    """
    check_dir(dir_path)
    for file_format in file_format_list:
        format_path = os.path.join(dir_path, file_format)
        check_dir(format_path)
    
    for struct_id,struct in struct_dict.items() :
        for file_format in file_format_list:
            struct_path = os.path.join(dir_path, file_format)
            struct_path = os.path.join(struct_path, struct_id)
            wrapper_write_struct(struct_path, struct, 
                                 file_format=file_format,
                                 combined=combined,
                                 overwrite=overwrite)
        
    
        
def wrapper_write_struct(path, struct, file_format='json', overwrite=False):
    """
    Wrapper function for directing the output of a single structure to the 
    correct function. 
    """
    
    if file_format == 'json':
        output_struct(path, struct, overwrite=overwrite)
        
    elif file_format == 'geometry' or \
         file_format == 'geo' or \
         file_format == 'aims':
        output_ase(path, struct, file_format="aims", overwrite=overwrite)
    elif file_format == "torch":
        output_torch(path, struct, overwrite=overwrite)
    else:
        try: ase_all_formats[file_format]
        except: 
            raise Exception("Unsupported file format. "+
              "Please try one of mcse file formats: {} \n"
              .format(mcse_file_formats) +
              "or one of ASE's supported file formats: {}"
              .format(ase_file_formats))
        # Outputs using ase if file_format is acceptable ase format
        else:    
            output_ase(path, struct, file_format, overwrite=overwrite)
            

def output_struct(file_path, struct, overwrite=False):
    """
    Outputing Structure as a json file. Function automatically replaces any 
    other file extension in the input argument with .json
    """
    struct.properties["mcse"] = "Structure file was written using mcse."
    # Fix file path with json extension
    file_path = file_ext(file_path, "json")
    # Checking overwrite
    check_overwrite(file_path, overwrite=overwrite)
    with open(file_path,'w') as f:
        f.write(struct.dumps())
    

def output_ase(file_path, struct, file_format, overwrite=False):
    # Fix file path with file_format as the extension. 
    # I'm not sure ASE provides a list of the proper file extensions so this 
    # is the way to do it for now.
    file_path = file_ext(file_path, file_format)
    # Checking overwrite
    check_overwrite(file_path, overwrite=overwrite)
    # Convert to atoms object
    atoms = struct.get_ase_atoms()
    ase.io.write(file_path, atoms, format=file_format)
    

def output_torch(file_path, struct, overwrite=False):
    """
    Output Structure.json file using torch.
    
    """
    file_path = file_ext(file_path, "pt")
    check_overwrite(file_path, overwrite=overwrite)
    ## Using Pymongo format for saving structure as dictionary
    struct_ = struct.document()
    torch.save(struct_, file_path)
    

def file_ext(file_path, file_ext):
    """
    Function replaces any other file extension in the input argument with 
    the file_ext argument
    
    Arguments
    ---------
    file_path: str
        Directory and file name for the file to be output
    file_ext: str
        File extension to be added to the structure. File extension should NOT
        include a '.'
        
    """
    # Split file path
    path,file_name = os.path.split(file_path)

    ## Check if current file extension is known by mcse
    temp_file_ext = file_name.split(".")[-1]
    if temp_file_ext in mcse_ext:
        ## Remove file extension if known
        file_name = file_name.replace(".{}".format(temp_file_ext),
                                      "")
    else:
        ## Otherwise, don't know how to handle file ext
        ## so do nothing
        pass
    
    ### Check that the given file_ext is formatted properly
    if file_ext not in extension2format:
        if file_ext in format2extension:
            file_ext = format2extension[file_ext]
        
    # Add specified file extension
    file_name = file_name + "."+ file_ext
    # Recombine
    file_path = os.path.join(path,file_name)

    return file_path


def check_parent_dir(file_path):
    """
    Checks if directory of input file_path already exists. If not, creates it.
    If it exists but it is not a directory, raises exception.
    """
    base,filename = os.path.split(file_path)
    if len(base) == 0:
        # No base path supplied, proceed in current directory
        return
    check_dir(base)


def check_dir(base):
    """
    Call to check if directory already exists. If it doesn't exist, then the
    directory will be made.
    """
    if os.path.exists(base):
        if os.path.isdir(base):
            # Proceed as expected
            return
        else:
            raise Exception("Intended parent directory {}".format(base) +
                    "is not a directory. "+
                    "Please check that the path {}".format(base) +
                    "is correct.")
    else:
        os.makedirs(base)
        return


def check_overwrite(file_path, overwrite=False):
    """
    Defines the behavior for the overwrite argument for all output types.
    
    Arguments
    ---------
    file_path: str
        Exact file path
    overwrite: bool
        True: Structure files will be overwritten in the dir_path
        False: Exception will be raised if file already exists
    """
    if os.path.exists(file_path):
        if overwrite == True:
            return
        else:
            raise Exception('Filepath {} already exists. '
                            'If you want to overwite, use overwrite=True'.
                            format(file_path))
