# -*- coding: utf-8 -*-


import os,json
import numpy as np

## torch is optional
try: 
    import torch
except:
    pass

import ase
from ase.io.formats import all_formats as ase_all_formats
from ase.io.formats import IOFormat

### Pymatgen CifParser is not necessary but can work much better than ASE
try:
    from pymatgen.io.cif import CifParser
    import warnings
    ### Ignore Pymatgen warnings because usually these just confuse users
    ### It is users responsibility to check that structure is how they expect
    warnings.filterwarnings("ignore")
except:
    pass

try:
    from CifFile import ReadCif
    read_cif_data = True
except:
    print("PyCifRW not installed. CIF property information will not be read.")
    read_cif_data = False

from mcse import Structure 
from mcse.io.check import format2extension,check_format,check_file_type,check_ext,mcse_ext

implemented_file_formats = mcse_ext

def read(struct_path, file_format='', 
         recursive=False, recursive_folder_names=[],
         verbose=False):
    """ 
    Read a single structure file or entire structure directory. The file format
    by default is automatically detected by using the file extension. 
    Alternatively, the user can specify a file format. 
    
    Arguments
    ---------
    struct_path: str
        Path to a structure file or a directory of structure files. 
    file_format: str
        File format to use. Can be any accepted ase file format. 
    recursive: bool
        Controls whether the entire folder tree will be explored to read in
        all structures files that are found. file_format will control which 
        file types are read in. For example, if file_format is json, then only
        json files will be read.
    recursive_folder_names: iterable
        If provided, only Structures within directories that match any in this 
        list will be read.
    
    Returns
    -------
    Structure or StructDict
        Returns a Structure object if the struct_path was pointed at a single 
        file.
        Returns a StructDict if the struct_path was pointed at a directory.
        
        
    """
    # If file format is used, check if it's an accepted value
    if len(file_format) > 0:
        check_format(file_format)
    
    if os.path.isdir(struct_path):
        if not recursive: 
            return read_dir(struct_path, file_format)
        else:
            return read_recursive(struct_path, file_format, 
                                  recursive_folder_names,
                                  verbose=verbose)
    elif os.path.isfile(struct_path):
        return read_file(struct_path, file_format)
    else:
        raise Exception("Input {} ".format(struct_path) +
                "was not recoginized as a file or a directory")


def read_dir(struct_dir, file_format=''):
    """
    Import any type of structure from directory to the structure class and 
    returns all structure objects as a dictionary, the best python data 
    structure due to hashing for O(1) lookup time.
    """
    file_list = os.listdir(struct_dir)
    struct_dict = {}
    for file_name in file_list:
        ## Hidden files should not be read
        if file_name[0] == ".":
            continue
        file_path = os.path.join(struct_dir,file_name)
        struct = read_file(file_path, file_format)
        if struct == None:
            continue
        struct_dict[struct.struct_id] = struct
            
    return struct_dict


def read_recursive(struct_dir, 
                   file_format="", 
                   recursive_folder_names=[],
                   verbose=False):
    """
    Looks through entire file structure below struct_dir to read in all 
    Structure files with the desired file format. 
    
    Arguments
    ---------
    struct_dir: str
        Path to the folder to search 
    file_format: str
        If a file_format is not specified, it will read in all files 
        identified to be structure files.
    
    """
    struct_dict = {}
    
    if len(file_format) > 0:
        desired_ext = format2extension[file_format]
    else:
        desired_ext = ""
    
    
    for root, dirnames, filenames in os.walk(struct_dir):
        ## No files in directory
        if len(filenames) == 0:
            continue
        for file_name in filenames:
            ## Hidden files should not be read
            if file_name[0] == ".":
                continue
            
            if len(recursive_folder_names) > 0:
                _,test_name = os.path.split(root)
                if test_name not in recursive_folder_names:
                    continue
            
            file_path = os.path.join(root, file_name)
            
            ## Check if the file should be read
            ext = check_ext(file_name)
            if len(desired_ext) > 0:
                if ext != desired_ext:
                    continue
                else:
                    struct = read_file(file_path, file_format)
                    if struct == None:
                        continue
                    struct.properties["Recurisve_Read_File"] = file_path
                    struct_dict[struct.struct_id] = struct
            ## Try to read in all files
            else:
                temp_file_format = check_file_type(file_path, verbose=verbose)
                if type(temp_file_format) == IOFormat:
                    temp_file_format = temp_file_format.name
                try:
                    struct = read_file(file_path, 
                                       temp_file_format, 
                                       verbose=verbose)
                except:
                    if verbose:
                        print("Could not load file: {}".format(file_path))
                    continue

                if struct == None:
                    if verbose:
                        print("Could not load file: {}".format(file_path))
                    continue
                
                ### Use file_path for struct_id in recursive to ensure that 
                ### each file found has a unique name
                struct.struct_id = "_".join(os.path.normpath(file_path).split(os.sep))
                struct.properties["Recurisve_Read_File"] = file_path
                struct_dict[struct.struct_id] = struct
                
    return struct_dict
            


def read_file(file_path, file_format='', verbose=False):
    """
    Imports a single file to a structure object.
    """
    if len(file_format) > 0:
        if file_format in ["geometry","geo", "aims"]:
            struct = import_ase(file_path)
        elif file_format == "json":
            struct = import_json(file_path)
        elif file_format == "cif":
            struct = import_cif(file_path)
        elif file_format == "ase":
            struct = import_ase(file_path)
        elif file_format == "torch":
            struct = import_torch(file_path)
        elif file_format == "xyz":
            struct = import_xyz(file_path)
        else:
            try: struct = import_ase(file_path)
            except:
                if verbose:
                    print("Could not load file {}. Check file_format argument."
                          .format(file_path))
                return None
            
    elif '.json' == file_path[-5:]:
            struct = import_json(file_path)
    elif '.cif' == file_path[-4:]:
        struct = import_cif(file_path)
    elif '.in' == file_path[-3:] or file_path.endswith('.next_step'):
        struct = import_ase(file_path, "aims")
    elif '.pt' == file_path[-3:]:
        struct = import_torch(file_path)
    elif '.xyz' == file_path[-4:]:
        struct = import_xyz(file_path)
    else:
        try: struct = import_ase(file_path)
        except: 
            if verbose: 
                print("Could not load file {}. Check file_format argument."
                    .format(file_path))
            return None
        
    return struct


def import_json(file_path):
    with open(file_path) as f:
        file_contents = f.read()
        try:
            file_dict = json.loads(file_contents)
        except json.decoder.JSONDecodeError as err:
            raise Exception("The following error was found for {}: {}".format(
                        file_path, err))
        if "struct_id" in file_dict:
            struct = Structure()
            struct.loads(file_contents)
            if struct.struct_id == None or len(struct.struct_id) == 0:
                file_name = os.path.basename(file_path)
                struct.struct_id = file_name.replace('.json','')
            return struct
        else:
            #### Assume combined 
            for struct_id,struct_dict in file_dict.items():
                temp_struct = Structure.from_dict(struct_dict)
                file_dict[struct_id] = temp_struct
            return file_dict
    
    # struct = Structure()
    # try:
    #     struct.build_struct_from_json_path(file_path)
    # except json.decoder.JSONDecodeError as err:
    #     raise Exception("The following error was found for {}: {}".format(
    #                 file_path, err))
        
    # if struct.struct_id == None or len(struct.struct_id) == 0:
    #     file_name = os.path.basename(file_path)
    #     struct.struct_id = file_name.replace('.json','')
    # return struct


def import_geo(file_path, struct_id=''):
    """
    Import single geometry file. 
    
    Arguments
    ---------
    file_path: str
        Path to the geometry file to be loaded
    struct_id: str
        Option to specify a struct_id. The default is an empty string which 
        specifies the behavior to use the file name as the struct_id. 
    """
    struct = Structure()
    struct.build_geo_from_atom_file(file_path)
    if len(struct_id) == 0:
        file_name = os.path.basename(file_path)
        struct.struct_id = file_name.replace('.in','')
    else:
        struct.struct_id = struct_id
    return struct


def import_cif(file_path,occupancy_tolerance=100):
    """
    Import cif with pymatgen. This is the current default.
    
    Aruments
    --------
    occupancy_tolerance: int
        Occupancy tolerance for pymatgen.io.cif.CifParser. Files from the 
        CSD can contain multiple atoms which sit on the same site when 
        PBC are taken into account. Pymatgen correctly recognizes these as the
        same atom, however, it has a tolerance for the number of atoms which it
        will assume are the same. This is the occupancy tolerance. I don't 
        believe this value should be limited and thus it's set at 100.
    """
    try:
        pycif = CifParser(file_path, occupancy_tolerance=occupancy_tolerance)
        pstruct_list = pycif.get_structures(primitive=False)
        struct = Structure.from_pymatgen(pstruct_list[0])
    except:
        struct = import_cif_ase(file_path)

    file_name = os.path.basename(file_path)
    struct.struct_id = file_name.replace('.cif','')
    
    #### Read in data that may be stored as a CIF property 
    if read_cif_data:
        ### Wrap in try except in case of PyCIFRW errors
        try: 
            cif = ReadCif(file_path)
        except:
            print("Error in reading CIF data for {}".format(file_path))
            return struct
        
        if len(cif) > 1:
            raise Exception("Must only have one structure per CIF file.")
        
        cif_struct = [x for x in cif.keys()]
        cif_data = cif[cif_struct[0]]
        for key,value in cif_data.items():
            ## Skip over geometry information
            if "_atom_" == key[0:len("_atom_")]:
                continue
            elif "_cell_" == key[0:len("_cell_")]:
                continue
            elif "_symmetry_equiv" == key[0:len("_symmetry_equiv")]:
                continue
            else:
                pass
            
            ## Otherwise add info to structure property information
            struct.properties[key] = value
            
    return struct


def import_cif_ase(file_path):
    """
    Import a single cif file using ase.io
    """
    atoms = ase.io.read(file_path,format='cif')#,primitive=False)
    struct = Structure.from_ase(atoms)
    file_name = os.path.basename(file_path)
    struct.struct_id = file_name.replace('.cif','')
    return struct


def import_ase(file_path, file_format=""):
    """
    Import general file using ase.io 
    """
    if len(file_format) == 0:
        atoms = ase.io.read(file_path)
    else:
        atoms = ase.io.read(file_path, format=file_format)
    struct = Structure.from_ase(atoms)
    file_name = os.path.basename(file_path)
    # Struct ID is filename before decimal 
    struct.struct_id = file_name.split('.')[0]
    return struct


def import_torch(file_path):
    """
    Import Structure.json file saved as a Pytorch file.
    
    """
    struct_ = torch.load(file_path)
    if "struct_id" in struct_: 
        struct = Structure.from_dict(struct_)
        return struct
    else:
        #### Assume it's combined file
        return struct_


def import_xyz(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    file_name = os.path.basename(file_path)
    init_id = file_name.split('.')[0]

    geo = []
    ele = []
    output = {}
    line_idx = 0
    struct_idx = 0
    while True:
        num_atoms = int(lines[line_idx])
        next_num_atoms_idx = line_idx + num_atoms + 2

        prop = lines[line_idx+1]

        atom_lines = lines[line_idx+2:line_idx+2+num_atoms]
        for temp_line in atom_lines:
            temp_split_line = temp_line.split()
            temp_ele = temp_split_line[0]
            if len(temp_ele) == 1:
                temp_ele = temp_ele.upper()
            elif len(temp_ele) == 2:
                temp_ele = temp_ele[0].upper() + temp_ele[1].lower()
            else:
                pass

            ele.append(temp_ele)
            geo.append([float(temp_split_line[1]),
                        float(temp_split_line[2]),
                        float(temp_split_line[3])])

        geo = np.vstack(geo)
        ele = np.hstack(ele)

        temp_id = "{}_{}".format(init_id, struct_idx)
        temp_struct = Structure(struct_id=temp_id, geometry=geo, elements=ele)
        output[temp_struct.struct_id] = temp_struct
        struct_idx += 1

        geo = []
        ele = []

        line_idx = next_num_atoms_idx
        if line_idx >= len(lines):
            break

    if len(output) == 1:
        return output[list(output.keys())[0]]
    else:
        return output



