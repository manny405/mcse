# -*- coding: utf-8 -*-


"""
Basic checking functions are located here. This is to perform folder management
during io. Note that these funcitons used to exist in mcse.io.write, however
they are now being moved here, to a more general location.
"""


import os
from ase.io.formats import extension2format as ase_extension2format
from ase.io.formats import filetype


## All possible extensions allowed in mcse
mcse_ext = ["in", "json","next_step","cif", "pt"]
mcse_ext += [x for x in ase_extension2format.keys()]


## Convert all acceptable extensions to their respective format
extension2format = {}
for key,value in ase_extension2format.items():
    extension2format[key] = value.name
extension2format["pt"] = "torch"
extension2format["cif"] = "cif"
extension2format["next_step"] = "aims"
extension2format["json"] = "json"
extension2format["xyz"] = "xyz"

## Convert all formats into the valid extension
format2extension = {}
format2extension.update([x for x in zip(extension2format.values(), 
                                        extension2format.keys())])

format2extension["aims"] = "in"
format2extension["geo"] = "in"
format2extension["geometry"] = "in"
    

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


def check_dir(base, comm=None):
    """
    Call to check if directory already exists. If it doesn't exist, then the
    directory will be made.
    
    """
    # if comm == None:
    #     comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    
    # if rank == 0:
    if os.path.exists(base):
        if os.path.isdir(base):
            # Proceed as expected
            pass
        else:
            raise Exception("Intended parent directory {}".format(base) +
                    "is not a directory")
    else:
        os.makedirs(base)
        pass

        ### Make sure that file system actually
        ### responds correctly before returning
        idx = 0
        while True:
            if os.path.isdir(base):
                return
            idx += 1
            if idx > 1e6:
                raise Exception("File system error when creating directory {}"
                           .format(base))
    
    # comm.Barrier()


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
    

def check_format(file_format):
    """
    Check if file_format is accepted by mcse
    
    """
    if file_format in format2extension:
        return True
    else:
        raise Exception("File format {} ".format(file_format) + 
                "is not accepted by mcse.")
    

def check_ext(file_name, verbose=False):
    """
    Checks if the given file_name has an extension that is compatible with 
    mcse. 
        
    """
    # Split file path if applicable
    path,file_name = os.path.split(file_name)
    
    # Use dot to split for file extension
    split_file_name = file_name.split(".")
    
    ## Check for presence of file extension
    if len(split_file_name) == 1:
        if verbose:
            print("No file extension found for {}".format(file_name))
        return ""
    
    ## Check if the filename contains more than a single dot and is not a
    ## geometry.in.next_step
    if len(split_file_name) > 2 and \
       split_file_name[-1] != "next_step":
           if verbose:
               print("File name {} may have ambigious placement of "
                     .format(file_name) +
                     "file extension. "+
                     "String after the last decimal will be used as the "+
                     "file extension")
           
    file_ext = split_file_name[-1]
    
    return file_ext


def check_file_type(file_path, verbose=True):
    """
    Returns file type if one is identified by either mcse or ASE.
    
    Returns
    -------
    str:
        Returns either an empty string if no file format was found or the 
        correctly identified file format for the given file path.
    
    """
    if not os.path.isfile(file_path):
        if verbose:
            print("{} is not a file".format(file_path))
        return ""
    
    ext = check_ext(file_path)
    
    if len(ext) == 0:
        try: 
            file_format = filetype(file_path)
        except:
            if verbose:
                print("ASE could not find filetype for {}"
                              .format(file_path))
            return ""
        
    if ext not in mcse_ext:
        if verbose:
            print("File extension \".{}\" was not recognized by mcse."
                   .format(ext))
        return ""
    else:
        file_format = extension2format[ext]
    
    return file_format
    

            
def check_struct_dir(path, verbose=True):
    """
    Checks if the directory is a valid directory with structures inside. 
    Otherwise, an error will be raised. A valid directory is defined as a 
    directory that already exists and either:
        - Has at least one structure file with a valid file extension 
        - Has at least one structure that can be read with ASE.io.read
    Returns True if either of these conditions are met. Returns False otherwise.
    
    
    Arguments
    ---------
    path: str
        Path to the directory to check. 
        
    Returns
    -------
    bool
        
    """
    if not os.path.exists(path):
        raise Exception("Directory {} does not exist.".format(path))
    
    ## Create file iterator because we may not wish to generate the entire
    ## file list    
    file_iter = os.scandir(path)
    
    for file_name in file_iter:
        file_ext = check_ext(file_name, verbose=verbose)
        
        ## Directory has at least one valid structure file.
        if file_ext in mcse_ext:
            return True
    
    ## Restart iterator
    file_iter = os.scandir(path)
    
    ## Now check if ASE can guess file format
    for file_name in file_iter:
        file_path = os.path.join(path, file_name)
        try: file_format = filetype(file_path)
        except:
            if verbose:
                print("ASE could not find filetype for {}"
                              .format(file_path))
            continue
        
        ## If we get this far, then format 
        if verbose:
            print("ASE identified {} as having file format {}"
                  .format(file_path, file_format))
        return True
    
    return False
    
    
    
    
    
    
    
    
    
    
