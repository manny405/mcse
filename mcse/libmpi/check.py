# -*- coding: utf-8 -*-

import os,shutil

from mcse.io import read,write
from mcse.io import check_dir,check_struct_dir
from mcse.libmpi.base import _NaiveParallel

from mpi4py import MPI


def null_check_fn(struct_path):
    """
    Checking Structure from input directory will always return False.

    """
    return False


def check_properties(struct_path, key="RSF"):
    """
    Basic check function. Check functions take a path as the first argument
    followed by any number of keyword arguments that may change the behavior
    of the check function. The function will return True or False. This 
    will indicate whether the structure needs to be calculated or not. 
    
    """
    struct = read(struct_path)
    if struct.properties.get(key):
        return True
    else:
        return False   


class Check(_NaiveParallel):
    """
    SUMMARY: - USE FOR RESTARTABLE MPI PARALLELIZED ANALYSIS OF STRUCTURES. 
             - USE TO PERFORM PARALLEL IO, DECREASING THE TIME TO CHECK
               IF CALCULATIONS HAVE COMPLETED FOR LARGE NUMBERS OF CALCULATIONS. 
    
    Results for simple test of parallel io scaling for setting up calculations 
    for 50 structures:
       Cores    Time x100 Iterations
        1           11.13
        2           6.367
        3           4.982
        4           3.987
        5           3.664
        6           3.704
    
    cores = [1,2,3,4,5,6]
    time = [11.13, 6.367, 4.982, 3.987, 3.664, 3.704]
    optimal = [11.13, 5.565, 3.71, 2.7825, 2.226, 1.855]
    
    Results for simple test of parallel io scaling for setting up calculations 
    for 5000 structures:
       Cores    Time x5 Iterations
        1           66.714
        2           43.423
        3           26.495
        4           22.245
        5           19.183
        6           18.175
    
    cores = [1,2,3,4,5,6]
    time = [66.714, 43.423, 26.495, 22.245, 19.183, 18.175]
    optimal = [66.6714, 33.3357, 22.2238, 16.66785, 13.33428, 11.1119]
    
    Large, parallel analysis may get interrupted unexpectedly. The advantage 
    of using this function is that it will enable the ability to check 
    in parallel what structures have been calculated and what still needs to
    be done if the analysis is very large. The function may also be used to 
    check if larger calculations, such as FHI-aims calculations, 
    have been completed or need to be restarted. 
    
    Basic operation is as follows:
        - There is a structure directory that you would like to perform a
          calculation on. 
        - The calculation will be performed in a temporary calculation 
          directory. 
        - The results from the calculation will be written to an output 
          directory. This directory can be the same as the input structure
          directory by overwriting the original files with the new information.
    
    Arguments
    ---------
    struct_dir: str
        Path to the directory of structures to use. 
    output_dir: str
        Path to the directory to place finished structures
    file_format: str
        File format for writing resulting structures. 
    calc_dir: str
        Path to the directory where calculations should take place.
    check_fn: callable(path, **check_fn_kw)
        Callable function that takes in the path as the first argument and 
        then any keyword arguments the user would like to provide. 
    check_fn_kw: dict
        Dictionary of the keyword functions to pass to check_fn
    calculator: callable(Structure, **calculator_kw)
        If the user would like to use the check function to also initialize
        or perform a calculation, a calculator can be set. 
    calculator_kw: dict
        If keywork arguments to the dictionary need to be passed in at 
        runtime, this can be done with this argument. 
        
    TODO: CHANGE THE CHECK_FN TO JUST A CLASS THAT TAKES IN THE STRUCTURE AND
          RETURNS THE TRUE OR FALSE. THIS IS MUCH MORE FLEXIBLE. 
        
    """
    def __init__(self, 
                 struct_dir,
                 output_dir="",
                 file_format="json",
                 calc_dir = "temp",
                 check_fn=check_properties,
                 check_fn_kw={"key": "RSF"},
                 calculator=None,
                 calculator_kw={},
                 comm=None,
                 ):
        if comm == None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.struct_dir = struct_dir
        ## Make sure that struct_dir is a proper structure directory
        if not check_struct_dir(self.struct_dir):
            raise Exception("Input {} was not recogized ".format(struct_dir)+
                    "as a valid structure directory.")
        
        if len(output_dir) == 0:
            self.output_dir = self.struct_dir
        else:
            self.output_dir = output_dir
        
        self.file_format = file_format
        self.calc_dir = calc_dir
        self.check_fn = check_fn
        self.check_fn_kw = check_fn_kw
        self.calculator = calculator
        self.calculator_kw = calculator_kw
        
        ## Setup calc and output dir if they don't already exist
        if self.comm.rank == 0:
            check_dir(self.calc_dir)
            check_dir(self.output_dir)
        self.comm.Barrier()
        
        ## Storage
        self.my_files = []
        
    
    
    def calc(self):
        """
        Main function that manages checking of the struct_dir and calc_dir
        
        """
        ## Start by building list of files that this rank needs to calculate
        self.my_files = self.get_files(self.struct_dir)
        self.construct_filename_dict()
        
        
        ## Check to see what's already in the output dir
        self.check_dir(self.output_dir, move=False)
        ## Reconstruct filename_dict with modified my_files
        self.construct_filename_dict()
        
        ## Check if anything is finished from calc_dir that should be moved
        ## to the output dir. Otherwise, since it is already in the calc
        ## directory, then should be removed from self.my_files as to not
        ## overwrite anything in the calc_dir
        self.check_dir(self.calc_dir, delete=True, move=True)
        ## Reconstruct filename_dict with modified my_files
        self.construct_filename_dict()
        
        ## Now setup files for calculation in the calc_dir
        self.setup_calc()

        self.comm.Barrier()
        
        ## And calculate if calculator was attached
        self.calculate()
        
    
    def construct_filename_dict(self):
        ## Create dict for fast checking and indexing purposes
        my_file_names = []
        for file in self.my_files:
            s = read(file)
            my_file_names.append(s.struct_id)
            
        self.my_filename_dict = {}
        self.my_filename_dict.update(
                zip(my_file_names, 
                    [idx for idx in range(len(my_file_names))]))
        
        
    
    def calculate(self):
        """
        Simple calculate function using mcse style calculators. In addition,
        the calculator could be as simple as creating a Slurm submission script
        and submitting it. 
        
        """
        if self.calculator != None:
            ## Get files for calc_dir
            self.my_files = self.get_files(self.calc_dir)
            
            for calc_dir in self.my_files:
                s = read(calc_dir)
                if len(s) == 1:
                    struct = [x for x in s.values()][0]
                    self.calculator.calc(struct)
                    self.calculator.write(self.output_dir, 
                                          file_format=self.file_format, 
                                          overwrite=True)
                else:
                    self.calculator.calc(s)
                    self.calculator.write(self.output_dir, 
                                          file_format=self.file_format, 
                                          overwrite=True)
                
                shutil.rmtree(calc_dir)
    
    
    def check_dir(self, path, delete=False, move=False):
        """
        
        
        Argumnets
        ---------
        path: str
            Path to the direcrtory to check
        delete: str
            If one of my_files is even found, then delete it from self.my_files. 
            This is necessary when checking the temp directory. 
        move: bool
            If the structure is identified as being finished, then move to 
            the output directory.
        
        """
        ## Get iterator
        path_file_iter = os.scandir(path)
        
        ## No need to check 
        if len(self.my_files) == 0:
            return
        
        del_list = []
        for file_direntry in path_file_iter:
            file_name = file_direntry.name
            file_path = os.path.join(path, file_name)
            
            ## If it's a file_path, assume it's a structure file
            if os.path.isfile(file_path):
                s = read(file_path)
                file_name = s.struct_id
            ## Otherwise it's a directory already named as a struct_id 
            ## So set file_path equal to path of structure file inside the 
            ## directory path
            else:
                file_path = os.path.join(file_path, file_name+".json")
            
            ## Check if this rank is responsible for this file
            if self.my_filename_dict.get(file_name) != None:
                ## Get idx w.r.t. self.my_files
                my_files_idx = self.my_filename_dict[file_name]
                
                if self.check_fn(file_path, **self.check_fn_kw):
                    ## Always delete from my_files if it passes
                    del_list.append(my_files_idx)
                    
                    ## Move to output dir if checking calc_dir
                    if move:
                        calc_dir = os.path.join(path, file_name)
                        
                        ## Write structure file to output directory
                        s = read(file_path)
                        temp_dict = {s.struct_id: s}
                        write(self.output_dir, 
                              temp_dict, 
                              file_format=self.file_format)
                        shutil.rmtree(calc_dir)
                
                else:
                    if delete:
                        del_list.append(my_files_idx)
                    
                    ## Otherwise, file should stay in self.my_files because
                    ## it needs to be calculated
                    pass
        
        ## Delete all indicies that have been collected in reverse order so 
        ## that index to be deleted is always correct
        for idx in sorted(del_list, reverse=True):
            del(self.my_files[idx])
                
                
    def setup_calc(self):
        """
        Sets up files for calculation in the calculation directory. In the 
        calculation directory, there will be a directory for each structure 
        that needs to be calculated. Inside will be the Structure.json file
        for each structure.
        
        """
        for file in self.my_files:
            s = read(file)
            temp_dict = {s.struct_id: s}
            output = os.path.join(self.calc_dir, s.struct_id)
            write(output, temp_dict, file_format=self.file_format)
                        
