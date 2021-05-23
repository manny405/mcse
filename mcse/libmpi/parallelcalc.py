# -*- coding: utf-8 -*-


import os,time
import numpy as np

from mcse.core.driver import BaseDriver_,Driver,PairwiseDriver
from mcse.io import read,write
from mcse.io.write import check_dir
from mcse.libmpi.base import _NaiveParallel,_JobParallel

from mpi4py import MPI


class ParallelCalc():
    def __init__(self, 
                 struct_path, 
                 output_path, 
                 driver, 
                 file_format="json",
                 overwrite=False,
                 use_class_write=True, 
                 comm=None, 
                 verbose=False):
        """
        Simple implementation that splits work evenly over all ranks.
        
        Arguments
        ---------
        struct_path: str
            Path to the directory of structures to calculate. The path is used
            to avoid loading a copy of the entire directory to every rank.
        driver: object 
            Class to calculate over every structure in the struct_path. Class
            must be initialized already.
        file_format: str
            File format to use to save the resulting file.
        use_class_write: bool
            If the class.write(output_path) should be used in-place of writing
            the modified structure object. Default is True. However, if the 
            code detects that the class doesn't have a class.write method, 
            it will behave as if the value was False. If False, the structure
            will be writen.
            
        """
        if comm == None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.verbose = verbose
        if self.verbose:
            print("(rank/size): ({}/{})".format(self.rank,self.size))
        
        self.struct_path = struct_path
        self.output_path = output_path
        self.file_format = file_format
        self.overwrite = overwrite
        self.driver = driver
        self.use_class_write = use_class_write
        self.naive = _NaiveParallel(comm=self.comm)
        self.jp = _JobParallel(comm=self.comm)
        if self.rank == 0:
            check_dir(self.output_path)
        comm.Barrier()
    
    
    def calc(self):
        """
        Wrapper for self.use_class_write
        
        """
        if self.use_class_write:
            write_method = getattr(self.driver, "write", None)
            if callable(write_method):
                calc_func = self._calc_write
            else:
                calc_func = self._calc
        else:
            calc_func = self._calc
        
        base_type = self.driver.__class__.__base__
        if base_type == BaseDriver_ or base_type == Driver:
            calc_func()
        elif base_type == PairwiseDriver:
            self._calc_pairwise()
    
    def _calc(self):
        my_files = self.naive.get_files(self.struct_path)
        total = len(my_files)
        for file_path in my_files:
            struct = read(file_path)
            self.driver.calc_struct(struct)
            
            temp_dict = {struct.struct_id: struct}
            write(self.output_path, temp_dict, 
                  file_format=self.file_format,
                  overwrite=self.overwrite)
            
            total -= 1
            if self.verbose:
                print("{}: {}".format(self.rank, total))
    
    
    def _calc_write(self):
        """
        Same as _calc but uses calls class.write(output_path) after calculation.
        
        """
        my_files = self.naive.get_files(self.struct_path)
        total = len(my_files)
        for file_path in my_files:
            struct = read(file_path)
            self.driver.calc_struct(struct)
            self.driver.write(self.output_path,
                                    file_format=self.file_format,
                                    overwrite=self.overwrite)
            
            total -= 1
            if self.verbose:
                print("{}: {}".format(self.rank, total))
                
    
    def _calc_pairwise(self):
        """
        Parallel implementation for a pairwise driver
        
        """
        if self.rank == 0:
            struct_dict = read(self.struct_path)
            pair_dict = self.driver._get_pair_dict(struct_dict)
            ### Can't pickle comm
            del(self.driver.comm)
            
            job_list = []
            for struct1_id,value in pair_dict.items():
                struct_dict[struct1_id].properties[self.__class__.__name__] = []
                for struct2_id in value:
                    job_list.append([self.driver.calc_struct, 
                        {"struct1": struct_dict[struct1_id],
                         "struct2": struct_dict[struct2_id]}])
            
            self.jp.job_list = job_list
        
        
        self.jp.send_jobs()
        all_results = self.jp.calc()
        
        if self.rank == 0:
            for inputs,result in all_results:
                struct_dict[inputs[1]["struct1"].struct_id].properties[
                        self.__class__.__name__].append(
                        (inputs[1]["struct2"].struct_id, result))
                struct_dict[inputs[1]["struct2"].struct_id].properties[
                        self.__class__.__name__].append(
                        (inputs[1]["struct1"].struct_id, result))
            
            write(self.output_path, 
                  struct_dict,
                  file_format=self.file_format,
                  overwrite=self.overwrite)
        
        

if __name__ == "__main__":
    pass

        
