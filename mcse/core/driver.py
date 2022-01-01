# -*- coding: utf-8 -*-

import numpy as np

from mcse.core.structure import Structure
from mcse.core.struct_dict import StructDict
from mcse.io import read,write

from mcse.workflow.settings import get_settings,from_settings

try: 
    from mpi4py import MPI
    use_mpi = True
except:
    use_mpi = False


class BaseDriver_():
    """
    Base mcse Driver class that defines the API for all mcse Drivers. 
    Any Driver should inherit this classes API.
    
    Arguments
    ---------
    comm: MPI.COMM
        MPI communicator. In general, this is optional and need not be 
        provided to any Driver. Although, some Drivers are MPI aware because
        their workflows are not necessarily naively parallel. 
    
    """
    def __init__(self, comm=None, **settings_kw):
        ## initialize settings
        self.init_mpi(comm)
        pass
    
    
    def init_mpi(self, comm=None):
        global use_mpi
        if not use_mpi:
            self.comm = None
            self.rank = 0
            self.size = 0
            return

        self.comm = comm
        if self.comm == None:
            self.comm = MPI.COMM_WORLD
        
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        
    
    
    def calc(self, struct_obj, write=False):
        """
        Wrapper function to enable operation for both a single Structure 
        object or an entire Structure Dictionary.
        
        Arguments
        ---------
        struct_obj: mcse.Structure or Dictionary
            Arbitrary structure object. Either dictionyar or structure.
            
        """
        if type(struct_obj) == dict or type(struct_obj) == StructDict:
            return self.calc_dict(struct_obj, write=write)
        else:
            return self.calc_struct(struct_obj)
            
    
    def calc_dict(self, struct_dict, write=False):
        """
        Calculates entire Structure dictionary.
        
        """
        for struct_id,struct in struct_dict.items():
            self.calc_struct(struct)
            if write:
                self.write("")
            
    
    def calc_struct(self, struct):
        """
        Perform driver calculation on the input Structure. 
        
        """
        raise Exception("Not implemented")
            
    
    def write(self, output_dir, file_format="json", overwrite=False):
        """
        Writes the Driver's output to the to the output_dir argument. The only 
        specification of the Driver.write function for the mcse API are the 
        arguments specified above. The output of a Driver is not necessarily a 
        Structure, although this is the most common situation so the write 
        arguments are tailored for this purpose. In principal, the output of an 
        mcse Driver is not specified by the API and could be anything.  
        
        Usually, a Driver will only need to output a Structure file, with the 
        struct_id as the name of the output file, to the output directory. 
        This is what's done here.
        
        """
        if len(output_dir) == 0:
            output_dir = self.__class__.__name__
            
        ## Define dictionary for write API. 
        temp_dict = {self.struct.struct_id: self.struct}
        write(output_dir, 
              temp_dict, 
              file_format=file_format, 
              overwrite=overwrite)
    
        
    def settings(self):
        """
        Returns the settings that would be use to initialize the Driver 
        
        """
        return get_settings(self)
    
    
    def get_settings(self):
        return self.settings()
    
    
    @classmethod
    def from_settings(cls,dictionary):
        return from_settings(dictionary)
        
    
    def restart(self, output_dir):
        """
        Identify the progress of the Driver, find the proper way to restart, 
        and begin calculation. 
        
        """
        raise Exception("Not Implemented")
    
    
    def check(self, output_dir):
        """
        Check if the calculation has been performed correctly. 
        
        """
        raise Exception("Not Implemented")
        
    
    def report(self, output_dir, report=None, ):
        """
        Contribute or create a report for the Driver calculation. 
        
        """
        raise Exception("Not Implemented")
        
    
    def __call__(self, struct_obj):
        return self.calc(struct_obj)
    
    
    
class Driver(BaseDriver_):
    """
    Just short-hand for BaseDriver_
    """
        
        
class PairwiseDriver(BaseDriver_):
    """
    Base mcse Driver class for performing calculations on pairs of Structures. 
    This is relevant, for example, when comparing the RMSD between two 
    molecules or crystals. 
    
    Default behavior is that if these Drivers are given a struct_dict, then 
    a pair-wise calculation is performed between all pairs in the struct_dict. 
    
    """
    def __init__(self, comm=None, verbose=False, **settings_kw):
        ## initialize settings
        self.init_mpi(comm)
        self.verbose = verbose
        self.pair_dict = {}
        self.total = 0
        pass
    
    
    def calc_dict(self, struct_dict, write=False):
        self.pair_dict = self._get_pair_dict(struct_dict)
        
        for struct_id,struct in struct_dict.items():
            struct.properties[self.__class__.__name__] = []
        
        idx = 1
        for struct_id,struct_id_list in self.pair_dict.items():
            for temp_struct_id in struct_id_list:
                struct1 = struct_dict[struct_id]
                struct2 = struct_dict[temp_struct_id]
                temp_result = self.calc_struct(struct1,struct2)
                
                ### Store result in struct properties
                struct1.properties[self.__class__.__name__].append(
                    (temp_struct_id, temp_result))
                struct2.properties[self.__class__.__name__].append(
                    (struct_id, temp_result))
                
                if write:
                    self.write("", overwrite=True)
                
                if self.verbose:
                    print("{}/{}: {},{}={:.6f}"
                          .format(
                              idx, self.total,
                              struct_id, temp_struct_id, 
                              temp_result))
                
                idx += 1
    
    
    def calc_struct(self, struct1, struct2):
        self.struct1 = struct1
        self.struct2 = struct2
        raise Exception("Not Implemented")
        
    
    def write(self, output_dir, file_format="json", overwrite=False):
        if len(output_dir) == 0:
            output_dir = self.__class__.__name__
        
        ## Define dictionary for write API. 
        temp_dict = {self.struct1.struct_id: self.struct1,
                     self.struct2.struct_id: self.struct2}
        write(output_dir, 
              temp_dict, 
              file_format=file_format, 
              overwrite=overwrite)
            
        
    def _get_pair_dict(self, struct_dict):
        """
        Get the dictionary of pair_dict that have to be made for pair mode
        esxecution.
        
        """
        self.total = 0
        
        ### Just doing this to get the correct shape and index values for a
        ## pairwise comparison
        temp = np.arange(0,len(struct_dict),1)
        square = temp + temp[:,None]
        idx = np.triu_indices(n=square.shape[0],
                              k=1,
                              m=square.shape[1]) 
        
        #### Build the dictionary of indicies that each structure must be 
        ## compared to for pairwise comparison.
        keys = [x for x in struct_dict.keys()]
        comparison_dict = {}
        for key in keys:
            comparison_dict[key] = []
            
        for idx_pair in zip(idx[0],idx[1]):
            key = keys[idx_pair[0]]
            comparison_dict[key].append(keys[idx_pair[1]])
            self.total += 1
        
        return comparison_dict
        
    
