


"""
File for structure checks:
    - Duplicates
    - Physical Structure
    - Molecule in structure checks

"""

import os,json
import numpy as np

from mcse import Structure
from mcse.io import read,write
from mcse.core.driver import PairwiseDriver

from pymatgen.analysis.structure_matcher import (StructureMatcher,
                                                ElementComparator,
                                                SpeciesComparator,
                                                FrameworkComparator)


class pymatgen_compare():
    def __init__(self, pymatgen_kw=
                 {
                        "ltol": 0.2,                                    
                        "stol": 0.3,
                        "angle_tol": 5,
                        "primitive_cell": True,
                        "scale": True,
                        "attempt_supercell": False
                 }):
        self.kw = pymatgen_kw
    
    
    def __call__(self, struct1, struct2):
        sm =  StructureMatcher(
                    **self.kw,                                 
                    comparator=SpeciesComparator())                           
                                                                               
        pstruct1 = struct1.get_pymatgen_structure()                                      
        pstruct2 = struct2.get_pymatgen_structure() 
        
        return bool(sm.fit(pstruct1, pstruct2))
    
    
    
class DuplicateCheck(PairwiseDriver):
    """
    Checks if there are duplicate structures in an entire structure dictionary. 
    Duplicaate check may be ran in a serial or parallel way using MPI. This is
    automatically detected and handled by the program. 
    
    Arguments
    ---------
    compare_fn: callable
        Callable that performs the comparison. This function can be arbitrary,
        but it must take in two structures as an argument and return True if
        the structures are duplicates, or False if they are not. See the 
        pymatgen_compare class for an example of what this might look like.
        
    """
    def __init__(self, 
                 compare_fn=pymatgen_compare(),
                 **kwargs):
        
        self.compare_fn = compare_fn
        self.duplicate_dict = {}
        self.struct_dict = {}
        
        super().__init__(**kwargs)
        
    
    def _check_mode(self, mode):
        self.modes = ["pair", "complete"]
        if mode not in self.modes:
            raise Exception("DuplicateCheck mode {} is not available. "
                            .format(mode) +
                            "Please use one of {}.".format(self.modess))
        else:
            self.mode = mode
            
            
    def calc_struct(self, struct1, struct2):
        self.struct1 = struct1
        self.struct2 = struct2
        result = self.compare(struct1, struct2)
        
        self.struct_dict[self.struct1.struct_id] = self.struct1
        self.struct_dict[self.struct2.struct_id] = self.struct2
        
        if self.struct1.struct_id not in self.duplicate_dict:
            self.duplicate_dict[self.struct1.struct_id] = [
                    self.struct1.struct_id
                ]
        if self.struct2.struct_id not in self.duplicate_dict:
            self.duplicate_dict[self.struct2.struct_id] = [
                    self.struct2.struct_id
                ]
        
        if result:
            self.duplicate_dict[struct1.struct_id].append(
                        struct2.struct_id)
            self.duplicate_dict[struct2.struct_id].append(
                    struct1.struct_id)
                
        return result
    
    
    def compare(self, struct1, struct2):      
        """
        Compare structures using pymatgen's StructureMatcher by default
        
        """                                                           
        return self.compare_fn(struct1, struct2)
    
    
    @property
    def unique(self):
        ### First find unique duplicates in the duplicate_dict
        id_used = {}
        unique = {}
        for name,duplicates in self.duplicate_dict.items():
            if len(duplicates) == 1:
                continue
            elif name in id_used:
                continue
            else:
                unique[name] = self.struct_dict[name]
                for struct_id in duplicates:
                    id_used[struct_id] = True
        
        return unique
    
            

if __name__ == "__main__":     
    pass
    
    