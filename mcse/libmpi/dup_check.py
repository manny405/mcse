# -*- coding: utf-8 -*-


import numpy as np

from scipy.special import comb

from mcse.libmpi.base import _NaiveParallel

from mpi4py import MPI  


class DupCheck(_NaiveParallel):
    """
    High performance implementation of duplicate check for a dictionary of 
    Structures. While some parts of the implementation may not look well 
    optimize, it eleviates the major bottleneck, which is for-loops in Python
    over millions or tens of millions of entries. For example, it is much better 
    to communicate the entire Structure list (which may be 1,000-10,000) at the
    beginning of the calculation than try to send a minimal list from rank 0. 
    
    """
    
    def calc(self, compare_fn, struct_dict, compare_fn_kw={}):
        """
        Arguments
        ---------
        compare_fn: callable
            Function that will compare structures and return True or False
        struct_dict: StructDict
            Structure dictionary
        compare_fn_kw: dict
            Dictionary of key word arguments for the comparison function. 
            
        """
        self.dup_dict = {}
        if type(compare_fn) == dict:
            raise Exception()
            
        ## Make sure struct_dict is identical on all ranks
        struct_dict = self.comm.bcast(struct_dict,root=0)
        
        ## Performing pairwise check
        ## Will be faster to build on each rank than communicate
        I,J = np.triu_indices(len(struct_dict), 1)
        pairwise = np.hstack([I[:,None], J[:,None]])
        keys = [x for x in struct_dict.keys()]
        
        my_list = self.get_list(pairwise)
        
        dup_dict = {}
        for row in my_list:
            struct1_id = keys[row[0]]
            struct2_id = keys[row[1]]
            struct1 = struct_dict[struct1_id]
            struct2 = struct_dict[struct2_id]
            kw = {"struct1": struct1, "struct2": struct2}
            result = compare_fn(**kw, **compare_fn_kw)
            
            if struct1_id not in dup_dict:
                dup_dict[struct1_id] = {struct1_id: True}
            if struct2_id not in dup_dict:
                dup_dict[struct2_id] = {struct2_id: True}
            
            if result:
                dup_dict[struct1_id][struct2_id] = True
                dup_dict[struct2_id][struct1_id] = True
        
        self.dup_dict = dup_dict
        all_dup_dict = self.comm.gather(dup_dict, root=0)
        
        ## Finish by parsing results on rank 0
        if self.rank ==0:
            
            self.master_dup_dict = {}
            for struct_id in struct_dict:
                self.master_dup_dict[struct_id] = {struct_id: True}
            
            for key,value in self.master_dup_dict.items():
                for entry_dict in all_dup_dict:
                    if key in entry_dict:
                        temp_dup_dict = entry_dict[key]
                        for temp_value in temp_dup_dict:
                            value[temp_value] = True
                        self.master_dup_dict[key] = value
                        
            ## Collect unique
            ## Sort through duplicate check to identify unique dimers
            used_id = {}
            unique_dict = {}
            for key,value in self.master_dup_dict.items():
                if key in used_id:
                    continue
                
                if len(value) == 1:
                    used_id[key] = True
                    dimer = struct_dict[key]
                    unique_dict[dimer.struct_id] = dimer
                    continue
                else:
                    dimer = struct_dict[key]
                    unique_dict[dimer.struct_id] = dimer
                    
                    for dimer_id in value:
                        used_id[dimer_id] = True
    
            return unique_dict
        else:
            return {}
                        
                        
if __name__ == "__main__":
    pass
    
        
        
        
