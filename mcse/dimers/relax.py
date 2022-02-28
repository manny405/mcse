
import numpy as np

from mcse.dimers.check import check_dimer,check_identical
from mcse.dimers.generate import params,hash_params



def offset(dimer, 
           dist, 
           idx=1,
           relative=True, 
           copy=True, 
           update_params=True, 
           check_identical_fn=check_identical,
           trusted=False):
    """
    Takes the input dimer and modified the COM distance between the molecules 
    without changing the orientation or direction of the relative COM vector. 
    This is used in constrained relaxation of dimer geometries. 

    Arguments
    ---------
    dimer: Structure
        Dimer to adjust offset
    dist: float   
        Amount to offset the dimer distance
    idx: int
        Determines the molecule that will be offset, either 0 or 1. This value 
        is typically 1 because molecule 0 will be typically be aligned at the 
        origin. 
    relative: bool
        If True, the distance argument is used to change the current distance 
        between the molecules. If False, the distance between the molecules is
        exactly set to this value, all without changing the direction of the 
        COM vector between the molecules of the dimer. 
    copy: bool
        If False, the input Structure will be modified. If False, a copy of the
        input Structure will be made to be modified. 
    update_params: bool
        If True, the params and struct_id of the input Structure will be 
        modified to match the new offset between the dimers. 
    check_identical_fn: callable
        Callable to check if the input dimer is made up of identical molecules.
        Default function is a general approach based on geometry. If it's known
        that the dimer is made up of identical molecules a definition such using
        lambda will be the best possible performance. For example, 
        check_identical_fn=lambda x: True
    trusted: bool
        If the inputs can be trusted or not. 

    """
    if not trusted:
        check_dimer(dimer)
    if copy:
        dimer = dimer.copy()
    if idx != 1:
        if idx != 0:
            raise Exception("Argument mol_idx must be 0 or 1")
        
    dist = float(dist)
    mols = dimer.molecules
    mol_ids = list(mols.keys())
    mol1 = mols[mol_ids[0]]
    mol2 = mols[mol_ids[1]]
    vec = mol2.com - mol1.com

    dot_value = dist / np.linalg.norm(vec)
    offset_vector = vec*dot_value
    offset_total_vector = vec + offset_vector
    
    ### Fast check for accuracy of implementation
    test_dist = np.linalg.norm(offset_total_vector - vec)
    if not trusted:
        assert np.abs(np.abs(dist) - test_dist) < 1e-4
    
    mol_idx = dimer.get_molecule_idx()
    if relative:
        dimer.geometry[mol_idx[idx]] += offset_vector
    else:
        ### First move COM to origin
        dimer.geometry[mol_idx[idx]] -= [mol1,mol2][idx].com
        ### Then translate by offset_vector
        dimer.geometry[mol_idx[idx]] += offset_vector

    
    dimer.properties["dimer_offset"] = float(dist)
    
    if update_params:
        if "dimer_params" not in dimer.properties:
            identical = check_identical_fn(dimer)
            init_params = params(dimer, 
                                 identical=identical, 
                                 ret_mol=False, 
                                 trusted=trusted)
            dparams = [[],[],[]]
            dparams[0] = init_params[0].tolist()
            dparams[1] = init_params[1].tolist()
            dparams[2] = init_params[2].tolist()
            dparams = tuple(dparams)
        else:
            init_params = dimer.properties["dimer_params"]
            init_hash = hash_params(*init_params[0:3])
            ### Remove this init_hash if it's in id of current structure
            if str(init_hash) in dimer.struct_id:
                dimer.struct_id = dimer.struct_id.replace("_{init_hash}","")
            dparams = [[],[],[]]
            dparams[0] = init_params[0]
            dparams[1] = init_params[1]
            dparams[2] = offset_total_vector.tolist()
            dparams = tuple(dparams)
            
        dimer.properties["dimer_params"] = dparams
        param_id = hash_params(*dparams)
        dimer.struct_id += f"_{param_id}"
        
    return dimer
    