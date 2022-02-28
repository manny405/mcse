
import numpy as np
from scipy.spatial.distance import cdist

from ase.data import chemical_symbols,vdw_radii,atomic_numbers,covalent_radii

from mcse.molecules.compare import compare_graph,compare_rmsd_pa
from mcse.dimers.cutoffs import get_cutoff_matrix


def check_dimer(struct, bonds_kw={}, exception=True):
    """
    Check if a structure is a dimer. 
    
    Arguments
    ---------
    struct: Structure
        Checks if struct is a dimer of identical molecules. 
    exception: bool 
        Controls if Excpetion will be raised if the structure is not a dimer. 
    
    """
    ## Check if has lattice vectors
    if len(struct.lattice) != 0:
        if not exception:
            return False
        
        raise Exception("Structure {} is not a dimer".format(struct))
    
    struct.get_bonds(**bonds_kw)
    if len(struct.molecules) != 2:
        if not exception:
            return False
        raise Exception("Structure {} is not a dimer".format(struct))

    return True


def check_dist(dimer, 
                min_sr=-1, 
                max_sr=-1, 
                vdw=[], 
                bonds_kw={},
                cutoff_matrix=[],
                check_idx=[]):
    """
    Efficient implementation for calulating intermolecular distances between 
    two molecules in a dimer configuration. By default, the minimum sr between
    the dimers is returned. If a the min_sr and/or max_sr is provided, then if 
    a bool is returned where False means the input dimer did not pass the 
    distance check. 
    
    Arguments
    ---------
    dimer: Structure
        Dimer structure to check 
    min_sr: float
        Minimum specific radius to use for dimer distance checks.
    max_sr: float
        Maximum specific radius multiplier that is allowed to be the minimum 
        distance between two dimers, thereby removing dimers formed from molecules
        that are far away. Default is that this is not checked. 
    vdw: list
        List of all vdw radii for all elements in periodic table
    bonds_kw: dict
        Arguments for Structure.get_bonds
    check_idx: list
        Indices to use in determining if the structure is inside of the provided
        min_sr to max_sr range
        
    """
    geo = dimer.geometry
    ele = dimer.elements
    mol_idx = dimer.get_molecule_idx(**bonds_kw)
    mol1 = dimer.get_sub(mol_idx[0], struct_id="mol1")
    mol2 = dimer.get_sub(mol_idx[1], struct_id="mol2")
    
    if len(mol_idx) != 2:
        raise Exception("Structure {} not recognized as dimer. "
                .format(dimer.struct_id)+
                "Found to have {} molecules.".format(len(mol_idx)))
    
    if len(vdw) == 0:
        vdw = vdw_radii
    
    if len(cutoff_matrix) == 0:
        cutoff_matrix = get_cutoff_matrix(mol1,mol2,
                                        vdw=vdw,
                                        bonds_kw=bonds_kw)
    
    geo1 = geo[mol_idx[0]]
    geo2 = geo[mol_idx[1]]
    
    dist = cdist(geo1,geo2)
    
    if dist.shape != cutoff_matrix.shape:
        raise Exception("Shape of dimer intermolecular distance matrix and "+
            "cutoff matrix did not match")
        
    dist_sr = np.abs(dist) / cutoff_matrix
    if len(check_idx) > 0:
        if type(check_idx) == list:
            if type(check_idx[0]) != list:
                raise Exception("Argument check_idx must be list of lists")
            check_idx = np.vstack(check_idx)
        dist_sr = dist_sr[check_idx[:,0], check_idx[:,1]]
    test_min_sr = np.min(dist_sr)
    
    if min_sr < 0 and max_sr < 0:
        return test_min_sr
    else:
        if test_min_sr < min_sr:
            ### By default, this should never return False if max_sr is set
            return False
        
        if max_sr > 0:
            if test_min_sr > max_sr:
                return False
        
        ### If haven't returned yet, must be True
        return True
    
    
def check_identical(dimer, 
                    rmsd_tol=0.15, 
                    bonds_kw={}, 
                    chiral=False, 
                    trusted=False):
    """
    Check if the input dimer is made up of identical molecules. This function
    will be relatively slow in order to achieve good accuracy for comparing
    the molecules in the dimer, and there's not much way around this. 

    """
    
    if not trusted:
        check_dimer(dimer)
        
    mols = dimer.molecules
    mol_ids = list(mols.keys())
    mol1 = mols[mol_ids[0]]
    mol2 = mols[mol_ids[1]]
    
    if not compare_graph(mol1,mol2,bonds_kw):
        return False

    results = compare_rmsd_pa(mol1,mol2,rmsd_tol,bonds_kw,chiral)
    if not results[0]:
        return False
    else:
        return True