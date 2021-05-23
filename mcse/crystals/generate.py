
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist,squareform

from mcse import Structure
from mcse.core.utils import com
from mcse.molecules.label import label
from mcse.molecules.compare import unique_by_formula
from mcse.molecules.rmsd import calc_rmsd,calc_rmsd_ele
from mcse.molecules.align import align,fast_align,match_principal_axes,align_molecules_rot
from mcse.crystals import rot_struct,combine,get_molecules
from mcse.crystals import preprocess as crystal_preprocess


def get_params(struct, 
               use_fast_align=True,
               spg=False, 
               preprocess=True, 
               unique_fn=unique_by_formula):
    """
    Obtains the parameters used to generate the crystal structure. These are 
    the lattice vectors and unique molecules with their COM positions and 
    relative orientations based on their principal axis. 
    
    (Can verify this is correct by starting with a crystal structure, 
     obtaining the parameters, passing to from_parameters, and finally
     testing overlap using RMSD)
    
    
    Arguments
    ---------
    fast_align: bool
        If True, will use fast alignment algorithm for molecule. If False, 
        will use slower, symmetry aware algorithm. 
    
    Returns
    -------
    (molecule_list, COM, orientations, lattice)
        molecule_list are the centered and aligned unique unique molecules 
          from the crystal structure. 
        COM is a 3-dimensional array where the 3rd dimension determines the
          corresponding molecule from the molecule_list
        orientations is a 3-dimensoinal array of euler angles where the 3rd
          dimension determines the corresponding molecule from the molecule_list
    
    """
    if spg == True:
        raise Exception("Obtaining parameters based on the space group "+
                "symmetry is not implemented.")
        
    if preprocess:
        crystal_preprocess(struct)
    
    mol_list = get_molecules(struct, ret="struct")
    unique_mol_list = unique_fn(mol_list)
    if len(unique_mol_list) > 1:
        raise Exception("Multiple unique molecules not implemented")
        
    for idx,temp_mol in enumerate(unique_mol_list):
        if use_fast_align:
            fast_align(temp_mol)
        else:
            align(temp_mol)
        
    ### Unique function may change geometries of molecules in mol_list
    ###   start with new copy
    mol_list = get_molecules(struct, ret="struct")
    com_list = [[] for x in unique_mol_list]
    orientation_list = [[] for x in unique_mol_list]
    for temp_mol in mol_list:
        ### TODO: Choose this dynamically, then multiple unique molecules 
        ###       will be fully implemented
        unique_mol_idx = 0
        temp_unique_mol = unique_mol_list[unique_mol_idx]
        
        ### Once COM is obtained, then can get orientation
        temp_com = com(temp_mol)
        temp_mol.translate(-temp_com)
        
        ### Unique Molecule should come second because the orientation should 
        ###   be given such that it transforms the unique_mol into the temp_mol
        _,temp_orientation = match_principal_axes(temp_unique_mol,
                                                  temp_mol,
                                                  pre_align=False)
        
        ### Assuming that temp_unique_mol has aligned principal axis, 
        ###   then the inverse==transpose is the correct operation to apply
        ###   to the unique_mol
        temp_orientation = temp_orientation.T
        
        com_list[unique_mol_idx].append(temp_com)
        orientation_list[unique_mol_idx].append(temp_orientation)
    
    com_params = np.array(com_list)
    orientation_params = np.array(orientation_list)
    lv = np.vstack(struct.get_lattice_vectors())
    
    return unique_mol_list,com_params,orientation_params,lv
    

def from_params(unique_mol_list, 
                com_list,
                orientation_list,
                lattice, 
                spg=0):
    """
    
    
    """
    if spg > 0:
        raise Exception("Generation by space group is not implemented")
    
    geo = np.array([[0,0,0]])
    ele = np.array([])
    
    for mol_idx,temp_com_list in enumerate(com_list):
        temp_unique_mol = unique_mol_list[mol_idx]
        temp_mol_geo = temp_unique_mol.get_geo_array()
        temp_mol_ele = temp_unique_mol.elements
        
        for com_idx,temp_com in enumerate(temp_com_list):
            temp_orient = orientation_list[mol_idx][com_idx]
            
            ### First need to apply orientation
            temp_new_geo = np.dot(temp_orient, temp_mol_geo.T).T
            ### Then apply translation by COM
            temp_new_geo += temp_com[None,:]
            
            ### Add to building geo
            geo = np.vstack([geo, temp_new_geo])
            ele = np.hstack([ele, temp_mol_ele])
            
    ### Just get rid of first entry that was used to make API easier
    geo = geo[1:]
    struct = Structure.from_geo(geo, ele, lat=list(lattice))
        
    ### Store parameters used to generate the crystal structure
    
    return struct
    

def generate(molecule, spg_list=[], sr=0, Z=0):
    """
    Random molecular crystal generation or given space group list
    
    """
    raise Exception("Not Implemented")

