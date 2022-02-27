
import numpy as np

from mcse.core import Structure
from mcse.molecules import rot_mol 
from mcse.molecules.compare import compare_rmsd_pa
from mcse.molecules.align import get_principal_axes,fast_align
from mcse.dimers.check import check_dimer


def params(struct, identical=True, ret_mol=False, trusted=False):
    """
    Returns a set of parameteters generate the given dimer, which are the 
    relative orientations and the COM vector separating them. If the molecules  
    in the dimer are not identical, then the orientations of each individual 
    molecular component are returned. 
    
    Arguments
    ---------
    trusted: bool
        This is only if you know that what is being input is a trusted dimer. 
        This saves 30% of compute time if the input is known to be valid. 
    
    """
    if not trusted:
        check_dimer(struct)
    
    mols = struct.molecules
    mol_ids = list(mols.keys())
    mol1 = mols[mol_ids[0]]
    mol2 = mols[mol_ids[1]]
    
    ### Compute trans
    trans = mol2.com - mol1.com
    ### Center molecules needed from this point on
    mol1.translate(-mol1.com)
    mol2.translate(-mol2.com)
    
    ### If mol1 and mol2 are not identical, then return the general orientations
    ###   and the oriented molecules
    if not identical:
        ret = [get_principal_axes(mol1), get_principal_axes(mol2), trans]
        if ret_mol:
            fast_align(mol1)
            fast_align(mol2)
            ret += [mol1, mol2]
        return tuple(ret)
    
    ### Compute exact relative rot
    fast_align(mol1)
    h = np.dot(mol2.geometry.T,mol1.geometry)
    u,s,v = np.linalg.svd(h)
    d = np.linalg.det(np.dot(v,u.T)) 
    d = np.array([[1,0,0],[0,1,0],[0,0,d]])
    rot = np.dot(u,np.dot(d,v))
    
    assert np.linalg.norm(mol2.geometry - np.dot(rot, mol1.geometry.T).T) < 1e-2
    
    ret = [np.diag([1,1,1]).astype(float), rot, trans]
    if ret_mol:
        ret += [mol1, mol1]
        
    return tuple(ret)
    

def generate(rot1,rot2,trans,mol1,mol2,check_orientations=True):
    """
    Takes parameters defining a dimer and generates it. The parameters are 
    over-defined in the sense that the rotation of the first molecule is not
    necessary in general. However, the user can easily input np.diag([1,1,1])
    as the first argument and use the desired relative rotation as the second
    argument. 
    
    Arguments
    ---------
    orientation: bool
        Checks that the orientations of the input molecules are valid. For 
        fast performance in the case that the user knows exactly what they are
        doing this can be turned off
    
    """
    ### Fast test for validity before generation
    if np.linalg.norm(mol1.com) > 1e-4:
        raise Exception("COM of mol1 must be at origin")
    if np.linalg.norm(mol2.com) > 1e-4:
        raise Exception("COM of mol2 must be at origin")
    
    ### Check orientation if desired
    if check_orientations:
        test1 = fast_align(mol1.copy())
        test2 = fast_align(mol2.copy())
        result1 = compare_rmsd_pa(mol1,test1)
        result2 = compare_rmsd_pa(mol2,test2)
        if not result1[0] or not result2[0]:
            raise Exception("Principal axes of the molecule are not aligned "+
                    "with the origin. To correct this use "+
                    "mcse.molecules.align.fast_align ")
    
    ### Copy molecules so as to not make any changes to the inputs, 
    ###   particularly if mol1 and mol2 are actually the same object
    m1 = mol1.geometry
    m2 = mol2.geometry
    ### Rotate and translate
    m1 = np.dot(rot1, m1.T).T
    m2 = np.dot(rot2, m2.T).T
    m2 += trans
    ### Combine 
    final_geo = np.vstack([m1, m2])
    final_ele = np.hstack([mol1.elements, mol2.elements])
    mol_idx = [np.arange(0,len(m1)).tolist()]
    mol_idx += [np.arange(len(m1),len(m1)+len(m2)).tolist()]
    bonds = mol1.get_bonds().copy()
    offset = len(bonds)
    for bond_list in mol2.get_bonds():
        temp_add_bonds = []
        for temp_atom_idx in bond_list:
            temp_atom_idx = temp_atom_idx+offset
            temp_add_bonds.append(temp_atom_idx)
        bonds += temp_add_bonds
    dimer_id = "dimer_{}_{}_{}".format(
        mol1.struct_id,
        mol2.struct_id, 
        hash_params(rot1,rot2,trans)
    )
    properties = {
        "dimer_params": (rot1.tolist(),
                         rot2.tolist(),
                         trans.tolist()),
        "bonds": bonds,
        ### Individual molecules don't need to be stored in dimer_params 
        ###  because these can always be extracted from the dimer from the 
        ###  manually given mol_idx here
        "molecule_idx": mol_idx
    }
    
    return Structure(struct_id=dimer_id,
                     geometry=final_geo,
                     elements=final_ele,
                     lattice=[],
                     bonds=[],
                     properties=properties)
    
    

def hash_params(rot1,rot2,trans):
    r1b = rot1.tobytes()
    r2b = rot2.tobytes()
    trans = trans.tobytes()
    return hash(r1b+r2b+trans)
    
    