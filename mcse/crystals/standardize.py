

import numpy as np

from mcse import Structure
from mcse.core.utils import com
from mcse.molecules.label import match_labels,label
from mcse.crystals import FindMolecules
           
       
def standardize(struct, 
               bonds_kw={"mult":1.20, "skin":0.0, "update":False},
               residues=0):
    """
    Makes all molecules in the unit cell whole molecules without breaking 
    covalent bonds across periodic boundary conditions
    
    """
    if bonds_kw == {}:
        bonds_kw = {"mult":1.20, "skin":0.0, "update":False}
    struct.get_bonds(**bonds_kw)
    struct.get_molecule_idx()
    
    fm = FindMolecules(residues=residues,
                        mult=bonds_kw["mult"], 
                        conformation=False, 
                        mult_range=np.arange(1.05, 1.25, 0.005),
                        supercell_radius=10,
                        verbose=False)
    fm.calc_struct(struct)
    
    ### After translation, the molecules will have whole representations
    struct.translate(fm.translate)
    bonds_kw["update"] = True
    struct.get_bonds(**bonds_kw)
    mol_idx = struct.get_molecule_idx(**bonds_kw)
    
    ### Next standardization step is to provide consistent labels for all the 
    ### molecules in the crystal structure
    mol_list = [x for x in struct.molecules.values()]
    target_mol = label(mol_list[0])
    for temp_mol in mol_list[1:]:
        match_labels(target_mol, temp_mol)
    
    ### Reconstruct the geometry given these new consistent labelings
    geo = np.array([[0,0,0]])        
    ele = np.array([])
    for temp_mol in mol_list:
        geo = np.vstack([geo, temp_mol.get_geo_array()])
        ele = np.hstack([ele, temp_mol.elements])
    geo = geo[1:]
    struct.from_geo_array(geo, ele)
    
    ### And finally recalculate bonds for the last time and update molecules
    bonds_kw["update"] = True
    struct.get_bonds(**bonds_kw)
    struct.get_molecule_idx(**bonds_kw)
    struct.molecules
    
    return struct


def preprocess(struct, 
               bonds_kw={"mult":1.20, "skin":0.0, "update":False},
               residues=0):
    """ Old Name """ 
    return standardize(struct, bonds_kw=bonds_kw, residues=residues)



if __name__ == "__main__":
    pass   
    
        
    
    
    
    
    
    
    
    
    
    