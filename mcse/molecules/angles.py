# -*- coding: utf-8 -*-


"""
Calculating and identifying dihedral angles

Searching over dihedral angles for molecules

"""

import numpy as np

from mcse.molecules.label import label


def angles(mol, 
           degrees=True,
           reorder=True,
           bonds_kw={"mult": 1.20, "skin": 0.0, "update": False},
           unique=True):
    """
    Returns all of the angles in the molecule. Current implementation is slow 
    but works.
    
    """
    mol.get_bonds(**bonds_kw)
    if reorder:
        mol = label(mol)
    
    ### Already got bonds according to input bonds_kw the first time
    bonds = mol.get_bonds(update=False)
    geo = mol.get_geo_array()
    
    angle_dict = {}
    for atom_idx,bond_list in enumerate(bonds):
        ### Can't construct angle between only two atoms
        if len(bond_list) == 1:
            continue
        
        for temp_atom_idx_1,bond_idx_1 in enumerate(bond_list):
            next_idx = temp_atom_idx_1+1
            for temp_atom_idx_2,bond_idx_2 in enumerate(bond_list[next_idx:]):
                pos1 = geo[atom_idx]
                pos2 = geo[bond_idx_1] - pos1
                pos3 = geo[bond_idx_2] - pos1
                
                temp_dot = np.dot(pos2, pos3) / (np.linalg.norm(pos2) * 
                                                 np.linalg.norm(pos3))
                if temp_dot > 1:
                    if temp_dot - 1 < 1e-3:
                        temp_dot = 1.0
                
                if degrees:
                    temp_angle = np.rad2deg(np.arccos(temp_dot))
                else:
                    temp_angle = np.arccos(temp_dot)
                
                if unique:
                    temp_ordering = np.sort([atom_idx, 
                                             bond_idx_1, 
                                             bond_idx_2])
                    temp_ordering = tuple(temp_ordering)
                    if temp_ordering in angle_dict:
                        continue
                else:
                    temp_ordering = (atom_idx, 
                                     bond_idx_1, 
                                     bond_idx_2)
                
                angle_dict[temp_ordering] = temp_angle
    
    return angle_dict






