# -*- coding: utf-8 -*-

import numpy as np

from mcse.molecules.label import get_bond_fragments
from mcse.molecules.align import fast_align,match_principal_axes
from mcse.molecules.rmsd import rmsd

 
def unique_by_formula(struct_dict):
    """
    Just checks for duplicate formulas
    
    """
    molecule_struct_list = [x for x in struct_dict.values()]
    
    ## First test that formulas are the same
    same_formula = []
    added_list = []
    for idx1,molecule1 in enumerate(molecule_struct_list):
        temp_list = []
        ## Check if molecule has already been added
        if idx1 in added_list:
            continue
        else:
            added_list.append(idx1)
            
        formula1 = molecule1.formula
        temp_list.append(molecule1)
        
        ## Compare with all others
        for idx2,molecule2 in enumerate(molecule_struct_list[idx1+1:]):
            ## Make sure to offset idx2 correctly
            idx2 += idx1+1
            formula2 = molecule2.formula
            if formula1 == formula2:
                temp_list.append(molecule2)
                added_list.append(idx2)
        
        ## Now add in groups of molecules
        same_formula.append(temp_list)
    
    unique_molecules = []
    for same_list in same_formula:
        unique_molecules.append(same_list[0])
    
    return unique_molecules

      
def unique_by_bonding(struct_dict):
    """
    Identify unique molecules based on their bonding
    
    """
    molecule_struct_list = [x for x in struct_dict.values()]
    
    ## First test that formulas are the same
    same_formula = []
    added_list = []
    for idx1,molecule1 in enumerate(molecule_struct_list):
        temp_list = []
        ## Check if molecule has already been added
        if idx1 in added_list:
            continue
        else:
            added_list.append(idx1)
            
        formula1 = molecule1.formula
        temp_list.append(molecule1)
        
        ## Compare with all others
        for idx2,molecule2 in enumerate(molecule_struct_list[idx1+1:]):
            ## Make sure to offset idx2 correctly
            idx2 += idx1+1
            formula2 = molecule2.formula
            if formula1 == formula2:
                temp_list.append(molecule2)
                added_list.append(idx2)
        
        ## Now add in groups of molecules
        same_formula.append(temp_list)
    
    ## There's a range of physically relevant mult
    ## ranges to search over for the bond neighborhood calculation. 
    ## This is physical issue caused by describing all covalent bonds by a 
    ## single number for every bonding environment.
    ## This method, while hack-ish, is physically motivated and more robust.
    bn_unique = []
    for mult in [0.85, 0.9, 0.95, 1.0, 1.1, 1.15, 1.2, 1.25, 1.3]:
        temp_unique = []
        
        for molecule_group in same_formula:
            if len(molecule_group) == 1:
                ## Must be unique
                temp_unique.append(molecule_group[0])
                continue
            
            ## Otherwise, compare bonding information
            bonding_info = []
            for molecule in molecule_group:
                fragments,count = get_bond_fragments(molecule, 
                                    bonds_kw={"mult": mult,"skin": 0,"update": True},
                                    return_counts=True)
                bonding_info.append((fragments.tolist(),list(count)))
                
            added_list = []
            for idx1,bonding1 in enumerate(bonding_info):
                if idx1 in added_list:
                    continue
                else:
                    added_list.append(idx1)
                    ## Must be unique at this point
                    temp_unique.append(molecule_group[idx1])
                    
                frag1,count1 = bonding1
                
                for idx2,bonding2 in enumerate(bonding_info[idx1+1:]):
                    idx2 += idx1+1
                    frag2,count2 = bonding2
                    
                    ## Compare
                    if frag1 == frag2 and count1 == count2:
                        added_list.append(idx2)
                    else:
                        ## Don't have to do anything because if the molecule is 
                        ## unique then it will be put in the unique list in 
                        ## the loop just outside of this one. 
                        pass
        
        bn_unique.append(temp_unique)
    
    ## Return smallest from bn unique since we have searched over all 
    ## physically relevant values. This gives a more robust result. 
    length = [len(x) for x in bn_unique]
    min_idx = np.argmin(length)
    
    return bn_unique[min_idx]


def unique_by_rmsd(struct_dict, tol=0.1):
    """
    Identify unique molecules by differences in their RMSD based on their 
    cartesian coordinates
    
    """
    ### Need to copy because will be rotating the molecules for RMSD
    molecule_struct_list = [x.copy() for x in struct_dict.values()]
    for temp_mol in molecule_struct_list:
        fast_align(temp_mol)
        
    num_mol = len(molecule_struct_list)
    
    ### Calculate difference matrix using RMSD and find the unique groups within
    ### the difference matrix
    difference_matrix = np.zeros((num_mol,num_mol))
    for i in range(num_mol):
        for j in range(i+1,num_mol):
            molecule_0 = molecule_struct_list[i]
            molecule_1 = molecule_struct_list[j]
<<<<<<< HEAD
            if molecule_0.formula != molecule_1.formula:
                temp_rmsd = 100
            else:
                _,_,_,temp_rmsd = rmsd(molecule_0, molecule_1)
=======
            _,_,_,temp_rmsd = rmsd(molecule_0, molecule_1)
>>>>>>> 2c13de4f90a9911aa429cd876b9ceb28a5d3d6c4
            difference_matrix[i,j] = temp_rmsd
            difference_matrix[j,i] = temp_rmsd
    
    molecule_groups = unique_groups(difference_matrix, tol=tol)
    unique_molecule_idx = [x[0] for x in molecule_groups]
    unique_dict = {}
    for idx in unique_molecule_idx:
        temp_mol = molecule_struct_list[idx]
        unique_dict[temp_mol.struct_id] = temp_mol
    
    return unique_dict
    
    
def unique_groups(difference_matrix, tol=1):
    """ 
    Breaks difference matrix into groups of similar molecules.
    Returns list molecules which make up unique groups for indexing into the 
    original molecule_struct_list.
    """
    # List of molecules which need to be sorted
    unsorted = np.array([x for x in range(difference_matrix.shape[0])])
    # List of groups of the same molecules
    molecule_groups = []
    
    while len(unsorted) != 0:
        # Pick first molecule which hasn't been sorted
        current_idx = unsorted[0]
        # Take its row of the difference matrix
        row = difference_matrix[current_idx,:]
        # Take positions of other molecules yet to be sorted
        row = row[unsorted]
        # Find those same by tolerance value
        same_tol = row < tol
        # Must be greater than 0 so value cannot be -1
        same_nonzero = row >= 0
        # Combine
        same = np.logical_and(same_tol, same_nonzero)
        # Gather results
        same_idx = np.where(same == True)[0]
        # Reference original molecule index which is obtained from the 
        # unsorted list for the final unique groups
        molecule_groups.append(unsorted[same_idx])
        # Delete indexs in unsorted which have now been sorted
        unsorted = np.delete(unsorted, same_idx)              
    
    return molecule_groups
    