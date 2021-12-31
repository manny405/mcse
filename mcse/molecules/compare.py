# -*- coding: utf-8 -*-


"""
Module for performing duplicate checks on molecules

The general API should be given as follows:
    (A) Takes in struct_dict of molecules and any other optional arguments
          such as geometric tolerances
    (B) Returns two dictionaries.
            1) Dictionary of just the unique molecules
            2) Dictionary containing two entries: "lookup_dups" and "lookup_unique"
                For "lookup_dups", every molecule that is passed in will have its
                  duplicates listed
                For "lookup_unique", every moleucle that is passed in will have 
                  its related unique chosen molecule
    (C) Because the function will make use of libmpi, it will be automatically 
          parallelizable
          
          
IN ADDITION, NO CHANGES TO THE MOLECULES SHOULD EVER BE APPLIED DURING COMPARISON, 
THEIR INTERNAL DATA (atom orders & geometry) MUST BE UNCHANGED AFTER RETURNING
FROM THE FUNCTION. 
"""

import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mcse import Structure
from mcse.molecules.label import mol2graph,message_passing_atom_descriptors,\
                                    get_symmetric_ordering,label_mp,get_hash_order
from mcse.molecules.align import get_principal_axes,fast_align
from mcse.molecules.utils import com
    

def compare(mol1, mol2, rmsd_tol=0.15, bonds_kw={}, chiral=True, fixed_order=[]):
    """
    Goes through a general set of comparisons for a molecule in increasing
    complexity. This will provide the most robust approach for identifying 
    if two molecules are duplicates. The default RMSD tolerance is set to a 
    reasonable, but relatively large value. Therefore, the default behavior is
    to remove confomers may be slightly different, but overall very similar.
    
    Comparison is as follows:
        1. Check formula
        2. Check graph structure of molecule
        3. Perform robust and efficient RMSD checking by
            (a) Attempt RMSD by taking advantage of properties of the principal
                inertial axes of the molecules
            (b) Calculate RMSD by considering all possible graphically symmetric
                labeling of the molecule 
    
    RMSD of molecules is a particularly difficult problem.  Although, there are
    parts that are fairly well defined. The Kabsch Algorithm will find the 
    optimal rotation between the atoms. However, this algorithm has a major
    issue. 
        - The result of the algorithm depends on the ordering of the atoms. 
           If the ordering of the atoms is not consistent, then the algorithm 
           will never provide a physically meaningful result and will likely 
           over-estimate the RMSD. 
        - Moreover, there is difficulty with atom ordering algorithms because 
           atoms that may be graphically symmetric. This creates a potentially 
           huge combinatorial problem from all possible index permutations. 
    
    This is overcome by a two step method that first uses alignment of the 
    principal axes of mol1 and mol2 in order to compute the RMSD. This algorithm 
    will find the RMSD by using the alignment of the principal axes as the 
    initial position to solve the linear sum assignment problem. The solution 
    to this problem may give atom orderings that match graphically symmetric 
    atoms provided that the input molecules have similar geometries. However, 
    this algorithm will not work in the case that there are large conformation 
    differences between the molecules. If such differences exist, the rmsd must
    be computed from all possible graphically symmetric orderings in order to 
    give the user the global minimum solution, which has been implemented in 
    compare_rmsd. 
    
    Chiral is not necessarily always if the molecule is chiral. It may also
    refer to the space group of the crystal structure that the molecules 
    come from. If the space is not chiral, meaning that it includes symmetry 
    operations with a determinant of -1, then chiral should be set to False. 
    This is because the notion of chiral is not a straightforward as presented
    in general organic chemistry courses. It is easily possible that through
    the combination of conformation differences of multiple side-groups of a 
    molecule that the overall 3D configuration of the molecule becomes chiral in
    the sense that for a geometry to perfectly overlay, it requires a symmetry
    operation with a determinant of -1 to be used. This is therefore not a 
    chiral center at a particular atom, but rather at the center of mass of 
    the molecule, and is known as lesser known name of inherant chirality. 
    Although, inherant chirality typically refers to a twisted backbone, the
    combination of "twisted" side-groups can lead to similar behavior as a 
    result of the full 3D conformation. 
    
    Returns
    -------
    (bool, [rotation0, rotation1], [atom_ordering0, atom_ordering1], chiral)
        Returns whether or not the molecules are duplicates, the rotation that
        brings them into the greatest amount of agreement, and the atom that
        brings the molecules into the greatest amount of agreement
        
    """
    ### Store default values
    geo1 = mol1.geometry
    geo2 = mol2.geometry
    com1 = com(mol1)
    com2 = com(mol2)
    ret_dups = False
    min_rmsd = -1
    min_trans = ((-com1,-com2))
    min_rot = (np.diag([1,1,1]),np.diag([1,1,1]))
    min_idx = ((np.arange(0,len(geo1)), np.arange(0,len(geo2))))
    
    ### 1. Check Formula
    f1 = mol1.formula
    f2 = mol2.formula
    
    if f1 != f2:
        return ret_dups,min_rmsd,min_trans,min_rot,min_idx,0
        
    ### 2. Check that graphs of molecules are the same
    graph_result = compare_graph(mol1, mol2, bonds_kw)
    if not graph_result:
        return ret_dups,min_rmsd,min_trans,min_rot,min_idx,0
    
    ### 3. Robust RMSD checking
    # (3a) Principal axes
    ret_dups,min_rmsd,min_trans,min_rot,min_idx,chiral = \
                        compare_rmsd_pa(mol1,mol2,rmsd_tol,bonds_kw,chiral,fixed_order)
    if ret_dups:
        ### Under the rmsd_tol, duplicates have already been found therefore
        ### the function may return here for maximal computational efficiency
        return ret_dups,min_rmsd,min_trans,min_rot,min_idx,chiral
    
    # (3b) All possible reorderings
    ret_dups,min_rmsd,min_trans,min_rot,min_idx,chiral = \
                        compare_rmsd(mol1,mol2,rmsd_tol,bonds_kw,chiral)
                        
    return ret_dups,min_rmsd,min_trans,min_rot,min_idx,chiral
    

def compare_rmsd(mol1, mol2, rmsd_tol=0.15,  bonds_kw={}, chiral=True, fixed_order=[]):
    """
    Obtains the global minimum rmsd between the input molecules by considering
    all graphically symmetric reordering of the atoms from mol1 and mol2
    
    Arguments
    ---------
    fixed_order: list
        List of lists that describes the symmetric ordering. If the molecular
        ordering is always identical for mol1 and mol2, then this can save
        significant time to pre-calculate if compare_rmsd is to be called
        many times. 
    
    Returns
    -------
    (bool, rmsd, [trans0,trans1], [rotation0, rotation1], [atom_ordering0, atom_ordering1])
        Returns whether or not the molecules are duplicates, the rotation that
        brings them into the greatest amount of agreement, and the atom that
        brings the molecules into the greatest amount of agreement
    
    """
    ### Copy and align molecules at origin
    geo1 = mol1.geometry
    ele1 = mol1.elements
    cmol1 = Structure(struct_id="copy_mol1", 
                      geometry=np.array(geo1), elements=ele1)
    cmol1.properties["bonds"] = mol1.properties["bonds"]
    
    geo2 = mol2.geometry
    ele2 = mol2.elements
    cmol2 = Structure(struct_id="copy_mol2", geometry=np.array(geo2), elements=ele2)
    cmol2.properties["bonds"] = mol2.properties["bonds"]
    
    com1 = com(cmol1)
    com2 = com(cmol2)
    cmol1.translate(-com1)
    cmol2.translate(-com2)
    
    pa1 = get_principal_axes(cmol1)
    pa2 = get_principal_axes(cmol2)
    
    geo1 = cmol1.geometry
    geo2 = cmol2.geometry
    
    geo1 = np.dot(pa1, geo1.T).T
    geo2 = np.dot(pa2, geo2.T).T
    
    ### Get the symmetric orderings of the given molecules
    if len(fixed_order) == 0:
        paths1 = get_symmetric_ordering(cmol1)
        paths2 = get_symmetric_ordering(cmol2)
    else:
        paths1 = fixed_order
        paths2 = fixed_order
    
    ### Choose the first ordering for mol1 and then check for mol2 orderings
    target = geo1[paths1[0]]
    
    ### Search over principal axes operations 
    diag0,diag1,diag2 = np.meshgrid([1,-1],[1,-1],[1,-1])
    diag = np.c_[diag0.ravel(), diag1.ravel(), diag2.ravel()]
    
    ### Initialize storage
    rmsd_list = []
    rot_list = []
    rmsd_path_list = []
    found = False
    for temp_diag in diag:
        temp_rot = np.diag(temp_diag)
        
        if np.linalg.det(temp_rot) == -1 and chiral:
            ### Skip operations that would changed the handed-ness
            continue
        
        ### Because the COM of the molecule has been centered at the origin, 
        ### which is also always commensurate with the principal axis of symmetry, 
        ### applying these symmetry operations is valid
        init_geo = np.dot(temp_rot, geo2.T).T
        
        for temp_path in paths2:
            temp_geo = init_geo[temp_path]
            
            ### Align geometries
            Rot,temp_rmsd = R.align_vectors(target, temp_geo)
            rot = Rot.as_matrix()
            temp_geo = np.dot(rot, temp_geo.T).T
            diff = target-temp_geo
            my_rmsd = np.sqrt((diff*diff).sum() / len(geo1))
            
            ### Store results
            rmsd_list.append(my_rmsd)
            rot_list.append(np.dot(rot, temp_rot))
            rmsd_path_list.append(temp_path)
            
            #### TEST PURPOSES SHOULD GIVE DIFF=0
            # tgeo2 = mol2.geometry - com2
            # tgeo2 = tgeo2[temp_path]
            # total_rot = np.dot(rot, np.dot(temp_rot, pa2))
            # total_geo = np.dot(total_rot, tgeo2.T).T
            
            # tgeo2 = np.dot(rot,np.dot(temp_rot, np.dot(pa2, tgeo2.T))).T
            # print(np.linalg.norm(temp_geo - total_geo), np.linalg.norm(temp_geo - tgeo2))
            
            ### Break if found
            if my_rmsd < rmsd_tol:
                found = True
                break
    
    identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    if found:
        ### Can just use last entries from loop
        order1 = paths1[0]
        order2 = temp_path
        min_rot = (pa1, np.dot(rot_list[-1], pa2))
        return True,my_rmsd,(-com1,-com2),min_rot,(order1,order2),0
    else:
        min_idx = np.argmin(rmsd_list)
        min_rmsd = rmsd_list[min_idx]
        min_path = rmsd_path_list[min_idx]
        min_rot = np.dot(rot_list[min_idx], pa2)
        order1 = paths1[0]
        order2 = min_path
        return False,min_rmsd,(-com1,-com2),(pa1,min_rot),(order1,order2),0
    
    
def compare_rmsd_pa(mol1, mol2, 
                    rmsd_tol=0.15,  
                    bonds_kw={}, 
                    chiral=False,
                    fix=True,
                    mp1=[],
                    mp2=[]):
    """
    Uses alignment of the principal axes of mol1 and mol2 in order to compute 
    the RMSD. In addition, there's the capability the mol1 is fixed in space 
    and the parameters are returned that will best match mol2 to mol1 providing
    the minimum possible rigid RMSD found by the algorithm. 
    
    Computing the RMSD for molecules includes certain difficulties. The most 
    notable of these is that the RMSD should only be computed between graphically
    symmetric atoms. This algorithm will find the RMSD by using the 
    alignment of the principal axes as the initial position to solve the 
    linear sum assignment problem. The linear sum assignment problem is solved 
    such that only comparisons between graphically symmetric sites may occur. 
    This gives a solution that should be either the global minimum or close 
    to it. For True global minimum computation, the compare_rmsd function 
    should be used where all graphically symmetric orderings are considered. 
    
    """
    geo1 = mol1.geometry
    ele1 = mol1.elements
    cmol1 = Structure(struct_id="copy_mol1", 
                      geometry=np.array(geo1), elements=ele1)
    
    geo2 = mol2.geometry
    ele2 = mol2.elements
    cmol2 = Structure(struct_id="copy_mol2", 
                      geometry=np.array(geo2), elements=ele2)
    
    cmol2_list = [cmol2]
    if not chiral:
        inv_geo2 = np.dot(np.diag([-1,-1,-1]), geo2.T).T
        inv_cmol2 = Structure(struct_id="copy_inv_mol2", 
                              geometry=inv_geo2, elements=ele2)
        cmol2_list.append(inv_cmol2)
    
    ### Get message passing results for each atom
    if len(mp1) == 0:
        mol1.get_bonds(**bonds_kw)
        g1 = mol2graph(cmol1)
        mp1 = message_passing_atom_descriptors(g1)
    if len(mp1) != len(geo1):
        raise Exception("Input for mp1 does not match number of atoms in geometry")
    mp1_list = [x for x in mp1.values()]
    
    if len(mp2) == 0:
        mol2.get_bonds(**bonds_kw)
        g2 = mol2graph(cmol2)
        mp2 = message_passing_atom_descriptors(g2)
    if len(mp2) != len(geo2):
        raise Exception("Input for mp2 does not match number of atoms in geometry")
    mp2_list = [x for x in mp2.values()]
    
    ### Build dictionary of allowed sites to compare for reordering
    mp1_array = np.array(mp1_list)
    mp2_array = np.array(mp2_list)
    unique_list = np.unique(mp1_array)
    unique_list2 = np.unique(mp2_array)
    if list(unique_list) != list(unique_list2):
        raise Exception("Molecules input to compare_rmsd_pa are not "+
                            "graphically symmetric")
    reorder_dict = {}
    lookup = {}
    for entry in unique_list:
        keep_idx_1 = np.where(mp1_array == entry)[0]
        keep_idx_2 = np.where(mp2_array == entry)[0]
        if len(keep_idx_1) != len(keep_idx_2):
            raise Exception("Molecules input to compare_rmsd_pa are not "+
                            "graphically symmetric")
        reorder_dict[entry] = [keep_idx_1, keep_idx_2]
        lookup[entry] = len(lookup)
    
    ### Storage for solution 
    solution_list = []
    rmsd_list = []
    rot_list = []
    idx_list = []
    chiral_list = []
    pa2_list = []
    com2_list = []
    for chiral_iter_idx,cmol2 in enumerate(cmol2_list):
        ### Store COM and PA so that it can be applied to best operation found
        com1 = com(cmol1)
        com2 = com(cmol2)
        cmol1.translate(-com1)
        cmol2.translate(-com2)
        
        pa1 = get_principal_axes(cmol1)
        pa2 = get_principal_axes(cmol2)
        
        geo1 = cmol1.geometry
        geo2 = cmol2.geometry
        
        if not fix:
            geo1 = np.dot(pa1, geo1.T).T
            geo2 = np.dot(pa2, geo2.T).T
        else:
            ### Rotate second molecule by principal axes
            geo2 = np.dot(pa2, geo2.T).T
            ### Then rotate second molecule by principal axes of mol2
            geo2 = np.dot(pa1.T, geo2.T).T 
            pa2 = np.dot(pa1.T, pa2)
        
        ### Search over principal axes operations 
        diag0,diag1,diag2 = np.meshgrid([1,-1],[1,-1],[1,-1])
        diag = np.c_[diag0.ravel(), diag1.ravel(), diag2.ravel()]

        for temp_diag in diag:
            temp_rot = np.diag(temp_diag)
            
            if np.linalg.det(temp_rot) == -1 and chiral:
                ### Skip operations that would changed the handed-ness
                continue
            
            temp_geo2 = np.dot(temp_rot, geo2.T).T
            
            ### Calculate solution to linear_sum_assignment for each matching
            ### symmetry unique locations
            idx1 = np.arange(0,len(geo1))
            idx2 = np.arange(0,len(geo2))
            iter_idx = 0
            for sym_idx_1,sym_idx_2 in reorder_dict.values():
                temp_sym_geo1 = geo1[sym_idx_1]
                temp_sym_geo2 = temp_geo2[sym_idx_2]
                temp_sym_dist = cdist(temp_sym_geo1,temp_sym_geo2)
                sridx1,sridx2 = linear_sum_assignment(temp_sym_dist)
                ### Dereference the reorders
                sridx1 = sym_idx_1[sridx1]
                sridx2 = sym_idx_2[sridx2]
                ### Store reorders in idx arrays allowing for global reordering of 
                ### indices
                end_idx = iter_idx + len(sym_idx_1)
                idx1[iter_idx:end_idx] = sridx1
                idx2[iter_idx:end_idx] = sridx2
                iter_idx += len(sym_idx_1)
                
            ### Check if solution produces graphically symmetric match
            temp_mp1 = [mp1_array[x] for x in idx1]
            temp_mp2 = [mp2_array[x] for x in idx2]
            if temp_mp1 != temp_mp2:
                # test_mp1 = [lookup[x] for x in temp_mp1]
                # test_mp2 = [lookup[x] for x in temp_mp2]
                # print([x for x in zip(test_mp1, test_mp2)])
                raise Exception("This should not be possible")
            else:
                temp_geo1 = geo1[idx1,:]
                temp_geo2 = temp_geo2[idx2,:]
                
            ### Perform final rotation matching
            Rot,temp_rmsd = R.align_vectors(temp_geo1, temp_geo2)
            rot = Rot.as_matrix()
            temp_geo2 = np.dot(rot, temp_geo2.T).T
            
            diff = temp_geo1-temp_geo2
            my_rmsd = np.sqrt((diff*diff).sum() / len(geo1))
            
            ### Store
            solution_list.append(1)
            rmsd_list.append(my_rmsd)
            rot_list.append(np.dot(rot, temp_rot))
            idx_list.append((idx1,idx2))
            chiral_list.append(chiral_iter_idx)
            pa2_list.append(pa2)
            com2_list.append(com2)
        
    # print(solution_list)
    # print(rmsd_list)
    
    ### Find best valid solution
    solution_list = np.array(solution_list)
    valid_idx = np.where(solution_list == 1)[0]
    if len(valid_idx) == 0:
        ### Return default values
        ret_dups = False
        min_rmsd = -1
        com2 = com2_list[0]
        min_trans = ((-com1,-com2))
        min_rot = (np.diag([1,1,1]),np.diag([1,1,1]))
        min_idx = ((np.arange(0,len(geo1)), np.arange(0,len(geo2))))
        return ret_dups,min_rmsd,min_trans,min_rot,min_idx,0
    
    rmsd_list = [rmsd_list[x] for x in valid_idx]
    rot_list = [rot_list[x] for x in valid_idx]
    idx_list = [idx_list[x] for x in valid_idx]
    
    min_idx = np.argmin(rmsd_list)
    chiral = chiral_list[min_idx]
    com2 = com2_list[min_idx]
    pa2 = pa2_list[min_idx]
    min_rmsd = rmsd_list[min_idx]
    if not fix:
        min_trans = ((-com1,-com2))
        min_rot = (pa1, np.dot(rot_list[min_idx], pa2))
    else:
        min_trans = ((np.array([0,0,0]), -com2 + com(mol1)))
        min_rot = (np.diag([1,1,1]), np.dot(rot_list[min_idx], pa2))
    min_idx = idx_list[min_idx]
    
    if min_rmsd < rmsd_tol:
        ret_dups = True
    else:
        ret_dups = False
        
    return ret_dups,min_rmsd,min_trans,min_rot,min_idx,chiral
    
    
def compare_graph(mol1, mol2, bonds_kw={}):
    """
    Compare if the graph of the input molecules are identical
    
    """
    mol1.get_bonds(**bonds_kw)
    mol2.get_bonds(**bonds_kw)
    
    ### Get message passing results for each atom
    g1 = mol2graph(mol1)
    g2 = mol2graph(mol2)
    mp1 = message_passing_atom_descriptors(g1)
    mp2 = message_passing_atom_descriptors(g2)
    
    ### Sort the messages by their hash values
    hl1 = []
    for _,message in mp1.items():
        hl1.append(hash(message))
    hl1 = np.array(hl1)
    sort_idx = np.argsort(hl1)
    hl1 = hl1[sort_idx]
    
    hl2 = []
    for _,message in mp2.items():
        hl2.append(hash(message))
    hl2 = np.array(hl2)
    sort_idx = np.argsort(hl2)
    hl2 = hl2[sort_idx]
    
    hl1 = list(hl1)
    hl2 = list(hl2)
    
    if hl1 != hl2:
        return False
    else:
        return True
    

def unique_by_formula(molecule_struct_list):
    """
    Just checks for duplicate formulas
    
    """
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
            
        formula1 = molecule1.formula()
        temp_list.append(molecule1)
        
        ## Compare with all others
        for idx2,molecule2 in enumerate(molecule_struct_list[idx1+1:]):
            ## Make sure to offset idx2 correctly
            idx2 += idx1+1
            formula2 = molecule2.formula()
            if formula1 == formula2:
                temp_list.append(molecule2)
                added_list.append(idx2)
        
        ## Now add in groups of molecules
        same_formula.append(temp_list)
    
    unique_molecules = []
    for same_list in same_formula:
        unique_molecules.append(same_list[0])
    
    return unique_molecules


def unique_by_rmsd(molecule_struct_dict):
    """
    Uses the RMSD of molecules in order to identify duplicates
    
    """
    
    
def generate_combined(mol1,mol2,results):
    """
    Results are from one of the compare functions
    
    """
    if len(results) != 6:
        raise Exception("Results must of the form: "+
            "(bool,rmsd,(trans,trans),(rot,rot),(idx,idx), chiral)")
    
    ### Avoid collisions
    collision = False
    if mol1.struct_id == mol2.struct_id:
        collision = True
        init_id = mol1.struct_id
        mol1.struct_id = "{}-1".format(mol1.struct_id)
        mol2.struct_id = "{}-2".format(mol2.struct_id)
    
    ### Apply inversion if chirality detected
    if results[5] > 0:
        geo2 = mol2.geometry
        geo2 = np.dot(np.diag([-1,-1,-1]), geo2.T).T
        ele2 = mol2.elements
        cmol2 = Structure(struct_id="inverse-{}".format(mol2.struct_id),
                          geometry=np.array(geo2), elements=ele2)
        cmol2.properties = dict(mol2.properties)
        mol2 = cmol2
            
    geo1 = mol1.geometry
    ele1 = mol1.elements
    geo2 = mol2.geometry
    ele2 = mol2.elements
    
    ### Reorder geometries
    geo1 = geo1[results[4][0]]
    ele1 = ele1[results[4][0]]
    geo2 = geo2[results[4][1]]
    ele2 = ele2[results[4][1]]
    
    ### Translate geometry to origin then rotate and translate back
    com1 = com(mol1)
    com2 = com(mol2)
    geo1 = geo1 - com1
    geo2 = geo2 - com2
    geo1 = np.dot(results[3][0],geo1.T).T
    geo2 = np.dot(results[3][1],geo2.T).T
    geo1 = geo1 + com1
    geo2 = geo2 + com2
    
    ### Translate geometry to final location
    geo1 = geo1 + results[2][0]
    geo2 = geo2 + results[2][1]

    temp_mol1 = Structure(struct_id=mol1.struct_id, 
                          geometry=np.array(geo1),elements=ele1)
    temp_mol2 = Structure(struct_id=mol2.struct_id, 
                          geometry=np.array(geo2),elements=ele2)
        
    comb_id = "Combined_{}_{}".format(mol1.struct_id,mol2.struct_id)
    comb_geo = np.vstack([geo1,geo2])
    comb_ele = np.hstack([ele1,ele2])
    comb = Structure(struct_id=comb_id,
                     geometry=comb_geo,elements=comb_ele)
    comb.properties["combined"] = {
        mol1.struct_id: temp_mol1.document(),
        mol2.struct_id: temp_mol2.document()
    }
    comb.properties["rmsd"] = results[1]
    
    ### Correct collission
    if collision:
        mol1.struct_id = init_id
        mol2.struct_id = init_id
    
    return comb
    