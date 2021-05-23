# -*- coding: utf-8 -*-
import time,copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ase.data import atomic_numbers,atomic_masses_iupac2016

from mcse import Structure
from mcse.core.utils import com
    

def get_symmetry(mol, max_rot=10, tol=0.1, pairwise=False):
    """
    Finds all rotational elements of a molecule. Does not necessarily consider 
    all of the combinations of the symmetry elements. 
    
    Arguments
    ---------
    mol: mcse.structure
        Structure object to find symmetry operations
    max_rot: int
        Maximum number of rotations to consider for finding the symmetry 
        elements of the molecule. 
    tol: float
        Tolerance for RMSD duplicate check.
    pairwise: bool
        If the pairwise operation of all symmetry elements should be computed
        before returning the symmetry operation matrices.
    
    """
    mol_com = com(mol)
    if np.linalg.norm(mol_com) > 1e-4:
        mol = copy.deepcopy(mol)
        mol.translate(-mol_com)
    
    all_op = np.vstack([get_rot_symmetry(mol,max_rot,tol), 
                        get_mirror_symmetry(mol,tol)])
    #### Make all zeros positive
    all_op += 0
    all_op = np.round(all_op, decimals=4)
    unique = np.unique(all_op,axis=0)
    if not pairwise:
        return unique
    else:
        all_pairwise = []
        total = 0 
        for idx1,entry1 in enumerate(unique):
            all_pairwise.append(entry1)
            for entry2 in unique[idx1+1:]:
                temp_result = np.dot(entry1,entry2)
                all_pairwise.append(temp_result)
                
        all_pairwise = np.round(np.array(all_pairwise), decimals=2)
        all_pairwise = np.unique(all_pairwise, axis=0)
        
        return all_pairwise
    

def get_rot_symmetry(mol, max_rot=10, tol=0.1, euler=False):
    """
    Gets symmetry operations of the molecule using a maximum principal value
    for the principal rotational axis. Default value is 10. The current
    implementation is not as fast as it could be. There's no reason to 
    make a molecule for each entry in the orientation_dict. This was
    just useful to copy from DGS for now. However, one still needs to 
    account for the elements matching up so this cannot be completely
    neglected. 
    
    """
    mol_com = com(mol)
    
    if np.linalg.norm(mol_com) > 1e-4:
        mol = copy.deepcopy(mol)
        mol.translate(-mol_com)
        
    grid_spacing = int(360/max_rot)
    angle_range = np.arange(0, 360, grid_spacing)
    angle1,angle2,angle3 = np.meshgrid(angle_range, 
                                       angle_range, 
                                       angle_range)
    
    angle_grid = np.c_[angle1.ravel(),
                       angle2.ravel(),
                       angle3.ravel()]
    
    ### Make sure geo is aligned with principal axes before rotations
    temp_mol = copy.deepcopy(mol)
    ele = temp_mol.elements
    for i in range(10):
        geo = temp_mol.get_geo_array()
        principal_axes = get_principal_axes(temp_mol)
        rot = principal_axes.T
        geo = np.dot(rot, geo.T).T
        temp_mol.from_geo_array(geo,ele)
    geo = temp_mol.get_geo_array()
    
    
    orientation_dict = {}
    save_angle_dict = {}
    for rot_vector in angle_grid:
        r = R.from_euler('xyz', rot_vector, degrees=True)
        rot = r.as_matrix()
        
        temp_geo = np.dot(rot, geo.T).T
        
        temp_orientation = Structure.from_geo(temp_geo, ele)
        temp_orientation.struct_id = "{}_{}_{}".format(
                rot_vector[0],
                rot_vector[1], 
                rot_vector[2])
        
        orientation_dict[temp_orientation.struct_id] = temp_orientation
        
        if not euler: 
            save_angle_dict[temp_orientation.struct_id] = rot
        else:
            save_angle_dict[temp_orientation.struct_id] = rot_vector
    
    sym_ops = []
    for struct_id,struct in orientation_dict.items():
        if calc_rmsd_ele(temp_mol, struct, tol):
            sym_ops.append(save_angle_dict[struct_id])
        
        ### Original doesn't account for different elements may overlap
        # temp_rmsd = calc_rmsd(mol, struct)
        # if temp_rmsd <= tol:
        #     sym_ops.append(save_angle_dict[struct_id])
    
    #### Make all zeros positive
    sym_ops = np.stack(sym_ops)
    sym_ops += 0
    sym_ops = np.round(sym_ops, decimals=4)
    unique = np.unique(sym_ops,axis=0)
    unique += 0
    
    return unique
    
    
def get_mirror_symmetry(mol, tol=0.1):
    """
    Mirror symmetry and inversion symmetry is much easier to test than
    rotational symmetry. I'm not interested to fully implement this right now, 
    but it would be very easy to do so. This method does not consider mirror
    symmetries that are not along the diagonal of the matrix. This may come 
    later from combinations of symmetry elements of the molecule instead.
    
    Another good idea. Symmetry reduction in the search space can be easily
    performed for any algorithm by just creating a database of systems that 
    have already been calculated and then checking if the next sample is
    identical to any previous sample. This search/checking can be sped up
    if you already know the symmetry of the system, but this is not necessarily 
    a requirement if a general dup check algorithm is used. 
    
    """
    mol_com = com(mol)
    if np.linalg.norm(mol_com) > 1e-4:
        mol = copy.deepcopy(mol)
        mol.translate(-mol_com)
        
    ### Generate all possible mirror symmetries including inversion center. 
    values = np.array([1,-1])
    value1,value2,value3 = np.meshgrid(values, 
                                       values, 
                                       values)
    
    sym_diag = np.c_[value1.ravel(),
                       value2.ravel(),
                       value3.ravel()]
    
    
    ### Make sure geo is aligned with principal axes before rotations
    temp_mol = copy.deepcopy(mol)
    ele = temp_mol.elements
    for i in range(10):
        geo = temp_mol.get_geo_array()
        principal_axes = get_principal_axes(temp_mol)
        rot = principal_axes.T
        geo = np.dot(rot, geo.T).T
        temp_mol.from_geo_array(geo,ele)
    geo = temp_mol.get_geo_array()
    
    orientation_dict = {}
    save_angle_dict = {}
    for entry in sym_diag:
        temp_rot = np.identity(3)
        np.fill_diagonal(temp_rot, entry)
        
        temp_geo = np.dot(geo, temp_rot)
        temp_orientation = Structure.from_geo(temp_geo, ele)
        temp_orientation.properties["rot"] = temp_rot
        temp_orientation.struct_id = "{}_{}_{}".format(
                entry[0],
                entry[1], 
                entry[2])
        
        orientation_dict[temp_orientation.struct_id] = temp_orientation
        save_angle_dict[temp_orientation.struct_id] = temp_rot
        
    sym_ops = []
    for struct_id,struct in orientation_dict.items():
        if calc_rmsd_ele(temp_mol, struct, tol):
            sym_ops.append(save_angle_dict[struct_id])     
            
    #### Make all zeros positive
    sym_ops = np.stack(sym_ops)
    sym_ops += 0
    sym_ops = np.round(sym_ops, decimals=4)
    unique = np.unique(sym_ops,axis=0)
    unique += 0
    
    return unique


def calc_rmsd_ele(struct1,struct2,tol=0.1):
    """
    Calculates the rmsd with an additional check that the optimal ordering
    of the indices must lead to an identical ordering of elements. If it 
    does not, then the RMSD value should be neglected as incorrect and False 
    is returned.
    
    """
    geo1 = struct1.get_geo_array()
    ele1 = struct1.elements
    
    geo2 = struct2.get_geo_array()
    ele2 = struct2.elements
    
    dist = cdist(geo1,geo2)
    
    idx1,idx2 = linear_sum_assignment(dist)
    
    geo1 = geo1[idx1]
    geo2 = geo2[idx2]
    
    rmsd = np.mean(np.linalg.norm(geo1 - geo2,axis=-1))
    
    ### Find that the elements match easily using list comparison
    ele1 = ele1[idx1].tolist()
    ele2 = ele2[idx2].tolist()
    
    if ele1 != ele2:
        return False
    
    if rmsd < tol:
        return True
    else:
        return False
    
    
def get_principal_axes(struct):
    """
    Obtain principal axes of inertia for the given structure. 
    
    """
    inertial_tensor = get_inertia_tensor(struct)
    ### SVD Decomposition to get axes of principal axes
    u,s,vh = np.linalg.svd(inertial_tensor)
    principal_axes = u
    return principal_axes


def get_inertia_tensor(struct):
    """
    Calculates inertia tensor of the given structure. 
    
    """
    geo = struct.get_geo_array()
    ele = struct.elements
    mass =  np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele])
    
    inertial_tensor = np.zeros((3,3))
    ## Handle diangonal calculations first
    inertial_tensor[0,0] = np.sum(mass*(np.square(geo[:,1]) + np.square(geo[:,2])))
    inertial_tensor[1,1] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,2])))
    inertial_tensor[2,2] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,1])))
    
    ## Handle off diagonal terms
    inertial_tensor[0,1] = -np.sum(mass*geo[:,0]*geo[:,1])
    inertial_tensor[1,0] = inertial_tensor[0,1]
    
    inertial_tensor[0,2] = -np.sum(mass*geo[:,0]*geo[:,2])
    inertial_tensor[2,0] = inertial_tensor[0,2]
    
    inertial_tensor[1,2] = -np.sum(mass*geo[:,1]*geo[:,2])
    inertial_tensor[2,1] = inertial_tensor[1,2]
    
    total_inertia = np.dot(mass,np.sum(np.square(geo), axis=-1))
    inertial_tensor = inertial_tensor / total_inertia
    
    return inertial_tensor
    