# -*- coding: utf-8 -*-

import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from mcse import Structure
from mcse.molecules import align,rot_mol,fast_align
from mcse.molecules.symmetry import get_symmetry


def rmsd(struct1, struct2, pre_align=True, pre_fast_align=True):
    """
    For the input molecules, attempts to align the structures as best as 
    possible. Returns the aligned structures, the rmsd, and the rotation matrix
    that aligns them. Note that this does not take into account that the 
    elements of molecules will match in the best geometric overlap.

    """
    if pre_align:
        ### Fast align is chosen by default to improve efficiency. Rigorous 
        ### alignment wrt symmetry will significantly increase RMSD time
        if pre_fast_align:
            fast_align(struct1)
            fast_align(struct2)
        else:
            align(struct1)
            align(struct2)
    
    geo1 = struct1.geometry
    geo2 = struct2.geometry
    
    dist = cdist(geo1,geo2)
    idx1,idx2 = linear_sum_assignment(dist)
    geo1 = geo1[idx1,:]
    geo2 = geo2[idx2,:]

    Rot,rmsd = R.align_vectors(geo1,geo2)
    rot = Rot.as_matrix()
    struct2.rotate(rot)
    geo2 = np.dot(rot, geo2.T).T

    ### Perform linear_sum_assignment again to get best match of final rotated
    ### system
    dist = cdist(geo1,geo2)
    idx1,idx2 = linear_sum_assignment(dist)
    geo1 = geo1[idx1]
    geo2 = geo2[idx2]
    result_rmsd = np.mean(np.linalg.norm(geo1 - geo2,axis=-1)) / len(geo1)

    return struct1,struct2,rot,result_rmsd


def calc_rmsd_rot(struct1, struct2, angle_grid=[], max_rot=10, 
                  pre_align=True, pre_fast_align=True):
    """
    Computes rmsd of the molecules. Considers symmetry operations of the 
    molecules and returns the smallest rmsd found. 
    
    In fact, symmetry operations don't make any sense to consider because of
    course the geometries of the molecules will remain unchanged upon 
    the application of their symmetry operations. What really needs to be 
    considered are differen mirror operations in order to start from different
    initial positions to account for discrepencies of aligning the same 
    molecule by the alignment algorithm.
    
    Actually, mirror operations are also not correct because then you cannot 
    identify if the difference is due to chirality. Therefore, rotations are 
    actually what should be used. 
    
    Arguments
    ---------
    max_rot: int
        For searching over rotations of the molecule. 
    tol: float
        For obtaining symmetry of molecule. 
    
    """
    if pre_align:
        ### Fast align is chosen by default to improve efficiency. Rigorous 
        ### alignment wrt symmetry will significantly increase RMSD time
        if pre_fast_align:
            fast_align(struct1)
            fast_align(struct2)
        else:
            align(struct1)
            align(struct2)
    
    if len(angle_grid) == 0:
        grid_spacing = int(360/max_rot)
        angle_range = np.arange(0, 360, grid_spacing)
        angle1,angle2,angle3 = np.meshgrid(angle_range, 
                                           angle_range, 
                                           angle_range)
        
        angle_grid = np.c_[angle1.ravel(),
                           angle2.ravel(),
                           angle3.ravel()]
    
    angle_grid = np.array(angle_grid)
    
    result_rmsd = []
    for rot_vector in angle_grid:
        rot_vector = np.array(rot_vector)
        
        if rot_vector.shape != (3,3):
            r = R.from_euler('xyz', rot_vector, degrees=True)
            temp_rot = r.as_matrix()
            temp_inv_rot = np.linalg.inv(temp_rot)
        else:
            ### Already a rotation matrix
            temp_rot = rot_vector
            temp_inv_rot = np.linalg.inv(temp_rot)
        
        ### It's okay to rotate base structure because the inverse will be 
        ### applied after. 
        struct2 = rot_mol(temp_rot, struct2)
        
        temp_rmsd = calc_rmsd_ele(struct1, struct2)
        result_rmsd.append(temp_rmsd)
        
        ### Apply inverse to get back to original molecules
        struct2 = rot_mol(temp_inv_rot, struct2)
    
    min_rmsd = np.min(result_rmsd)
    
    ### Apply final rotation to struct2 so that it has the geometry with the 
    ### best agreement when it is return from this function
    min_rot_idx = np.argmin(result_rmsd)
    min_rot = angle_grid[min_rot_idx]
    
    if min_rot.shape != (3,3):
        r = R.from_euler('xyz', min_rot, degrees=True)
        temp_rot = r.as_matrix()
    else:
        temp_rot = angle_grid[min_rot_idx]
        
    struct2 = rot_mol(temp_rot, struct2)
    
    return min_rmsd


def compare_rmsd(struct1, struct2, angle_grid=[], tol=0.1, max_rot=10):
    """
    Turns calc_rmsd_symmetry into comparison function. 
    
    """
    struct1,struct2,rot,result_rmsd = rmsd(struct1, struct2)
    if result_rmsd < tol:
        return True
    else:
        return False

    
def compare_rmsd_rot(struct1, struct2, angle_grid=[], tol=0.1, max_rot=10):
    """
    Turns calc_rmsd_symmetry into comparison function. 
    
    """
    result_rmsd = calc_rmsd_rot(struct1, 
                                struct2, 
                                angle_grid=angle_grid, 
                                max_rot=max_rot)
    if result_rmsd < tol:
        return True
    else:
        return False


def calc_rmsd(struct1, struct2):
    """
    Basic rmsd calculator for molecules and molecular clusters. 

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
    
    return rmsd

def calc_rmsd_ele(struct1,struct2):
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
        return 1000
    else:
        return rmsd
    # if rmsd < tol:
    #     return True
    # else:
    #     return False
    
def compare_rmsd_ele(struct1, struct2, tol=0.1):
    """
    Turns calc_rmsd_ele into comparison function. 
    
    """
    result_rmsd = calc_rmsd_ele()
    
    if result_rmsd < tol:
        return True
    else:
        return False
    
    
    
    
    
    
    
    
