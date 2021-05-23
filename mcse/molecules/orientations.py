# -*- coding: utf-8 -*-



import numpy as np

from mcse.molecules.symmetry import get_rot_symmetry


def get_unique_angle_grid(mol, 
                          angle_spacing=15, 
                          max_angle=360,
                          max_rot=10,
                          tol=0.1):
    """
    Generates angles to use in a grid search for the given molecule. Angles
    that are symmetrically degenerate are automatically removed from 
    consideration. Cannot be used with an angle spacing less than 1.
    
    Arguments
    ---------
    mol: mcse.structure
        Molecule for which the grid should be generated
    angle_spacing: float
        Grid spacing to use for angles
    max_angles: iterable
        Maximum angle for each principal axis of the molecule
    max_rot: int
        Passed to get_symmetry. Maximum number of rotations to consider for
        finding the symmetry elements of the molecule. 
    tol: float
        Passed to get_symmetry. Tolerance for RMSD duplicate check.
    
    """
    ### Fail-safe
    if (max_angle / angle_spacing) > 360:
        raise Exception("That's not a good idea. Grid search is too large." +
            "Consider writing a parallel version of this function.")
    
    ### Only need rotational symmetry operations so call this directly
    rot_ops = get_rot_symmetry(mol,max_rot,tol,euler=True)
    
    angle_range = np.arange(0, max_angle, angle_spacing)
    angle1,angle2,angle3 = np.meshgrid(angle_range, 
                                       angle_range, 
                                       angle_range)
    angle_grid = np.c_[angle1.ravel(),
                       angle2.ravel(),
                       angle3.ravel()]
    
    ### Keep track of angles that have already been used with dictionary. 
    ### Dictionary has most efficient lookup speed and avoids pair-wise 
    ### comparison making this algorithm O(N) where N is the size of the 
    ### angle_grid rather than O(N^2)
    ### However, this code introduces a lot of branching. I don't know
    ### how expensive that makes it. 
    used_dict = {}
    unique_angle_grid = []
    for angle in angle_grid:
        unique = True
        fix_array = np.array([360,360,360])
        for rot in rot_ops:
            temp_angle = angle + rot
            mask = temp_angle >= 360
            temp_angle = temp_angle - fix_array*mask
            ### Turn into hashable tuple
            temp_angle = (int(temp_angle[0]), 
                          int(temp_angle[1]),
                          int(temp_angle[2]))
            if temp_angle in used_dict:
                unique = False
                break
            else:
                used_dict[temp_angle] = True
        if unique:
            unique_angle_grid.append(angle)
            
    return np.vstack(unique_angle_grid)