
import numpy as np

from scipy.spatial.transform import Rotation as R

from ase.data import atomic_numbers,atomic_masses_iupac2016
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as PGA

from mcse import Structure


def check_molecule(struct, exception=True):
    # Check for valid molecule_struct
    if len(struct.get_lattice_vectors()) > 0:
        if exception:
            raise Exception("Structure with lattice vectors {} was passed "
                .format(struct.get_lattice_vectors_better())+
                "into MoleculeBonding class. Molecule structure "+
                "without lattice vectors should be passed to "+
                "MoleculeBonding.")
        else:
            return False
    else:
        return True


def center_com(struct):
    """
    Move center of mass of the structure to the origin. 
    
    """
    temp_com = com(struct)
    struct.translate(-com)
    return struct


def com(struct):
    """
    Calculates center of mass of the system. 

    """
    geo_array = struct.get_geo_array()
    element_list = struct.elements
    mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]] 
                     for x in element_list]).reshape(-1)
    total = np.sum(mass)
    com = np.sum(geo_array*mass[:,None], axis=0)
    com = com / total
    return com


def rot_mol(rot, struct, wrt="origin", degrees=True, seq="xyz"):
    """
    Rotate molecule using rotation matrix. 
    
    Arguments
    ---------
    rot: array 
        Can be either a list of 3 euler angles in the given order or a 3,3 
        rotation matrix. 
    wrt: str
        Rotation performed with respect to any of the following options,
            ["origin", "com"]. Although, only origin implemented now.
    order: str
        Order for euler angles if rotation is given as euler angles. 
    """
    if wrt != "origin":
        raise Exception("Not Implemented")
    
    rot = np.array(rot)
    if rot.shape == (3,3):
        pass
    elif rot.ravel().shape == (3,):
        ### Assume euler angles
        Rot = R.from_euler(seq, rot.ravel(), degrees=degrees)
        rot = Rot.as_matrix()
    else:
        raise Exception(
            "Only rotation matrices and euler angles are currently implemented.")
        
    geo = struct.get_geo_array()
    ele = struct.elements
    
    rot_geo = np.dot(rot, geo.T).T
    
    struct.from_geo_array(rot_geo, ele)
    
    return struct


    
    
    
    
    
    