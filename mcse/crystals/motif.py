
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

from mcse import Structure
from mcse import BaseDriver_
from mcse.core.utils import com
from mcse.molecules.rmsd import calc_rmsd
from mcse.molecules.symmetry import get_symmetry
from mcse.molecules.align import align,align_molecules_rot,get_principal_axes,rot_mol
from mcse.crystals.supercell import Supercell
from mcse.crystals import preprocess as crystal_preprocess



class Motif(BaseDriver_):
    """
    Identification of packing motifs in molecular crystal structures. 
    
    Arguments
    ---------
    molecule_plane_normal: iterable
        Vector with respect to the principal molecular axes  
        which describes the direction perpendicular to the plane of the 
        molecule. Any algorithm for finding this axis can be used outside
        of this Driver. However, it's typically the case that the desired 
        axis is the [1,0,0].
    supercell: int
        Supercell size to use for motif identification. 
    nn: int
        Number of nearest neighbors to use for motif identification. 
    sheet_angle_tolerance: float
        Tolerance of relative angle for identification of sheet packing motifs
        given in degrees. 
    sym_tol: float
        Tolerance to use for identification of molecular symmetry and for 
        molecular rmsd. 
    
    """
    def __init__(self, 
                 molecule_plane_normal=[1,0,0],
                 supercell_mult=(7,7,7),
                 nn=32, 
                 sheet_angle_tolerance=10,
                 sym_tol=0.1,
                 preprocess=True, 
                 bonds_kw={"mult": 1.20, "skin": 0.0, "update": False}):
        self.supercell_mult = supercell_mult
        self.nn = nn
        self.sheet_angle_tolerance = sheet_angle_tolerance
        self.sym_tol = sym_tol
        self.bonds_kw = bonds_kw
        self.preprocess = preprocess
        self.molecule_plane_normal = np.array(molecule_plane_normal)
        self.supercell_driver = Supercell(
                 mult=supercell_mult,
                 track_molecules=True,
                 bonds_kw=self.bonds_kw)
        
        
    def calc_struct(self, struct):
        self.struct = struct
        
        if self.preprocess:
            crystal_preprocess(self.struct)
        else:
            self.struct.get_molecule_idx(**self.bonds_kw)
            self.struct.molecules
            
        self.supercell = self.supercell_driver.calc_struct(self.struct)
        
        geo = self.supercell.get_geo_array()
        ele = self.supercell.elements
        lv = np.vstack(self.struct.get_lattice_vectors())
        self.supercell_molecules = []
        for mol_idx in self.supercell.properties["molecule_idx"]:
            temp_mol = Structure.from_geo(geo[mol_idx], ele[mol_idx])
            self.supercell_molecules.append(temp_mol)
        
        self.supercell_com = np.vstack([com(x) for x in self.supercell_molecules])
        
        ### Get molecule at the center of the unit cell
        com_dist = np.linalg.norm(self.supercell_com - np.mean(lv, axis=0), axis=-1)
        center_idx = np.argmin(com_dist)
        self.center_idx = center_idx
        
        self.com_dist = cdist(self.supercell_com[self.center_idx][None,:],
                              self.supercell_com)[0]
        
        self.com_dist_sort_idx = np.argsort(self.com_dist)
        
        #### START HERE
        #### Finding nearest neighbors and the rotation matrix between nearest 
        #### neighbors. 
        ### Skip first because it's the same molecule
        self.nn_idx = self.com_dist_sort_idx[1:self.nn+1]
        
        #### Finding relative orientation with respect to the given 
        #### Molecular plane normal
        self.rot_list = []
        self.center_mol = self.supercell_molecules[self.center_idx]
        for idx in self.nn_idx:
            self.temp_mol = self.supercell_molecules[idx]
            ### Want the rotation that turns the center mol into the nearest
            ### neighbor
            temp_rot = align_molecules_rot(self.temp_mol, self.center_mol)
            self.rot_list.append(temp_rot)
        
        
        ### Need to transform the rotation from the PA coordinate system
        ### Into the cartesian/xyz coordinate system so that euler angles
        ### can be easily analyzed
        center_com = com(self.center_mol)
        self.center_mol.translate(-center_com)
        self.pa = get_principal_axes(self.center_mol)
        self.center_mol.translate(center_com)
        
        
        #### Maybe it's easier to transform the whole system such that the 
        #### PA of the center molecule is already aligned with x,y,z
        self.test_center_mol = copy.deepcopy(self.center_mol)
        center_com = com(self.test_center_mol)
        self.test_center_mol.translate(-center_com)
        self.test_pa = get_principal_axes(self.test_center_mol)
        rot_mol(self.test_pa, self.test_center_mol)
        self.test_trans_pa = get_principal_axes(self.test_center_mol)
        
        #### Rotate entire system wrt PA of center mol
        for idx,temp_mol in enumerate(self.supercell_molecules):
            if idx == self.center_idx:
                continue
            ### Rotate according to center molecule principal axes
            temp_mol.translate(-center_com)
            rot_mol(self.test_pa, temp_mol)
        
        #### Finding relative orientation with respect to the given 
        #### Molecular plane normal in the rotated system
        self.test_rot_list = []
        self.center_mol = self.supercell_molecules[self.center_idx]
        for idx in self.nn_idx:
            self.temp_mol = self.supercell_molecules[idx]
            ### Want the rotation that turns the center mol into the nearest
            ### neighbor
            temp_rot = align_molecules_rot(self.temp_mol, self.test_center_mol)
            self.test_rot_list.append(temp_rot)
    
        
        self.euler_list = []
        for entry in self.test_rot_list:
            temp_R = R.from_matrix(entry)
            ### Make sure to use "XYZ" instead of "xyz" for intrinsic rotation
            temp_euler = temp_R.as_euler("XYZ", degrees=True)
            self.euler_list.append(temp_euler)
        
        self.euler_list = np.vstack(self.euler_list)
        self.motif = self.motif_definitions(self.euler_list, 
                                            self.com_dist[self.nn_idx],
                                            self.molecule_plane_normal)
        
        self.struct.properties["Motif"] = self.motif

        ### Store max relative angle 
        self.struct.properties["Max_Angle"] = np.max(self.euler_list[:,[1,2]])
        self.struct.properties["Max_Plane_Angle"] = np.max(self.euler_list[:,0])
        
        return self.motif
        
    
    
    def motif_definitions(self, euler, com_dist, molecule_plane_normal):
        """
        Assigns defintion of the motif given the euler angles and COM 
        distances for a cluster of molecules. 
        
        Implementing general molecular plane normal would be as easy as 
        rotating the system such that the given molecular plane normal is in 
        the [1,0,0] direction. Then, analysis of the euler & com_dist holds
        for this system. Just need to test that this is the case... But I 
        don't have any test cases. 
        
        """
        if molecule_plane_normal[0] != 1:
            if np.sum(molecule_plane_normal) > 1:
                raise Exception("Molecular plane normal must be [1,0,0]." +
                        "If this feature is needed, send me the test case so "
                        +"this can be implemented.")
        
        ### For motif identification, it doesn't matter if the molecule is 
        ### rotated wrt the molecular plane normal axis
        
        ### Also, it will be assumed that a rotation of the molecule by 180 
        ### degrees yields the same motif definiton
        correct_euler_idx_1 = np.where(np.abs(np.abs(euler[:,1]) - 180) < np.abs(euler[:,1]))
        correct_euler_idx_2 = np.where(np.abs(np.abs(euler[:,2]) - 180) < np.abs(euler[:,2]))
        
        euler[:,1][correct_euler_idx_1] = np.abs(euler[:,1][correct_euler_idx_1]) - 180
        euler[:,2][correct_euler_idx_2] = np.abs(euler[:,2][correct_euler_idx_2]) - 180
        
        ### Only interested in absolute rotations from here
        euler = np.abs(euler)
        
        ### Sheet definition: Tolerance is angle of 3
        rotated_idx = np.where(
            np.logical_or(
                            euler[:,1] > self.sheet_angle_tolerance,
                            euler[:,2] > self.sheet_angle_tolerance))[0]

        if len(rotated_idx) == 0:
            return "sheet"
        else:
            return "gamma"
    
        
        
        
        



if __name__ == "__main__":
    pass
    
        
    
