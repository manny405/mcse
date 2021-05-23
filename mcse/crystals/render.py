# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R

from ase.data import atomic_numbers,atomic_masses_iupac2016

from mcse.molecules.utils import *
from mcse.molecules.align import align
from mcse.crystals.supercell import SupercellSphere,Supercell
from mcse.plot.structures import Render

import vtk

from mcse.plot.structures import OptimizedRender
class AlignedRender(OptimizedRender):
    """
    Aligns the images to be in the given direction. Default direction is the 
    (1,1,0) direction which provides a standard viewing angle for crystal 
    systems. 
    
    
    Arguments
    ---------
    viewing_adjustment: interable
        Euler angles of adjustment to be made after aligning the viewing 
        direction with the z axis. 
    custom: bool
        I've devised a custom viewing angle that works very well. 
    orient_y: iterable
        Which direction should be oriented in the [0,1,0] direction of the 
        render. This is equal to the direction that is pointing up in the 
        image. 
    
    
    """
    def __init__(self, 
                 viewing_direction=(1,-1,0), 
                 viewing_adjustment=(-45,0,0),
                 orient_y=(0,0,1),
                 custom=True,
                 **kwargs):
        self.viewing_direction = viewing_direction
        self.viewing_adjustment = viewing_adjustment
        self.custom = custom
        self.orient_y = orient_y
        super().__init__(**kwargs)
    
    
    def calc_struct(self, struct):
        self.struct = struct
        self.adjust_viewing_direction(struct)
        
        ### Decide if plotting unit cell
        if len(struct.get_lattice_vectors()):
            if self.unit_cell:
                unit_cell = True
                
        self.window_size = self.get_window_size(struct, self.dpa, self.vdw, unit_cell)
        self.scale = self.get_scale(struct, unit_cell=unit_cell)
        self.extent = self.get_extent(struct, unit_cell=unit_cell)
        self.initialize_render(self.window_size, self.scale, self.extent)
        
        self.add_geometry(struct)
        self.add_close_interactions()
        
        if len(self.struct.get_lattice_vectors()) > 0:
            self.add_unit_cell(np.vstack(struct.get_lattice_vectors()))
            self.add_unit_cell_labels(np.vstack(struct.get_lattice_vectors()))
        
        if self.interactive == True:
            self.start_interactive()
            
        
    def adjust_viewing_direction(self, struct, rot=[]):
        """
        When rotating crystal structure, have to also rotate the lattice 
        vectors of the structure. 
        
        """
        if len(struct.get_lattice_vectors()) > 0:
            lv = np.vstack(struct.get_lattice_vectors())
        else:
            ### No default viewing angle change can be provided if there are
            ### no lattice vectors
            return
            
        viewing_mask = np.array(self.viewing_direction).astype(bool)
        viewing_mult = np.array(self.viewing_direction)[viewing_mask][:,None]
        masked_lv = lv[viewing_mask] 
        masked_lv = masked_lv * viewing_mult
        viewing_dir = np.sum(masked_lv, axis=0)
        
        if self.custom: 
            ### Want 1a - 1b+0.5c+0.5a which is the the vector between the 
            ### a lattice vector and a face center of the unit cell
            viewing_dir = lv[0] - (lv[1] + 0.5*lv[-1] + 0.5*lv[0])
    
        ### Add orientation in the +y direction
        orient_mask = np.array(self.orient_y).astype(bool)
        orient_mult = np.array(self.orient_y)[orient_mask][:,None]
        orient_vector = lv[orient_mask] * orient_mult
        
        ### Add orientation to viewing direction requirements
        viewing_dir = np.vstack([viewing_dir,
                                 orient_vector])
        
        Rot,rmsd = R.align_vectors(
                    np.array([[0,0,1],[0,1,0]]),
                    viewing_dir)
        
        if len(rot) == 0:
            rot = Rot.as_matrix()
            
        geo = struct.get_geo_array()
        geo = np.dot(rot, geo.T).T
        
        if len(struct.get_lattice_vectors()) > 0:
            lv = np.dot(rot, lv.T).T
            struct.set_lattice(lv)
        
        struct.from_geo_array(geo, struct.elements)
        
        return rot
        

class SupercellRender(AlignedRender):
    """
    Render supercell outside of unit cell. 
    
    """
    def __init__(self, 
                 supercell_mult=(3,3,3),
                 standardize=True, 
                 track_molecules=True,
                 bonds_kw={}, 
                 **kwargs):
        
        self.supercell_driver = Supercell(mult=supercell_mult,
                 standardize=standardize,
                 track_molecules=track_molecules,
                 bonds_kw=bonds_kw)
        
        super().__init__(**kwargs)
        
    
    def calc_struct(self, struct):
        self.struct = struct
        self.supercell = self.supercell_driver.calc_struct(struct)
        self.supercell.set_lattice_vectors(self.struct.get_lattice_vectors())
        self.adjust_viewing_direction(self.supercell)
        
        ### Decide if plotting unit cell
        if len(struct.get_lattice_vectors()):
            if self.unit_cell:
                unit_cell = True
                
        self.window_size = self.get_window_size(self.supercell, 
                                                self.dpa, self.vdw, unit_cell)
        self.scale = self.get_scale(self.supercell, unit_cell=unit_cell)
        self.extent = self.get_extent(self.supercell, unit_cell=unit_cell)
        self.initialize_render(self.window_size, self.scale, self.extent)
        
        add_lv = np.vstack(self.supercell.get_lattice_vectors())
        self.supercell.lattice = []
        self.add_geometry(self.supercell)
        self.add_close_interactions(self.supercell)
        
        if len(self.struct.get_lattice_vectors()) > 0:
            self.add_unit_cell(add_lv)
            self.add_unit_cell_labels(add_lv)
            
        
        if self.interactive == True:
            self.start_interactive()
            
            
class MotifRender(AlignedRender):
    """
    Attempts to render the best possible angle for viewing the motif of 
    the structure. This is done by minimizing the projected area onto the 
    x,y perpendicular axes of viewing. 
    
    Argumnets
    ----------
    angle_spacing: float
        Spacing with which angles between 0 and 360 will be generated in order
        to minimize the viewed area of the structure. 
    
    """
    def __init__(self, 
                 supercell_mult=(3,3,3),
                 angle_spacing=15,
                 standardize=True, 
                 track_molecules=True,
                 bonds_kw={}, 
                 **kwargs):
        
        self.angle_spacing = angle_spacing
        self.supercell_driver = Supercell(mult=supercell_mult,
                 standardize=standardize,
                 track_molecules=track_molecules,
                 bonds_kw=bonds_kw)
        
        super().__init__(**kwargs)
        
    
    def calc_struct(self, struct):
        self.struct = struct
        
        lv = self.struct.get_lattice_vectors()
        if len(lv) > 0:
            self.supercell = self.supercell_driver.calc_struct(struct)
            self.supercell.set_lattice_vectors(self.struct.get_lattice_vectors())
            self.adjust_viewing_direction(self.supercell)
        else:
            self.supercell = self.struct
            self.adjust_viewing_direction(self.struct)
        
        ### Decide if plotting unit cell
        if len(struct.get_lattice_vectors()):
            if self.unit_cell:
                unit_cell = True
                
        self.window_size = self.get_window_size(self.supercell, 
                                                self.dpa, self.vdw, unit_cell)
        self.scale = self.get_scale(self.supercell, unit_cell=unit_cell)
        self.extent = self.get_extent(self.supercell, unit_cell=unit_cell)
        self.initialize_render(self.window_size, self.scale, self.extent)
        
        if len(lv) > 0:
            add_lv = np.vstack(self.supercell.get_lattice_vectors())
            self.supercell.lattice = []
        else:
            add_lv = []
            
        self.add_geometry(self.supercell)
        self.add_close_interactions(self.supercell)
        
        if len(self.struct.get_lattice_vectors()) > 0:
            self.add_unit_cell(add_lv)
            self.add_unit_cell_labels(add_lv)
        
        if self.interactive == True:
            self.start_interactive()
            
            
    def adjust_viewing_direction(self, struct):
        """
        Minimize the viewing area of the structure wrt the x,y projection
        
        """
        geo = struct.get_geo_array()
        ele = struct.elements
        
        angle_range = np.arange(0, 360, self.angle_spacing)
        angle1,angle2,angle3 = np.meshgrid(angle_range, 
                                           angle_range, 
                                           angle_range)
        angle_grid = np.c_[angle1.ravel(),
                           angle2.ravel(),
                           angle3.ravel()]
        
        self.angle_grid = angle_grid
        
        molecule_idx = struct.get_molecule_idx(**self.bonds_kw)
        area_list = []
        for angles in angle_grid:
            rot = R.from_euler("xyz", angles, degrees=True).as_matrix()
            temp_geo = np.dot(rot, geo.T).T
            
            ### Don't want to evaluate the entire image because that's 
            ### incorrect for a sheet structure with huge interplanar spacing
            ### The correct thing to evaluate is the per-molecule area
            temp_mol_area = 0
            for mol_idx in molecule_idx:
                temp_mol_geo = temp_geo[mol_idx]
                temp_proj_min = np.min(temp_mol_geo[:,0:2], axis=0)
                temp_proj_max = np.max(temp_mol_geo[:,0:2], axis=0)
                temp_proj_area = (temp_proj_max[0] - temp_proj_min[0])*(
                                    temp_proj_max[1] - temp_proj_min[1])
                temp_mol_area += temp_proj_area
            
            area_list.append(temp_mol_area)
        
        self.area_list = area_list
        
        min_angle_idx = np.argmin(area_list)
        
        final_angle = angle_grid[min_angle_idx]
        rot = R.from_euler("xyz", final_angle, degrees=True).as_matrix()
        geo = np.dot(rot, geo.T).T
        lv = struct.get_lattice_vectors()
        
        if len(lv) > 0:
            lv = np.vstack(lv)
            lv = np.dot(rot, lv.T).T
        else:
            pass
        
        ### Ensure that the horizontal/x direction is always the longest
        ### Don't care about the length of the Z direction
        length = np.max(geo, axis=0) - np.min(geo, axis=0)
        length_sort_idx = np.argsort(length[0:2]).astype(int)
        self.length = length
        if length_sort_idx[0] != 1:
            self.test=True
            ### Swap x and y
            length_rot = np.array([[0,1,0],[1,0,0],[0,0,1]])
            geo = np.dot(length_rot, geo.T).T
            
            if len(lv) > 0:
                lv = np.dot(length_rot, lv.T).T
            
        struct.from_geo_array(geo, ele)
        
        if len(lv) > 0:
            struct.set_lattice(lv)
         


if __name__ == "__main__":
    pass
    
    
    
    