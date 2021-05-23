# -*- coding: utf-8 -*-
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

import vtk

from ase.data import atomic_numbers,atomic_masses_iupac2016

from mcse import Structure
from mcse.molecules.utils import *
from mcse.molecules.align import align
from mcse.molecules.orientations import get_unique_angle_grid
from mcse.plot.structures import Render,all_radii


class AlignedRender(Render):
    """
    Aligns the principals axis of the dimer in such a way that a good image 
    is created. 
    
    """
    def calc_struct(self, struct):
        self.struct = struct
        align(struct)
        self.adjust_viewing_angle(struct)
        
        self.window_size = self.get_window_size(struct, self.dpa)
        self.scale = self.get_scale(struct)
        self.extent = self.get_extent(struct)
        self.initialize_render(self.window_size, self.scale, self.extent)
        self.add_geometry(struct)
        self.add_close_interactions()
        
        if self.interactive == True:
            self.start_interactive()
            
        
    def adjust_viewing_angle(self, struct):
        geo = struct.get_geo_array()
        
        view_angle_x = R.from_euler("xyz", [-30,0,0], degrees=True)
        view_angle_y = R.from_euler("xyz", [0,-90,0], degrees=True)
        view_angle_z = R.from_euler("xyz", [0,0,180], degrees=True)
        
        # view_angle_x = R.from_euler("xyz", [0,0,0], degrees=True)
        # view_angle_y = R.from_euler("xyz", [0,-90,0], degrees=True)
        # view_angle_z = R.from_euler("xyz", [0,0,0], degrees=True) 
        
        # view_angle = view_angle_x.apply(view_angle_y)
        # geo = np.dot(geo, view_angle)
        geo = view_angle_x.apply(view_angle_z.apply(view_angle_y.apply(geo)))
        struct.from_geo_array(geo, struct.elements)
        

class AreaMaximizedRender(Render):
    """
    Aligns the principals axis of the dimer. Then performs a grid search to 
    find the viewing angle that maximizes the exposed area of the molecule in
    the x,y plane.
    
    """
    def calc_struct(self, struct):
        self.struct = struct
        align(struct)
        self.adjust_viewing_angle(struct)
        
        self.window_size = self.get_window_size(struct, self.dpa)
        self.scale = self.get_scale(struct)
        self.extent = self.get_extent(struct)
        self.initialize_render(self.window_size, self.scale, self.extent)
        self.add_geometry(struct)
        self.add_close_interactions()
        
        if self.interactive == True:
            self.start_interactive()
            
    def adjust_viewing_angle(self, struct):
        angle_grid = get_unique_angle_grid(
                          struct, 
                          angle_spacing=10, 
                          max_angle=90,
                          max_rot=10,
                          tol=0.1)
        geo = struct.get_geo_array()
        ele = struct.elements
        
        ### Need to get molecule_idx for molecular clusters
        molecule_idx = struct.get_molecule_idx(**self.bonds_kw)
        
        result_list = []
        for entry in angle_grid:
            rot_matrix = R.from_euler("xyz", entry, degrees=True).as_matrix()
            temp_geo = np.dot(rot_matrix,geo.T).T
            
            ### If it's a molecular custer, don't want to evaluate the entire image
            ### The correct thing to evaluate is the per-molecule area and 
            ### remove overlapping regions
            temp_mol_area = 0
            for mol_idx in molecule_idx:
                temp_mol_geo = temp_geo[mol_idx]
                temp_proj_min = np.min(temp_mol_geo[:,0:2], axis=0)
                temp_proj_max = np.max(temp_mol_geo[:,0:2], axis=0)
                temp_proj_area = (temp_proj_max[0] - temp_proj_min[0])*(
                                    temp_proj_max[1] - temp_proj_min[1])
                
                temp_mol_area += temp_proj_area
            
            result_list.append(temp_mol_area)
        
        best_rot_idx = np.argmax(result_list)
        final_angle = angle_grid[best_rot_idx]
        final_rot = R.from_euler("xyz", final_angle, degrees=True).as_matrix()
        
        geo = np.dot(final_rot, geo.T).T
        struct.from_geo_array(geo, ele)
        
        return final_rot
        
        
class AreaMinimizedRender(Render):
    """
    Aligns the principals axis of the dimer. Then performs a grid search to 
    find the viewing angle that maximizes the exposed area of the molecule in
    the x,y plane.
    
    """
    def calc_struct(self, struct):
        self.struct = struct
        align(struct)
        self.adjust_viewing_angle(struct)
        
        self.window_size = self.get_window_size(struct, self.dpa)
        self.scale = self.get_scale(struct)
        self.extent = self.get_extent(struct)
        self.initialize_render(self.window_size, self.scale, self.extent)
        self.add_geometry(struct)
        self.add_close_interactions()
        
        if self.interactive == True:
            self.start_interactive()
            
    def adjust_viewing_angle(self, struct):
        angle_grid = get_unique_angle_grid(
                          struct, 
                          angle_spacing=10, 
                          max_angle=360,
                          max_rot=10,
                          tol=0.1)
        geo = struct.get_geo_array()
        ele = struct.elements
        
        ### Need to get molecule_idx for molecular clusters
        molecule_idx = struct.get_molecule_idx(**self.bonds_kw)
        
        result_list = []
        for entry in angle_grid:
            rot_matrix = R.from_euler("xyz", entry, degrees=True).as_matrix()
            temp_geo = np.dot(rot_matrix,geo.T).T
            
            ### If it's a molecular custer, don't want to evaluate the entire image
            ### The correct thing to evaluate is the per-molecule area and 
            ### remove overlapping regions
            temp_mol_area = 0
            stored_rectangles = []
            for mol_idx in molecule_idx:
                temp_mol_geo = temp_geo[mol_idx]
                temp_proj_min = np.min(temp_mol_geo[:,0:2], axis=0)
                temp_proj_max = np.max(temp_mol_geo[:,0:2], axis=0)
                temp_proj_area = (temp_proj_max[0] - temp_proj_min[0])*(
                                    temp_proj_max[1] - temp_proj_min[1])
                
                #### Removing overlapping regions
                #### However, cases where multiple rectangles overlapped together 
                #### Are not accounted for
                # for entry in stored_rectangles:
                #     x1_min = entry[0][0]
                #     y1_min = entry[0][1]
                #     x1_max = entry[1][0]
                #     y1_max = entry[1][1]
                    
                #     dx = min(temp_proj_max[0], x1_max) - max(temp_proj_min[0], x1_min)
                #     dy = min(temp_proj_max[1], y1_max) - max(temp_proj_min[1], y1_min)
                    
                #     if (dx>=0) and (dy>=0):
                #         temp_proj_area -= dx*dy
                
                temp_mol_area += temp_proj_area
                stored_rectangles.append((temp_proj_min, temp_proj_max))
            
            # mol_area = np.max(temp_geo[:,0:2], axis=0) - \
            #             np.min(temp_geo[:,0:2], axis=0)
            # mol_area = -mol_area[0] + mol_area[1]
            
            result_list.append(temp_mol_area)
        
        best_rot_idx = np.argmin(result_list)
        final_angle = angle_grid[best_rot_idx]
        final_rot = R.from_euler("xyz", final_angle, degrees=True).as_matrix()
        
        geo = np.dot(final_rot, geo.T).T
        struct.from_geo_array(geo, ele)
        
        

class OverlappingClusters(AlignedRender):
    """
    For plotting of two overlapping clusters. Idea is that the  first cluster 
    will be plotted as a stick digram of the molecules colored by the atomic 
    element. The second cluster will be plotted as a ball and stick where the
    color of that atom corresponds to its distance from the equivalent position
    in the second structure. Using such an approach, it is possible to see 
    both the atomic species in the overlapping molecular clusters and the 
    approximate distance for each atom.
    
    Arguments
    ---------
    
    """
    def __init__(self, 
                 cluster1_vdw = 0.15,
                 cluster2_vdw = 0.15,
                 cmap = cm.viridis,
                 individual=False,
                 proj="",
                 atom_type="stick",
                 vmin=0,
                 vmax=2,
                 **kwargs):
        self.cluster1_vdw = cluster1_vdw
        self.cluster2_vdw = cluster2_vdw
        self.individual=individual
        self.cmap = cmap
        self.proj = proj
        self.vmin = vmin
        self.vmax = vmax
        
        if proj == "max":
            pass
        elif proj == "min":
            pass
        elif len(proj) > 0:
            raise Exception("Unrecognized projection argument")
        else:
            pass
        
        super().__init__(**kwargs)
        
        if atom_type == "stick":
            self.vdw_radii = [1 for x in self.vdw_radii]
        else:
            pass
        
        
    def calc_struct(self, struct, struct2=None):
        """
        Can pass either the structure as a combined structure. The combined
        information needs to be included in the properties of the structure
        in order for plotting to work. Alternatively, pass in the first 
        cluster as struct and the second cluster as struct2. 
        
        """
        if struct2 != None:
            self.combined = combine(struct, 
                                    struct2,
                                    bonds=True, 
                                    bonds_kw=self.bonds_kw)
        else:
            self.combined = struct
        
        self.struct = self.combined
        if self.proj == "max":
            rot = max_area_rot(self.struct, bonds_kw=self.bonds_kw)
        elif self.proj == "min":
            rot = min_area_rot(self.struct, bonds_kw=self.bonds_kw)
        elif len(self.proj) > 0:
            raise Exception("Unrecognized projection argument")
        else:
            rot = []
        
        self.rot = rot
        if len(rot) > 0:
            rot_mol(rot, self.struct)
        
        if "combined" not in self.struct.properties:
            raise Exception("Combined properties not found in input structure." +
                "Using overlapping clusters requires use of "+
                "mcse.molecules.render.combine with inputs of two "+
                "molecular clusters.")
            
        ### Usual stuff
        self.window_size = self.get_window_size(self.struct, self.dpa, self.vdw)
        self.scale = self.get_scale(self.struct)
        self.extent = self.get_extent(self.struct)
        self.initialize_render(self.window_size, self.scale, self.extent)
        
        cluster_id = [x for x in self.struct.properties["combined"].keys()]
        cluster_1_dict = self.struct.properties["combined"][cluster_id[0]]
        cluster_1 = Structure.from_dict(cluster_1_dict)
        if len(rot) > 0:
            rot_mol(rot, cluster_1)
            ### Save the rotated structure
            self.struct.properties["combined"][cluster_id[0]] = cluster_1.document()
            
        self.add_geometry(cluster_1, vdw=self.cluster1_vdw)
        
        cluster_2_dict = self.struct.properties["combined"][cluster_id[1]]
        cluster_2 = Structure.from_dict(cluster_2_dict)
        if len(rot) > 0:
            rot_mol(rot, cluster_2)
            ### Save the rotated structure
            self.struct.properties["combined"][cluster_id[1]] = cluster_2.document()
        
        ### Get colormapping wrt distances
        geo1 = cluster_1.get_geo_array()
        geo2 = cluster_2.get_geo_array()
        dist = np.linalg.norm(geo1 - geo2, axis=-1)
        
        if self.individual:
            # norm = mpl.colors.Normalize(vmin=np.min(dist), vmax=np.max(dist))
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            colors = self.cmap(norm(dist))[:,:-1]
        else:
            mol_idx_list = cluster_2.get_molecule_idx()
            mean_dist_list = []
            for mol_idx in mol_idx_list:
                temp_mean_dist = np.mean(dist[mol_idx])
                mean_dist_list.append(temp_mean_dist)
            norm = mpl.colors.Normalize(vmin=self.vmin, 
                                        vmax=self.vmax)
            colors = np.zeros((len(dist),3))
            dist = np.zeros((dist.shape))
            for idx,mol_idx in enumerate(mol_idx_list):
                temp_color = self.cmap(norm(mean_dist_list[idx]))
                colors[mol_idx] = temp_color[:-1]
                dist[mol_idx] = mean_dist_list[idx]
            
        self.norm = norm        
        self.add_geometry(cluster_2, vdw=self.cluster2_vdw, colors=colors)
        
        ### Add the colorbar: TO-DO
        
        if self.interactive == True:
            self.start_interactive()
    
    
    def matplotlib_colorbar(self, ax=None, label="RMSD"):
        if ax == None:
            fig = plt.figure(figsize=(8,1), constrained_layout=True)
            ax = fig.add_subplot(111)
        
        cb = mpl.colorbar.ColorbarBase(ax, 
                                cmap=self.cmap,
                                norm=self.norm,
                                orientation='horizontal')
        
        if len(label) > 0:
            cb.set_label(label, fontsize=16)
        
        return cb
        
            
        
        
def combine(struct1, struct2, lat=True, 
            bonds=True,
            bonds_kw={"mult": 1.20, "skin": 0.0, "update": False}):
    """
    Combines two structures. 
    
    Arguments
    ---------
    lat: bool
        If True, keeps lattice vectors of first structure. 
    """
    if bonds:
        bonds1 = copy.deepcopy(struct1.get_bonds(**bonds_kw))
        bonds2 = copy.deepcopy(struct2.get_bonds(**bonds_kw))

        ### Need to adjust index of bonds2 for combined structure
        adjust_idx = len(struct1.get_geo_array())
        for idx1,bond_list in enumerate(bonds2):
            for idx2,atom_idx in enumerate(bond_list):
                bonds2[idx1][idx2] = atom_idx + adjust_idx
        combined_bonds = bonds1 + bonds2 
        
    geo1 = struct1.get_geo_array()
    ele1 = struct1.elements
    
    combined = Structure.from_geo(geo1,ele1)
    
    if lat == True:
        lattice = struct1.get_lattice_vectors()
        if len(lattice) > 0:
            combined.set_lattice_vectors(lattice)
    
    geo2 = struct2.get_geo_array()
    ele2 = struct2.elements
    
    for idx,coord in enumerate(geo2):
        combined.append(coord[0],coord[1],coord[2],ele2[idx])
    
    combined.properties["combined"] = {
        struct1.struct_id: struct1.document(),
        struct2.struct_id: struct2.document()
        }
    
    if bonds:
        combined.properties["bonds"] = combined_bonds
        combined.get_bonds(**bonds_kw)
    
    return combined


def max_area_rot(struct, bonds_kw={"mult": 1.20, "skin": 0.0, "update": False}):
    """
    Find the rotation that maximizes the projected area wrt the molecules in 
    the structure
    
    """
    angle_grid = get_unique_angle_grid(
                      struct, 
                      angle_spacing=36, 
                      max_angle=360,
                      max_rot=10,
                      tol=0.1)
    geo = struct.get_geo_array()
    ele = struct.elements
    
    ### Need to get molecule_idx for molecular clusters
    molecule_idx = struct.get_molecule_idx(**bonds_kw)
    
    result_list = []
    for entry in angle_grid:
        rot_matrix = R.from_euler("xyz", entry, degrees=True).as_matrix()
        temp_geo = np.dot(rot_matrix,geo.T).T
        
        ### If it's a molecular custer, don't want to evaluate the entire image
        ### The correct thing to evaluate is the per-molecule area and 
        ### remove overlapping regions
        temp_mol_area = 0
        stored_rectangles = []
        for mol_idx in molecule_idx:
            temp_mol_geo = temp_geo[mol_idx]
            temp_proj_min = np.min(temp_mol_geo[:,0:2], axis=0)
            temp_proj_max = np.max(temp_mol_geo[:,0:2], axis=0)
            temp_proj_area = (temp_proj_max[0] - temp_proj_min[0])*(
                                temp_proj_max[1] - temp_proj_min[1])
            
            temp_mol_area += temp_proj_area
            stored_rectangles.append((temp_proj_min, temp_proj_max))
        
        result_list.append(temp_mol_area)
    
    best_rot_idx = np.argmax(result_list)
    final_angle = angle_grid[best_rot_idx]
    final_rot = R.from_euler("xyz", final_angle, degrees=True).as_matrix()
    
    return final_rot


def min_area_rot(struct, bonds_kw={"mult": 1.20, "skin": 0.0, "update": False}):
    """
    Find the rotation that minimizes the projected area wrt the molecules in 
    the structure
    
    """
    angle_grid = get_unique_angle_grid(
                      struct, 
                      angle_spacing=36, 
                      max_angle=360,
                      max_rot=10,
                      tol=0.1)
    geo = struct.get_geo_array()
    ele = struct.elements
    
    ### Need to get molecule_idx for molecular clusters
    molecule_idx = struct.get_molecule_idx(**bonds_kw)
    
    result_list = []
    for entry in angle_grid:
        rot_matrix = R.from_euler("xyz", entry, degrees=True).as_matrix()
        temp_geo = np.dot(rot_matrix,geo.T).T
        
        ### If it's a molecular custer, don't want to evaluate the entire image
        ### The correct thing to evaluate is the per-molecule area and 
        ### remove overlapping regions
        temp_mol_area = 0
        stored_rectangles = []
        for mol_idx in molecule_idx:
            temp_mol_geo = temp_geo[mol_idx]
            temp_proj_min = np.min(temp_mol_geo[:,0:2], axis=0)
            temp_proj_max = np.max(temp_mol_geo[:,0:2], axis=0)
            temp_proj_area = (temp_proj_max[0] - temp_proj_min[0])*(
                                temp_proj_max[1] - temp_proj_min[1])
            
            temp_mol_area += temp_proj_area
            stored_rectangles.append((temp_proj_min, temp_proj_max))
        
        result_list.append(temp_mol_area)
    
    best_rot_idx = np.argmin(result_list)
    final_angle = angle_grid[best_rot_idx]
    final_rot = R.from_euler("xyz", final_angle, degrees=True).as_matrix()
    
    return final_rot
        


if __name__ == "__main__":
    pass
    