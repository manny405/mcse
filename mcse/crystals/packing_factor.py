# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree 

from ase.data import vdw_radii,atomic_numbers,covalent_radii
from ase.data.vdw_alvarez import vdw_radii as vdw_radii_alvarez

from mcse import Structure
from mcse import BaseDriver_
from mcse.crystals.supercell import SupercellSphere


all_radii = []
for idx,value in enumerate(vdw_radii):
    if np.isnan(value):
        ## Approximate
        value = covalent_radii[idx]+0.8
    else:
        if not np.isnan(vdw_radii_alvarez[idx]):
            alvarez_value = vdw_radii_alvarez[idx]
            value = np.min([value, alvarez_value])
    all_radii.append(value)
all_radii = np.array(all_radii)
## 1.1 is more appropriate for molecular crystal structures, particularly 
## when hydrogen bonds are present. 
all_radii[1] = 1.1

all_radii_dict = {}
for key,value in atomic_numbers.items():
    if value >= len(all_radii):
        continue
    all_radii_dict[key] = all_radii[value]


class PackingFactor(BaseDriver_):
    """
    Calculatees the geometric packing factor using the vdW radii of the 
    structure and a user specified grid spacing. The algorithm is as follows:
        1. Generate Supercell of user specified size. This is to ensure that all
           of the necessary atoms are within the unit cell.
        2. Keep only atoms that are within the unit cell plus a correction 
           equal to the largest van der Waals radius in the system.
        3. Generate a grid using the specified grid spacing. This is done by 
           computing the grid spacing that should be used based on the lattice
           vector norm in every direction. Then, this grid is generated in 
           fraction space and converted finally to real space. 
        5. For each location of the grid spacing, calculate how far it is from 
           each atom. 
        6. For the distance to each atom, divide by the vdW radius of  the
           respective atom. 
        7. All values less than 1 are occupied and all values greater than 1 
           are considered empty. 
        8. Divide filled by total to get the packing factor.
        
    Arguments
    ---------
    spacing: float
        Spacing is given in Angstroms
    vdw: iterable
        Iterable that can be indexed where the index is equal to the atomic 
        number and the value at the index is the radius to use. 
    supercell_mult: int
        Size of supercell to build in every lattice direction.
    low_memory: bool
        If True, an implementation that requires a smaller, fixed amount of 
        system memory is used at the expense of additional compute time. This
        should be set to True if the user would like to use grid spacings
        below 0.25 Angstrom.
    
    """
    def __init__(self, 
                 spacing=0.25, 
                 vdw=all_radii, 
                 supercell_mult=1,
                 low_memory=False):
        self.spacing = spacing
        self.vdw = vdw
        self.supercell_mult = supercell_mult
        self.supercell = SupercellSphere(mult=self.supercell_mult)
        self.low_memory = low_memory
        ### Batch size not directly exposed user as argument to API cleaner.
        ### Change if you're confident.
        self.batch_size = 25000
    
    
    def calc_struct(self, struct):
        self.struct = struct
        self.supercell_struct = self.supercell.calc_struct(struct)
        
        self.lat = np.vstack(self.struct.get_lattice_vectors())
        self.norm = np.linalg.norm(self.lat,axis=-1)
        self.linv = np.linalg.inv(self.lat)
        
        self.unit_cell_struct = self.keep_inside(self.supercell_struct)
        
        self.num = self.norm / self.spacing
        self.grid_frac = self.generate_grid(self.num)
        self.grid_cart = np.dot(self.grid_frac, self.lat)
        
        if self.low_memory == False:
            ### Compute pairwise distances with unit cell modified structure
            dist = cdist(self.grid_cart, self.unit_cell_struct.get_geo_array())
            
            ### Divide dist by vdW radius
            ele = self.unit_cell_struct.elements
            self.vdw_array = [self.vdw[atomic_numbers[x]] for x in ele]
            self.vdw_array = np.array(self.vdw_array)[None,:]
            dist = dist / self.vdw_array
            self.min_dist = np.min(dist, axis=-1)
            self.occupied_idx = np.where(self.min_dist < 1)[0]
            
            ## Exact definition of packing factor
            packing_factor = np.where(self.min_dist < 1)[0].shape[0] / \
                                        self.min_dist.shape[0]
        else:
            #### Certainly a batch size of 25000 should be possible for 
            #### reasonably size molecular cryestal structures on a modern
            #### server. 
            #### In addition, this type of operation would be excellent for 
            #### GPU implementation. 
            ele = self.unit_cell_struct.elements
            self.vdw_array = [self.vdw[atomic_numbers[x]] for x in ele]
            self.vdw_array = np.array(self.vdw_array)[None,:]
            total_occupied = 0
            total_points = 0
            total = len(self.grid_cart[::self.batch_size])
            
            for idx,value in enumerate(self.grid_cart[::self.batch_size]):
                print("{}: {}".format(total, idx))
                
                start_idx = idx*self.batch_size
                end_idx = (idx+1)*self.batch_size
                if end_idx > len(self.grid_cart):
                    end_idx = len(self.grid_cart)
                
                idx_values = np.arange(start_idx,end_idx,1)
                temp_cart = self.grid_cart[idx_values]
                temp_dist = cdist(temp_cart, 
                                  self.unit_cell_struct.get_geo_array())
                
                ### Divide dist by vdW radius
                temp_dist = temp_dist / self.vdw_array
                
                temp_min_dist = np.min(temp_dist, axis=-1)
                temp_occupied_idx = np.where(temp_min_dist < 1)[0]
                
                total_occupied += temp_occupied_idx.shape[0]
                total_points += temp_cart.shape[0]
        
            
            packing_factor = total_occupied / total_points
                
        
        self.struct.properties["PackingFactor"] = packing_factor
        
        return packing_factor
    
    
    def keep_inside(self, struct):
        """
        Keeps only the atoms that are inside the unit cell plus a factor equal
        to the largest vdW radius. 
        
        """
        geo = struct.get_geo_array()
        ele = struct.elements
        
        vdw_list = [self.vdw[atomic_numbers[x]] for x in ele]
        correction = np.max(vdw_list)
        max_frac_correction = correction / np.min(self.norm)
        max_frac_correction = max_frac_correction*2
        
        frac = np.dot(geo,self.linv)
        
        keep_idx = np.where(
                np.logical_and(
                    (frac>=(0-max_frac_correction)).all(axis=-1),
                    (frac<=(1+max_frac_correction)).all(axis=-1)
                        ))[0]
        
        geo = geo[keep_idx]
        ele = ele[keep_idx]
        
        unit_cell_struct = Structure.from_geo(geo, ele)
        unit_cell_struct.set_lattice_vectors(self.lat)
        
        return unit_cell_struct
        
    
    def move_atoms_in(self):
        geo = self.struct.get_geo_array()
        ele = self.struct.elements
        
        frac = np.dot(self.linv,geo.T).T
        
        ## Fix values greater than 1
        greater_idx = np.where(frac > 1)
        subtract = frac[greater_idx].astype(int)
        frac[greater_idx] -= subtract
        
        ## Fix values less than zero
        less_idx = np.where(frac < 0)
        add = -frac[less_idx].astype(int)+1
        frac[less_idx] += add
        
        cart = np.dot(frac, self.lat)
        
        struct = Structure.from_geo(cart, ele)
        struct.set_lattice_vectors(self.lat)
        
        return struct
        
    
    def generate_grid(self, num):
        """
        Generates appropriate grid for the system. 
        
        Arguments
        ---------
        num: Iterable
            Iterable of length 3 that describes how many values should be 
            generated in each direction for the grid. 
        
        """
        ## This should be adjusted so that it's exactly at the edge of the 
        ## unit cell.
        range_x = np.arange(0,num[0],1) / num[0]
        range_y = np.arange(0,num[1],1) / num[1]
        range_z = np.arange(0,num[2],1) / num[2]
        
        grid_x,grid_y,grid_z = np.meshgrid(range_x,range_y,range_z,
                                           indexing="ij")
        
        grid = np.c_[grid_x.ravel(),
                     grid_y.ravel(),
                     grid_z.ravel()]
        
        return grid
        
        
        
if __name__ == "__main__":
    pass
    
##        
        
        
        
        
        
        