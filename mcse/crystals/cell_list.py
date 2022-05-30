# -*- coding: utf-8 -*-

import copy

import numpy as np
from scipy.spatial.distance import cdist

from mcse.core import Structure
from mcse.core.driver import BaseDriver_
from mcse.crystals.supercell import Supercell
from mcse.crystals.lebedev import lebedev_5,lebedev_64
lebedev_64 = np.array(lebedev_64["coords"])
lebedev_5 = np.array(lebedev_5["coords"])


def cell_list(struct, cell_size=0.01):
    """
    Computes the cell-list of the given structure to improve the speed of 
    neighborlist calculations
    
    Arguments
    ---------
    structure: Structure
        Crystal geometry to compute the cell-list for every atom within the 
        geometry
    cell_size: float
        Voxel size of the cell-list given in fractional coordinates
    
    """
    lv = np.vstack(struct.lattice)
    
    if len(lv) == 0:
        raise Exception("Cannot compute cell-list of non-periodic "+
                "geometry {}".format(struct))
    
    lv_inv = np.linalg.inv(lv.T)
    geo = struct.geometry
    frac = np.dot(lv_inv, geo.T).T
    
    ### Should always be able to compute % 1 of frac here. A structure without
    ### a properly defined lattice should not be given to this program
    frac %= 1

    cell_loc = (frac / cell_size).astype(int)
    struct.properties["cell-list"] = cell_loc
    

def get_sc_mult(struct, radius):
    """
    The multiplicity for the supercell must enclose all lattice sites that
    are less than the given radius away
    """
    lv = np.vstack(struct.lattice)
    lv_inv = np.linalg.inv(lv.T)
    min_norm = np.min(np.linalg.norm(lv, axis=-1))
    
    ### Calculate fractional coordinates of sphere of given radius at each 
    ###   lattice cite of the original unit cell. 
    vert = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    vert_cart = np.dot(vert,lv)
    frac_list = []
    for temp_pos in vert_cart:
        ### Make sure to operate on copy of coords
        coords = lebedev_5.copy()
        coords = coords + temp_pos
        coords *= radius
        frac = np.dot(lv_inv, coords.T).T
        frac_list.append(frac)
    ### Get max and min and round appropriately
    frac = np.vstack(frac_list)
    max_frac = np.ceil(np.max(frac, axis=0))
    min_frac = np.floor(np.min(frac, axis=0))
    final_range = max_frac - min_frac
    return final_range
        
      
    # ### Just choose to be a very large value such that even for very oblique cells
    # ###   it will be robust
    # max_range = np.ceil(((radius / min_norm)+1)*3).astype(int)
    # lv_range = np.hstack([np.arange(0,max_range+1), np.arange(-max_range,0)])

    # ### Get fractional points
    # a_grid,b_grid,c_grid = np.meshgrid(lv_range, lv_range, lv_range, 
    #                                     indexing="ij")
    # frac_grid = np.c_[a_grid.ravel(), 
    #                     b_grid.ravel(), 
    #                     c_grid.ravel()]

    # ### Get cartesian translations
    # cart_grid = np.dot(frac_grid, lv)

    # ### Cartesian distances must be greater than radius away from any lattice site
    # ### of the original unit cell 
    # keep_mask = np.zeros(len(cart_grid,))
    # vert = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    # vert_cart = np.dot(vert,lv)
    # ### Divide by 2 is correct, but I'll use divide by 3 for extra buffer 
    # ###  against bad behavior at small inefficiency expense
    # dist = np.linalg.norm(cart_grid - vert_cart[:,None], axis=-1)/3
    # keep_mask = np.max(dist < (radius+0.1), axis=0)

    # keep_frac = frac_grid[keep_mask]

    # max_frac = np.max(keep_frac,axis=0)
    # min_frac = np.min(keep_frac,axis=0)
    # final_range = max_frac - min_frac
    
    # return final_range+1
    
    
def supercell_cell_list(struct, cell_size=0.01, radius=4):
    """
    Computes the supercell and the cell-list of the supercell thus making the 
    API easier for the implementation of finding fully periodic neighborlists 
    
    """
    supercell_mult = get_sc_mult(struct, radius)
    
    ### NEED TO USE A CENTERED SUPERCELL FOR THIS TO WORK
    supercell_driver = CenteredSupercell(
                                mult=supercell_mult,
                                standardize=False,
                                track_molecules=False,
                                lattice=True)
    
    supercell = supercell_driver.calc_struct(struct)
    
    ### Now, only need to keep atoms in the unit cell that are within the radius
    ### of the centered supercell. Cell-list computed at this step. 
    supercell = supercell_driver.remove_radius(supercell, radius, cell_size)
    
    return supercell
    
    
def get_cell_neigh(struct, atom_idx=[], cell_size=0.01, radius=4):
    """
    Obtains the cell neighborhood for given atoms using a specified cutoff 
    radius
    
    """
    if len(atom_idx) == 0:
        num_atom = len(struct.geometry)
        atom_idx = np.arange(0,num_atom)
        
    cell_radius = get_cell_radius(struct, cell_size=cell_size, radius=radius)
    atom_cell_idx = []
    for temp_atom_idx in atom_idx:
        atom_cell_idx.append(_get_cell_neigh(struct, cell_radius, temp_atom_idx))
    
    return atom_cell_idx
        

def get_cell_radius(struct, cell_size=0.01, radius=4):
    """
    Converts the given radius, in angstrom, to the integer size of the fractional
    cells given the cell size. Returned is the number of cells in each direction
    representing a rectangular volume in fractional coordinates that fully 
    encompasses a sphere of the given cartesian radius.  
        
    """
    lv = np.vstack(struct.lattice)
    lv_inv = np.linalg.inv(struct.lattice.T)
    ### Get spherical coordinates
    coords = lebedev_64
    ### Scale by radius
    coords = coords*radius
    ### Compute positions in fractional space
    frac_coords = np.dot(lv_inv, coords.T).T
    ### Get rectangular volume in fractional space
    rect = np.max(frac_coords, axis=0)
    ### Calculate cells in each direction for this rectangular volume
    cell_radius = np.ceil(rect / cell_size).astype(int)
    return cell_radius+1
    
    
def _get_cell_neigh(struct, 
                   cell_radius,
                   single_atom_idx):
    """
    Obtains the atom idx that in the cell neighborhood of the given atom
    in the given structure. Must pass in frac radius because otherwise this 
    function is too slow.
    
    By moving cell_radius outside of this function, the execution time went from
    70 us to 30 us which is a huge savings. 
    
    """        
    if "cell-list" not in struct.properties:
        cell_list(struct)
    
    cl = struct.properties["cell-list"]
    atom_cell = cl[single_atom_idx]
    ### Obtain copy of atom_cell
    atom_cell = atom_cell + 0
    min_cell = atom_cell - cell_radius
    max_cell = atom_cell + cell_radius
    
    ### Fast way to avoid adding the relevant atom itself
    cl[single_atom_idx] = [100000000,100000000,100000000]
    
    ### This needs to be a lot faster... Currently slowing everything down...
    ### Alternatively, the supercell should be made smaller to reduce the number
    ### of operations...
    atom_cell_idx = np.where(
        np.logical_and(cl[:,0] >= min_cell[0],
        np.logical_and(cl[:,1] >= min_cell[1],
        np.logical_and(cl[:,2] >= min_cell[2],
        np.logical_and(cl[:,0] <= max_cell[0],
        np.logical_and(cl[:,1] <= max_cell[1],
                       cl[:,2] <= max_cell[2]))))))[0]
    ### In addition, any dense implementation is just going to be too memory
    ### intensive if cell size is small... But usually cell size is not small...
    
    ### Only other alternative is to remember the cell neighborhoods that 
    ### have been calculated previously and use these for reference...
    ### But this is probably not fruitful...
    
    ### Reset cl to correct entry
    cl[single_atom_idx] = atom_cell
    
    # return np.hstack([single_atom_idx, atom_cell_idx])
    
    ### Append single_atom_idx to the front of the atom_cell_idx
    ret_idx = np.zeros((len(atom_cell_idx)+1)).astype(int )
    ret_idx[0] = single_atom_idx
    ret_idx[1:] = atom_cell_idx
    return ret_idx


def move_inside_cell(struct):
    lv = np.vstack(struct.lattice)
    
    if len(lv) == 0:
        raise Exception("Cannot compute cell-list of non-periodic "+
                "geometry {}".format(struct))
    
    lv_inv = np.linalg.inv(lv.T)
    geo = struct.get_geo_array()
    frac = np.dot(lv_inv, geo.T).T
    frac %= 1
    
    cart = np.dot(frac, lv)
    
    struct.geometry = cart
    return struct


class CenteredSupercell(Supercell):
    """
    Builds supercell of structure. In addition, ensures that the first unit 
    cell in the list of atomic positions in a unit cell that is centered 
    in the middle of the given supercell. 
    
    PLEASE NOTE: This this needs to be fully tested to ensure that no issue
    occurs from the definition that is built by the centered unit cell in 
    all supercell multiplicities
    
    Put this here to test a different approach. This approach first translates
    the atoms in the unit cell and then constructs supercell. This is instead
    of finding the reordering to product the same geometry. 
    
    IMPORTANT: The remove_radius method does not preserve that the supercell 
    produced will be equivalent to the input structure. This method is meant for 
    maximum computational efficiency to calculate periodic neighborlists, not 
    for preserving chemical composition. 
    
    
    Arguments
    ---------
    mult: iterable
        Multiplicity to construct the supercell in the a,b,c 
        directions. 
    
    """
    def __init__(self, 
                 mult=(3,3,3),
                 standardize=False,
                 track_molecules=False,
                 lattice=True,
                 bonds_kw={}):
        self.mult = np.array(mult)
        self.standardize = standardize
        self.track_molecules = track_molecules
        self.bonds_kw = bonds_kw
        self.lattice = lattice
        
        for entry in mult:
            if entry < 0:
                raise Exception("Cannot use negative supercell multiplicity.")
        
        self.odd_bool = (np.array(self.mult) % 2).astype(bool)
        self.half_mult_int = (np.array(self.mult) / 2).astype(int)
        
        self.all_odd_mult = np.array(mult)
        self.all_odd_mult[np.logical_not(self.odd_bool)] += 1
            
        
    def calc_struct(self, struct):
        self.struct = struct
        init_geo = struct.geometry
        if self.standardize:
            raise Exception("Standardize not allowed safely")
        if self.track_molecules:
            self.molecule_idx = struct.get_molecule_idx(**self.bonds_kw)
        
        self._centered_translation(self.struct)
                    
        self.supercell = self.construct_supercell(self.struct)
        # self.supercell = self.reorder(self.supercell)
        
        ### Move back inside because _centered_translation edited positions
        ### Only costs a millisecond whereas copying structure is more costly
        struct.from_geo_array(init_geo, struct.elements)
        
        return self.supercell
    

    def _centered_translation(self, struct):
        """
        Finds the translation that will produce a centered unit cell in the 
        final supercell.
        
        """
        ### I was copying the structure in memory, but this doesn't matter 
        ### because the structure can also be moved inside after supercell 
        ### is finished
        prepared_struct = struct
                
        lv = np.vstack(prepared_struct.lattice)
        lv_inv = np.linalg.inv(lv.T)
        geo = prepared_struct.geometry.copy()
        
        frac = np.dot(lv_inv, geo.T).T
        frac %= 1
        
        frac_round = np.round(frac).astype(int)
        
        ### If the unit cell construction is even, then for atoms that have a 
        ### fractional coordinate less than 0.5, these are treated the same 
        no_adjust_idx = np.where(frac_round == 0) 
        adjust_idx = np.where(frac_round == 1)
        
        trans_frac = np.zeros(frac.shape).astype(int)
        ### Apply first across entire target frac even though this is not 
        ### correct. Otherwise, the indexing doesn't work because it only 
        ### goes into an "image" of the real matrix. So, instead, just 
        ### reset odd multiplicities later
        trans_frac[no_adjust_idx] += \
            np.take(self.half_mult_int, no_adjust_idx[1])
            
        ### However, if fractional coordinate is greater than 0.5, then one
        ### less lattice translation should be used in that direction to 
        ### obtained the position of the atom for the centered unit cell
        trans_frac[adjust_idx] += \
            np.take(self.half_mult_int, adjust_idx[1])-1
            
        ### Treat odd last after resetting target_frac for odd entries 
        trans_frac[:,self.odd_bool] = 0
        trans_frac[:,self.odd_bool] += self.half_mult_int[self.odd_bool]
        
        translation = np.dot(trans_frac, lv)
        geo += translation
        
        prepared_struct.geometry = geo
        prepared_struct.properties["centered_translation"] = translation
        prepared_struct.properties["centered_translation_frac"] = trans_frac
        
        ### Calculate the new origin for this unit cell
        new_frac_origin = self.half_mult_int.astype(float)
        new_frac_origin[np.logical_not(self.odd_bool)] = \
                new_frac_origin[np.logical_not(self.odd_bool)]/2
        new_origin = np.dot(new_frac_origin, lv)
        
        prepared_struct.properties["centered_origin_frac"] = new_frac_origin
        prepared_struct.properties["centered_origin"] = new_origin
                
        return prepared_struct
    
    
    def construct_supercell(self, struct=None):
        """
        Because of the way the initial unit cell has been treated in 
        _centered_translation, it is possible to construct the unit cell using 
        negative translations for all odd multiplicities and then remove all 
        atoms outside the desired multiplicity to achieve the final structure. 
        
        """
        if struct == None:
            struct = self.struct
        
        geo = struct.geometry.copy()
        ele = struct.elements
                
        ## Convert coords to fractional representation
        lv = np.vstack(struct.get_lattice_vectors())
        lv_inv = np.linalg.inv(lv.T)
        
        ranges = (self.all_odd_mult / 2).astype(int)
        a_range = np.hstack([np.arange(0,ranges[0]+1), np.arange(-ranges[0],0)])
        b_range = np.hstack([np.arange(0,ranges[1]+1), np.arange(-ranges[1],0)])
        c_range = np.hstack([np.arange(0,ranges[2]+1), np.arange(-ranges[2],0)])
        
        ### Get fractional points
        a_grid,b_grid,c_grid = np.meshgrid(a_range, b_range, c_range, 
                                           indexing="ij")
        frac_grid = np.c_[a_grid.ravel(), 
                          b_grid.ravel(), 
                          c_grid.ravel()]
        
        ### Get cartesian translations
        cart_grid = np.dot(frac_grid, lv)
        desired_lv = np.dot(np.diag(self.mult), lv)
        desired_lv_inv = np.linalg.inv(desired_lv.T)
        
        supercell_geo = np.vstack(cart_grid[:,None]+geo[None,:])
        supercell_ele = np.tile(ele, reps=len(cart_grid))
        supercell_trans = np.repeat(cart_grid, len(geo), axis=0)
        original_idx = np.tile(np.arange(0,len(geo)), reps=len(cart_grid))
        
        supercell_frac = np.dot(desired_lv_inv, supercell_geo.T).T
        keep_idx = np.where(
            np.logical_and((supercell_frac > 0).all(axis=-1), 
                            (supercell_frac < 1).all(axis=-1)))[0]
        
        keep_geo = supercell_geo[keep_idx]
        keep_ele = supercell_ele[keep_idx]
        keep_original_idx = original_idx[keep_idx]
        keep_trans = supercell_trans[keep_idx]
        final_trans = keep_trans + \
                struct.properties["centered_translation"][keep_original_idx]
        
        supercell_id = "{}_CenteredSupercell_{}_{}_{}".format(
            struct.struct_id, self.mult[0], self.mult[1], self.mult[2])
        supercell = Structure(struct_id=supercell_id,
                              geometry=keep_geo, 
                              elements=[], 
                              lattice=desired_lv,
                              bonds=[[]],
                              trusted=True)
        ### This is much faster 
        supercell.elements = keep_ele
        
        supercell.properties["original_idx"] = keep_original_idx
        supercell.properties["original_supercell_trans"] = final_trans
        supercell.properties["original_lv"] = lv
        supercell.properties["original_lv_inv"] = lv_inv
        supercell.properties["original_origin"] = \
                        struct.properties["centered_origin"]
        supercell.properties["original_origin_frac"] = \
                        struct.properties["centered_origin_frac"]
        supercell.properties["lv_inv"] = desired_lv_inv
        supercell.properties["offsets_cart"] = keep_trans
        
        if self.track_molecules:
            raise Exception()
        
            ### Molecule idx is going to change based on the translation applied
            ### to the atoms in the unit cell. These translations need to be 
            ### matched together with the given structure to construct the 
            ### correct tracking without having to explicitly calculate the 
            ### molecule indices
            
            ### How to connect the molecules across the edges of the supercell?
            ### I guess can just do this by identifying...
            
            pass
            
            # temp_molecule_idx_offset = np.arange(0,len(cart_grid))*len(geo)
            # supercell_molecule_idx = []
            # for idx,offset in enumerate(temp_molecule_idx_offset):
            #     ### For each molecule, keep track of mol_idx
            #     for temp_mol_idx in self.molecule_idx:
            #         temp_array = np.array(temp_mol_idx) + offset
            #         supercell_molecule_idx.append(
            #             temp_array.tolist()
            #             )
                
            # supercell.properties["molecule_idx"] = supercell_molecule_idx
            # ### First get ordering of molecules wrt geometry
            # molecule_image_idx = np.zeros((geo.shape[0],))
            # for temp_mol_idx,entry in enumerate(self.molecule_idx):
            #     molecule_image_idx[entry] = temp_mol_idx
                
            # ### Due to the way the supercell geometry is constructed, the 
            # ### molecule_idx for the geometry just needs to be repeated
            # ### the same number of times as the number of translations
            # ### which is easily accomplished with np.tile
            # supercell_geo_molecule_idx = np.tile(molecule_image_idx,
            #                                     (len(cart_grid),))
            # supercell.properties["molecule_image_idx"] = \
            #                         supercell_geo_molecule_idx.astype(int)
        
        # self.keep_idx = keep_idx
        # self.supercell_frac = supercell_frac
        # self.desired_lv = desired_lv
        # self.frac_grid = frac_grid
        # self.keep_original_idx = keep_original_idx
        # self.geo = geo
        # self.cart_grid = cart_grid
        # self.supercell_ele = supercell_ele
        # self.keep_trans = keep_trans
        # self.final_trans = final_trans
        
        return supercell
    
    
    def remove_radius(self, supercell, radius, cell_size=0.01, tol=0.1):
        """
        Removes all atoms that are outside the given radius from the original 
        centered unit cell. This is an approximate operation. If the method 
        was exact, then no time could be saved anyways. 
        
        For 49,024 atoms in the supercell this takes ~15 ms
        For 2,584 atoms in the supercell this takes 1 ms
        
        """
        natoms = len(self.struct)
        cell_list(supercell, cell_size)
        cell_radius = get_cell_radius(supercell, cell_size, radius)
        
        ### Perform approximate removal based on the cell_radius 
        init_cell_list = supercell.properties["cell-list"][0:natoms]
        max_cell = np.max(init_cell_list, axis=0)+cell_radius+1
        min_cell = np.min(init_cell_list, axis=0)-cell_radius-1
        
        ### Find all atoms within this rectangular volume from fraction space
        cl = supercell.properties["cell-list"]
        keep_idx = np.where(
                np.logical_and(cl[:,0] >= min_cell[0],
                np.logical_and(cl[:,1] >= min_cell[1],
                np.logical_and(cl[:,2] >= min_cell[2],
                np.logical_and(cl[:,0] <= max_cell[0],
                np.logical_and(cl[:,1] <= max_cell[1],
                               cl[:,2] <= max_cell[2]))))))[0]

        # geo = supercell.geometry.copy()
        # original_lv = supercell.properties["original_lv"]
        # original_lv_norm = np.linalg.norm(original_lv, axis=-1)
        # original_lv_inv = supercell.properties["original_lv_inv"]
        # original_origin = supercell.properties["original_origin"]
        # geo -= original_origin
        # original_frac = np.dot(original_lv_inv, geo.T).T
        # original_frac_radius = radius / original_lv_norm
        # frac_tol = tol / original_lv_norm
        
        # min_frac = 0-original_frac_radius-frac_tol
        # max_frac = 1+original_frac_radius+frac_tol
        
        # dist = np.linalg.norm(geo-geo[0:len(self.struct)][:,None], axis=-1)
        # keep_mask = np.max(dist < (radius+1), axis=0)
        # keep_idx = np.where(keep_mask == True)[0]
        
        # self.original_frac = original_frac
        # self.original_frac_radius = original_frac_radius
        # self.keep_idx = keep_idx
        
        ### Need to update the properties
        new_id = "{}_Radius_Removed".format(supercell.struct_id)
        radius_removed = supercell.get_sub(keep_idx, struct_id=new_id, bonds=False)
        radius_removed.properties["original_idx"] = \
                    supercell.properties["original_idx"][keep_idx]
        radius_removed.properties["original_supercell_trans"] = \
                    supercell.properties["original_supercell_trans"][keep_idx]
        radius_removed.properties["original_lv"] = \
                    supercell.properties["original_lv"]
        radius_removed.properties["original_lv_inv"] = \
                    supercell.properties["original_lv_inv"]
        radius_removed.properties["original_origin"] = \
                    supercell.properties["original_origin"]
        radius_removed.properties["original_origin_frac"] = \
                    supercell.properties["original_origin_frac"]
        radius_removed.properties["lv_inv"] = \
                    supercell.properties["lv_inv"]
        radius_removed.properties["radius_removed"] = radius
        
        radius_removed.properties["offsets_cart"] = \
            supercell.properties["offsets_cart"][keep_idx]
        
        if self.track_molecules:
            init_mol_idx = supercell.properties["molecule_idx"]
            init_mol_image_idx = supercell.properties["molecule_image_idx"]
            
            ### init_mol_image_idx is just for each atom, therefore can be 
            ### easily masked by keep_idx to get the correct result
            final_image_idx = init_mol_image_idx[keep_idx]
            
            ### getting correct init_mol_idx requires re-indexing every atom 
            ### and removing indices which are no longer present
            lookup = {}
            for iter_idx,atom_idx in enumerate(keep_idx):
                lookup[atom_idx] = iter_idx
            final_mol_idx =  []
            for temp_atom_list in init_mol_idx:
                temp_idx_list = []
                for temp_atom_idx in temp_atom_list:
                    if temp_atom_idx in lookup:
                        temp_idx_list.append(lookup[temp_atom_idx])
                if len(temp_idx_list) > 0:
                    final_mol_idx.append(temp_idx_list)
                            
            radius_removed.properties["molecule_idx"] = final_mol_idx
            radius_removed.properties["molecule_image_idx"] = final_image_idx
        
        return radius_removed

