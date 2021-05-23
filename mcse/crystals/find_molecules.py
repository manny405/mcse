
import os,copy
import numpy as np

from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ase.neighborlist import NeighborList,natural_cutoffs

from mcse import Structure
from mcse.core.struct_dict import StructDict
from mcse.io import read,write
from mcse.core.utils import com
from mcse.core.driver import BaseDriver_                             
                               
from mcse.core.utils import com
from mcse.molecules.label import label,match_labels,get_bond_fragments
from mcse.molecules.unique import unique_by_rmsd,unique_by_bonding
from mcse.crystals.lebedev import lebedev_5


class FindMolecules(BaseDriver_):
    """
    Find molecular units in periodic and non-periodic Structures. In addition, 
    a whole molecule representation may be constructed easily from the initial
    structure by using the ``translate`` property to translate the original
    geometry by exact lattice translations. 
    
    Arguments
    ---------
    residues: int
        Number of unqiue molecules the user would like to find. If the value is 
        zero, then default settings are used. However, if ean integer value is 
        supplied, then the mult parameter will be varied to try to achieve an 
        identical number of unique residues in the system. 
    conformation: bool
        If True, will account for conformational degrees of freedom when
        identifying unique molecules. If False, then molecules will be 
        duplicates if they have the exact same covalent bonds present. 
    mult_range: iterable
        Values for mult parameter to search over if residues is fixed nubmer.
    
    """
    def __init__(self, 
                 residues=0,
                 mult=1.05, 
                 conformation=True, 
                 mult_range=np.arange(1.05, 1.25, 0.005),
                 rmsd_tol=0.1,
                 supercell_radius=10,
                 verbose=False):
        
        self.residues=residues
        self.mult = mult
        self.conformation = conformation
        self.supercell_radius = supercell_radius
        self.rmsd_tol = rmsd_tol
        
        if type(residues) != int or residues < 0:
            raise Exception("Residues argument can only be a positive integer.")
        
        self.molecules = {}
        self.unique = {}
        self.supercell_driver = SupercellSphere(radius=supercell_radius, 
                                                mult=0, 
                                                correction=True,
                                                track_molecules=False)
        
        ### Holds the translation required by every atom in order to provide a 
        ### whole molecule representation. Each translation will be by exactly
        ### a combination of lattice vectors
        self.translate = np.array([])
    
    
    def calc_struct(self, struct):
        ## Reset internal storage
        self.struct = struct
        self.molecules = {}
        self.unique = {}
        self.translate = np.array([[0.,0.,0.] for x in self.struct])

        if self.struct.lattice.shape != (3,3):
            raise Exception("FindMolecules input must be a periodic crystal")
        
        if self.residues == 0:
            self._calc_struct(struct, self.mult)
        else:
            ## Otherwise search over mult range
            step = 1
            
            for mult in self.mult_range[::step]:
                if self.verbose:
                    print("{}: Trying mult={:.3f}".format(self.struct.struct_id,
                                                      mult))
                self._calc_struct(struct, mult)
                ## Now check if obtained the correct number of unique molecules
                if len(self.unique) == self.residues:
                    if self.verbose:
                        print("{}: Success finding {} unique residues"
                              .format(self.struct.struct_id, self.residues))
                    break
                else:
                    ## If not correct, reset calculation. 
                    ## If using none of the mult values work, then this will 
                    ## leave the len of unique_molecule_list as zero indicating
                    ## failure. 
                    self.molecules = {}
                    self.unique = {} 
                    self.translate = np.array([[0.,0.,0.] for x in self.struct])
    
    
    def _calc_struct(self, struct, mult=1.05):
        """
        Wrapper for performing calculation using specific mult parameter.
        
        """           
        ### Correct initial molecules that may not be connected
        init_mol_dict = struct.molecules
        mol_idx = struct.get_molecule_idx()
        iter_idx = 0
        for temp_mol_id,temp_mol in init_mol_dict.items():
            temp_whole_mol = self.get_whole_molecule(temp_mol)
            temp_whole_mol.struct_id = temp_mol_id
            temp_mol_idx = mol_idx[iter_idx]
            self._get_whole_trans(temp_mol, 
                                  temp_whole_mol, 
                                  temp_mol_idx)
            self.molecules[temp_mol_id] = temp_whole_mol
            iter_idx += 1
            
        ### Find unique molecules from molecule dictionary based on init args
        if self.conformation:
            self.unique = unique_by_rmsd(self.molecules, tol=self.rmsd_tol)
        else:
            self.unique = unique_by_bonding(self.molecules)

    
    def get_whole_molecule(self, mol, mult=-1, struct=None):
        """
        Algorithm Responsible for taking a fragmented molecule from the unit 
        cell and finding a whole molecule representation. This is done by 
        using the unit cell parameters of the initial structure, constructing
        a large supercell, and finding a whole molecule that is completed by
        successive additions of unit cells.
        
        """
        if mult <= 0:
            mult = self.mult
        if struct == None:
            struct = self.struct
            
        # Check if molecule is fully connected
        mol_index_list = self._find_molecules(mol,mult=mult)
        if len(mol_index_list) == 1:
            return mol
        
        mol_lv = mol.copy()
        mol_lv.lattice = struct.lattice
        mol_supercell = self.supercell_driver.calc_struct(mol_lv)
        mol_index_list = self._find_molecules(mol_supercell, mult)
        target_length = len(mol)
        success = False
        for temp_idx in mol_index_list:
            if len(temp_idx) == target_length:
                whole_mol = mol_supercell.get_sub(temp_idx, lattice=False)
                success = True
                break
        
        if success == False:
            raise Exception('No whole represenation was found for the molecule '+
                'without periodic boundary conditions. Please check the ' +
                'structure for any irregularities.')
        
        ### Reduce fractional coordinate of the COM of molecule
        frac_com = struct.cart2frac(com(whole_mol)).round(6)
        reduced_frac_com = frac_com % 1
        frac_trans = reduced_frac_com - frac_com
        trans = struct.frac2cart(frac_trans)
        whole_mol.translate(trans)
        
        return whole_mol
            
        
    def _find_molecules(self, struct, mult=-1):
        if mult <= 0:
            mult = self.mult
        atoms = struct.get_ase_atoms()
        cutOff = natural_cutoffs(atoms, mult=mult)
        ## Skin=0.3 is not a great parameter, but it seems to work pretty well
        ## for mulicule identification. In addition, it's not the same affect as 
        ## change the mult value because it's a constant addition to all 
        ## covalent bonds.
        neighborList = NeighborList(cutOff, skin=0.0)
        neighborList.update(atoms)
        matrix = neighborList.get_connectivity_matrix()
        n_components, component_list = sparse.csgraph.connected_components(matrix)
        molecule_list = [np.where(component_list == x)[0] 
                        for x in range(n_components)]
        return molecule_list
    
    
    def _get_whole_trans(self, 
                         temp_mol, 
                         temp_whole_mol, 
                         temp_mol_idx):
        """
        From the initial molecule geometry, finds the exact combination of 
        lattice translations for each atom of the original geometry that 
        produces the whole molecule representation
        
        """
        ogeo = self.struct.cart2frac(temp_mol.geometry)
        wgeo = self.struct.cart2frac(temp_whole_mol.geometry)
        
        ### Find the mapping from ogeo to wgeo that provides exact lattice
        ### translations for all atoms
        mask = np.ones((wgeo.shape[0],))
        map_idx = []
        for pos in ogeo:
            temp_diff = pos[None,:] - wgeo[mask.astype(bool)]
            temp_diff = temp_diff.round(6) + 0
            temp_frac_err = np.linalg.norm(temp_diff % 1, axis=-1)
            if np.min(temp_frac_err) > 0.1:
                raise Exception("Obtaining translation for whole molecule "+
                    "representation resulted in non-integer lattice "+
                    "translation. This shouldn't be possible.")
            chosen_idx = np.argmin(temp_frac_err)
            ### De-reference chosen_idx wrt the mask
            avail_idx = np.where(mask == 1)[0]
            chosen_idx = avail_idx[chosen_idx]
            map_idx.append(chosen_idx)
            mask[chosen_idx] = 0

        ### Finally, find translation from ogeo to wgeo using map_idx
        final_trans = wgeo[map_idx] - ogeo
        self.translate[temp_mol_idx] = self.struct.frac2cart(final_trans)
    
    
    def _reorder_atoms(self):
        """
        Reorders atoms using mcse label function and match_labels
        
        """
        unique_molecule = self.unique[0]
        label(unique_molecule)
        
        for molecule in self.molecules:
            match_labels(unique_molecule, molecule, warn=True)   
            
            
    
    def write(self, output_dir, file_format="json", overwrite=False):
        if len(output_dir) == 0:
            output_dir = "{}_Molecules".format(self.struct.struct_id)
        
        write(output_dir, self.molecules, file_format=file_format, overwrite=overwrite)
        
        unique_output = {}
        for struct_id,struct in self.unique.items():
            temp_unique = struct.copy()
            temp_unique.struct_id += "_unique"
            unique_output[temp_unique.struct_id] = temp_unique
        
        write(output_dir, unique_output, file_format=file_format, overwrite=overwrite)
            
        
class SupercellSphere(BaseDriver_):
    """
    Builds a spherical supercell using the provided distance as the radius of 
    the sphere. Accounts for atoms outside of the unit cell by computing a 
    correction to add to the radius radius. 
    
    Arguments
    ---------
    radius: float
        By default, will use this radius value. Will construct of sphere of 
        unit cells greater than or equal to this radius value. 
    mult: int
        If multiplicity is greater than zero, then the radius will be 
        calculated as the largest lattice vector multiplied by mult. 
    correction: bool
        If True, a correction will be automatically calculated for atoms
        outside the unit cell to expand the supercell to account for this.
        If False, no correction will be used and the exact radius will be 
        used. 
    track_molecules: bool
        If True, the molecule in the unit cell from which each atom comes 
        from will be stored as a property in the supercell. Also, this will
        provide a algorithm for building the molecule_idx list for the 
        entire supercell based on the unit cell. 
    bond_kw: bool
        Settings for struct.get_bonds used for finding molecules.
    
    """
    def __init__(self, 
                 radius=10, 
                 mult=0, 
                 correction=True,
                 track_molecules=True,
                 bonds_kw={}):
        self.radius = radius
        self.mult = mult
        self.correction=correction
        self.track_molecules = track_molecules
        self.bonds_kw = bonds_kw
        self.molecule_idx = []

    
    def calc_struct(self,struct):
        self.struct = struct
        if self.track_molecules:
            self.molecule_idx = self.struct.get_molecule_idx(**self.bonds_kw)        
        
        ## Calculate radius based on mult
        if self.mult > 0:
            self.radius_by_mult()
        
        correction = self.get_correction(struct)
        if not self.correction:
            correction = 0
            
        self.supercell = self.construct_supercell(struct, correction)
        self.supercell.struct_id = "{}_Supercell_{}".format(
                self.struct.struct_id,self.radius)
        self.supercell.properties["Radius"] = self.radius
        self.supercell.properties["Mult"] = self.mult
        
        return self.supercell
    
    
    def write(self, output_dir, file_format="json", overwrite=False):
        temp_dict = {self.supercell.struct_id: self.supercell}
        write(output_dir, 
              temp_dict, 
              file_format=file_format, 
              overwrite=overwrite)
    
    
    def radius_by_mult(self):
        """
        Calculates radius as the largest lattice vector norm multiplied by the 
        multiplicity.
        
        """
        lat = np.vstack(self.struct.get_lattice_vectors())
        norms = np.linalg.norm(lat, axis=-1)
        max_norm = np.max(norms)
        self.radius = max_norm*self.mult
    
    
    def get_correction(self, struct=None):
        """
        Some atoms of moleceules may be outside the unit cell. This atom that
        is furthest outside the unitcell needs to be used as a correction to 
        the radius distance to guarantee proper construction of unit cell 
        sphere. 
        
        """
        if struct == None:
            struct = self.struct
        
        geo = struct.get_geo_array()
        lv = np.vstack(struct.get_lattice_vectors())
        lv_inv = np.linalg.inv(lv.T)
        
        ## Get all fractional coordinates
        frac = np.dot(lv_inv, geo.T).T
        
        ## Find largest magnitude
        frac = np.abs(frac)
        max_frac = np.max(frac)
        
        if max_frac < 1:
            return 0
        
        ## Compute cartesian correction to radius
        idx = np.argmax(frac)
        vec_idx = idx % 3
        correction = np.linalg.norm(lv[vec_idx,:])*(max_frac - 1)
        
        return correction
        
        
    def construct_supercell(self, struct=None, correction=0):
        """
        Uses points on Lebedev grid to construct points on the surface of the 
        desired radius radius. Then, the nearest lattice point to each of these
        points which is greater than the radius distance away is used. Once
        we know the points furtherst away for the supercell, all intermediary 
        lattice points may be added easily. Translation vectors for the 
        molecules in the unit cell may be applied to fill out the supercell 
        sphere. 
        
        """
        if struct == None:
            struct = self.struct
        
        geo = struct.get_geo_array()
        ele = struct.elements
            
        ## Get 50 coordinates on the Levedev grid by using n=5
        ## This is a reasonable number because the lattice sites will be very 
        ## corse grained compared to the grid
        coords = np.vstack(lebedev_5["coords"])
        
        radius = self.radius + correction
        ## Modify coords for unit sphere for radius
        coords *= radius
        
        ## Convert coords to fractional representation
        lv = np.vstack(struct.get_lattice_vectors())
        lv_inv = np.linalg.inv(lv.T)
        frac_coords = np.dot(lv_inv, coords.T).T
        
        
        ## Round all fractional coordinates up to the nearest integer 
        ## While taking care of negative values to round down
        sign = np.sign(frac_coords)
        frac_coords = np.ceil(np.abs(frac_coords))
        frac_coords *= sign
        
        unique = np.unique(frac_coords, axis=0)
        unique_cart = np.dot(unique, lv)
        
        ## Need to fill in grid
        max_norm = np.max(np.linalg.norm(unique_cart, axis=-1))
        max_idx = np.max(np.abs(frac_coords))
        all_idx = self.prep_idx(max_idx, -max_idx)
        all_idx_cart = np.dot(all_idx, lv)
        all_norm = np.linalg.norm(all_idx_cart, axis=-1)
        take_idx = np.where(all_norm <= max_norm)[0]
        
        all_cart_points = all_idx_cart[take_idx]
        all_frac_points = self.struct.cart2frac(all_cart_points)
        
        ## For visualization
        self.lattice_points = all_cart_points
        
        ## Constructure supercell structure
        supercell_geo = np.zeros((geo.shape[0]*all_cart_points.shape[0],3))
        supercell_ele = np.repeat(ele[:,None], all_cart_points.shape[0],axis=-1)
        supercell_ele = supercell_ele.T.reshape(-1)
        supercell_frac_trans = []
        for idx,trans in enumerate(all_cart_points):
            start = idx*geo.shape[0]
            end = (idx+1)*geo.shape[0]
            supercell_geo[start:end,:] = geo+trans
            ### Track frac that was used for translating each atom
            temp_frac_trans = [tuple(all_frac_points[idx]) for x in geo]
            supercell_frac_trans += temp_frac_trans
            
        
        supercell_struct = Structure.from_geo(supercell_geo, supercell_ele)
        
        if len(self.molecule_idx) > 0:
            ### First get ordering of molecules wrt geometry
            molecule_image_idx = np.zeros((geo.shape[0],))
            for temp_mol_idx,entry in enumerate(self.molecule_idx):
                molecule_image_idx[entry] = temp_mol_idx
            ### Due to the way the supercell geometry is constructed, the 
            ### molecule_idx for the geometry just needs to be repeated
            ### the same number of times as the number of translations
            ### which is easily accomplished with np.tile
            supercell_geo_molecule_idx = np.tile(molecule_image_idx,
                                                 (len(all_cart_points),))
            supercell_struct.properties["molecule_image_idx"] = \
                                        supercell_geo_molecule_idx.astype(int)
                                        
            ### Construct the molecule_idx list for the supercell using the 
            ### molecule_idx of the unit cell
            total = 0
            supercell_molecule_idx = []
            for x in range(len(all_cart_points)):
                for mol_idx in self.molecule_idx:
                    adjusted_idx = np.array(mol_idx) + total
                    supercell_molecule_idx.append(adjusted_idx.tolist())
                    
                ### Go to next unit cell in supercell
                total += geo.shape[0]
            supercell_struct.properties["molecule_idx"] = supercell_molecule_idx
        
        supercell_frac_trans = np.array(supercell_frac_trans).round()
        supercell_struct.properties["frac_trans"] = supercell_frac_trans.tolist()
            
        return supercell_struct
        
    
    def prep_idx(self, max_idx, min_idx):
        """
        Return a list of all index permutations between a max index and 
        minimum index. 
        
        """
        if int(max_idx) != max_idx or \
           int(min_idx) != min_idx:
               raise Exception("Max index {} and min index {} "
                    .format(max_idx,min_idx)+
                    "cannot safely be converted to integers.")
        
        idx_range = np.arange(min_idx, max_idx+1)[::-1]
        
        ## Sort idx_range array so the final list is sorted by magnitude 
        ## so that lower index, and positive index, planes are given preference
        sort_idx = np.argsort(np.abs(idx_range))
        idx_range = idx_range[sort_idx]
        return np.array(
                np.meshgrid(idx_range,idx_range,idx_range)).T.reshape(-1,3)


if __name__ == "__main__":
    pass
