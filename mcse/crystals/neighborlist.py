# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ase.data import chemical_symbols,vdw_radii,atomic_numbers,covalent_radii

from mcse.core.driver import BaseDriver_
from mcse.molecules import com
from mcse.crystals.cell_list import supercell_cell_list,get_cell_neigh,move_inside_cell
from mcse.crystals import standardize as crystal_standardize


class MCSENeighborList(BaseDriver_):
    """
    Will perform similarly as the neighborlist method in ASE, however, 
    additional features have been implemented such as only constructing the 
    neighborlist for specific atoms in the system. This is a major advantage 
    for the adsorbate ACSF descriptors. 
    
    In addition, an improved API will be introduced such that cutoffs can be 
    generated on the fly. 

    Lastly, this is actually must faster then the currently implemented 
    neighborlist in ASE. Need to show this difference across a benchmarking 
    dataset. 
    
    Arguments
    ---------
    mult: float
        Multiplicity to use for covalent cutoff distances
    cutoffs: list 
        List of floats that provides the cutoffs for every atom in the input 
        structures. More options are available that provide an easier API to 
        build the cutoff list on the fly for each structure passed to calc_struct. 
        By default, if no cutoffs are provided, then the covalent cutoff distances
        will be used
    overlap: bool
        If True, then if the cutoff radii of two atoms overlap then this is 
          considered to be neighbors. For example, this is the definition
          used by chemistry for covalent radii and vdW radii. 
        If False, then the neighbors of an atom of interest is determined if 
          the other atoms are within by the cutoff distance of just the atom 
          of interest. 
    vdw: bool
        If False, covalent bond radii are used as the cutoff distances. 
        If True, vdw cutoff distances are used. 
    cutoff_radius: float
        If no cutoffs are provided and covalent cutoffs is false, then will 
        default to a constant cutoff distance
    atom_idx: list
        List of indices if only specific atoms should be calculated for each
        structure. This is particularly helpful for improving the calculation
        time of the adsorbate ACSF
    approximate: bool
        If approximate neighborlists should be used. Approximate means that 
        there may be too many neighbors, but there will never be any neighbors
        missing. If False, then the exact neighbors are given as determined by
        the cutoff settings. 
    cell_list: float
        Cell size, in fraction coordinates, for the cell-list algorithm. The 
        cell-list algorithms provides that this function scales O(A*N) instead
        of typical O(N^2) scaling. However, A is typically a large real number, 
        so large system sizes may be required to see improvements. However, 
        most molecular crystal systems are large enough that this is definately
        beneficial. 
    include_self_idx: bool
        If False, will not include an atom's own index in its neighborlist
    acsf: bool
        If True, will store the necessary data to use for the ACSF descriptor
        calculations. 
    unique_ele: list
        Only used in ACSF preparation. List of strings of unique elements that 
        will be found in each structure provided to calc_struct. 
    
    """
    def __init__(self, 
                 mult=1.05,
                 skin=0,
                 cutoffs=[], 
                 overlap=True, 
                 vdw=False, 
                 cutoff_radius=8, 
                 atom_idx=[],
                 cell_size=0.001, 
                 approximate=False,
                 include_self_idx=False,
                 acsf=False,
                 unique_ele=[],):
        self.mult = mult
        self.skin = skin
        if skin != 0:
            raise Exception("Skin not implemented yet")
        
        self.cutoffs = cutoffs
        self.overlap = overlap
        self.vdw = vdw
        self.cutoff_radius = cutoff_radius
        
        self.atom_idx = []
        
        self.cell_size = cell_size
        self.approximate = approximate
        self.include_self_idx = include_self_idx
        
        self.acsf = acsf
        self.unique_ele = unique_ele
        
        ### Initialize internal objects
        self.struct = None       ### Calculated structure
        self.idx = []            ### Neighbor idx of each atom
        self.sc = None           ### Supercell structure, used if periodic
        self.init_sc_idx = []    ### Initial approximate supercell neighborhoods
        self.sc_idx = []         ### Neighbor of each atom using supercell idx
        self.molecule_idx = []   ### Molecule idx object
        
        if self.vdw == True:
            ### Have to set overlap to True as well. Using vdW without overlap
            ### is not meaningful. 
            self.overlap = True
            
    
    def calc_struct(self, struct, atom_idx=[], offsets=False):
        if len(atom_idx) == 0:
            atom_idx = self.atom_idx
        
        ### Initialize internal objects
        self.struct = struct     ### Calculated structure
        self.idx = []            ### Neighbor idx of each atom
        self.sc = None           ### Supercell structure, used if periodic
        self.init_sc_idx = []    ### Initial approximate supercell neighborhoods
        self.sc_idx = []         ### Neighbor of each atom using supercell idx
        self.molecule_idx = []   ### Molecule idx object
        self.offsets_cart = []   ### Storage for offsets in cartesian coordinates
        
        if len(self.struct.get_lattice_vectors()) > 0:
            return self._periodic_calc_struct(struct, atom_idx, offsets)
        else:
            return self._nonperiodic_calc_struct(struct, atom_idx)
    
    
    def _periodic_calc_struct(self, struct, atom_idx=[], offsets=False):
        if len(atom_idx) == 0:
            atom_idx = self.atom_idx
        
        self.struct = struct
        
        move_inside_cell(self.struct)
        
        geo = self.struct.get_geo_array()
        ele = self.struct.elements
        cutoffs = np.array(self._get_cutoffs(self.struct))
        
        if len(cutoffs) != len(geo):
            raise Exception("Number of cutoffs must match the number of "+
                "atoms in the structure")
        
        num_atom = len(geo)
        if not self.overlap:
            max_radius = np.max(cutoffs)
        else:
            max_radius = np.max(cutoffs+cutoffs[:,None])
        if len(atom_idx) == 0:
            atom_idx = np.arange(0,num_atom)
        
        ### Returns centered supercell object with cell-list defined in properties
        self.sc = supercell_cell_list(
                        self.struct, self.cell_size, max_radius)
        ### Returns atom index for cell neighborhoods of the given atom_idx
        self.init_sc_idx = get_cell_neigh(
                        self.sc, atom_idx, self.cell_size, max_radius)
        reference_idx = self.sc.properties["original_idx"]
        
        ### If approximate neighbors are okay, then 10-30% of time can be saved here
        ### depending on system size
        if not self.approximate:  
            ### Otherwise, calculate exact distances and only keep those atoms within 
            ### the exact distance
            sc_geo = self.sc.get_geo_array()
            
            final_neigh_idx = []
            for iter_idx,temp_neigh_idx in enumerate(self.init_sc_idx):
                temp_atom_idx = atom_idx[iter_idx]
                temp_cutoff = cutoffs[temp_atom_idx]
                if self.overlap:
                    ### If covalent cutoffs, have to consider if the atoms 
                    ### covalent radii overlap
                    temp_cutoff = cutoffs[reference_idx[temp_neigh_idx]] + temp_cutoff
                atom_pos = sc_geo[temp_atom_idx]
                temp_neigh_geo = sc_geo[temp_neigh_idx]
                temp_dist = cdist(atom_pos[None,:], temp_neigh_geo)[0]
                temp_keep_idx = np.where(temp_dist < temp_cutoff)[0]
                final_neigh_idx.append(temp_neigh_idx[temp_keep_idx])
            
            self.sc_idx = final_neigh_idx
        else:
            self.sc_idx = self.init_sc_idx
        
        ### Finally, need to convert the supercell idx back to the atom 
        ### indices from the original unit cell
        self.idx = []
        if self.include_self_idx:
            for iter_idx,temp_idx in enumerate(self.sc_idx):
                self.idx.append(reference_idx[temp_idx])
        else:
            for iter_idx,temp_idx in enumerate(self.sc_idx):
                self.sc_idx[iter_idx] = temp_idx[1:]
                # temp_array = np.sort(reference_idx[temp_idx[1:]])
                temp_array = reference_idx[temp_idx[1:]]
                self.idx.append(temp_array)
                
        ### Handle storing ACSF data if that's required
        if self.acsf:
            self.store4acsf(self.struct, atom_idx)
    
        ### Obtain offsets from storage in supercell construction
        if offsets:
            self.offsets_cart = []
            sc_offsets_cart = self.sc.properties["original_supercell_trans"]
            for iter_idx,temp_atom_idx in enumerate(atom_idx):
                nidx_list = self.sc_idx[iter_idx]
                temp_offsets = sc_offsets_cart[nidx_list]
                temp_offsets -= sc_offsets_cart[temp_atom_idx]
                self.offsets_cart.append(temp_offsets)
            return self.idx,self.offsets_cart
        else:
            return self.idx
    
    
    def _nonperiodic_calc_struct(self, struct, atom_idx=[]):
        """
        Has not been step-by-step verified... Can verify using a dataset of 
        molecules and molecular clusters compared to ASE later on...
        
        """
        self.struct = struct
        geo = struct.get_geo_array()
        ele = struct.elements
        cutoffs = np.array(self._get_cutoffs(struct))
        
        if len(atom_idx) == 0:
            atom_idx = self.atom_idx
        if len(atom_idx) == 0:
            atom_idx = np.arange(0,len(geo)) 
        
        if len(cutoffs) != len(geo):
            raise Exception("Number of cutoffs must match the number of "+
                "atoms in the structure")

        if self.overlap:
            cutoff_matrix = cutoffs[:,None] + cutoffs
        else:
            cutoff_matrix = cutoffs[:,None] + np.zeros((len(cutoffs),))
        
        ### Although squareform is not the most efficient, it is much easier
        ### to implement
        dist = squareform(pdist(geo))
        keep_idx = np.where((dist - cutoff_matrix) <= 0)
        keep_idx = [x for x in zip(keep_idx[0], keep_idx[1])]
        
        if self.include_self_idx:
            final_idx = []
            for idx in range(len(geo)):
                final_idx.append([idx])
        else:
            final_idx = [[] for x in range(len(geo))]
        
        for temp_idx in keep_idx:
            if temp_idx[0] == temp_idx[1]:
                continue
            final_idx[temp_idx[0]].append(temp_idx[1])
        self.idx = [final_idx[x] for x in atom_idx]
        self.ti = final_idx
        
        ### Handle storing ACSF data if that's required
        if self.acsf:
            self.store4acsf(self.struct, atom_idx)
            
        return self.idx

    
    def update(self, struct):
        """
        Use this if the analysis is for sequential steps in a trajectory. Will
        speed up the building of neighbor lists for these cases. 
        
        For analyzing an entire trajectory, this will be an extremely efficient
        method. Analyzing the first entry in the trajectory might take 0.5-2 
        seconds, however, successive entries should be on the order of 
        1 ms in the best case and 500 ms in the worst case.
        
        In addition, finding lithium clusters is only finding neighborhoods for 
        a small number of atoms. So that should be taken advantage of as well.
        
        """
        ### Doesn't do anything special right now, just calls calc_struct
        self.calc_struct(struct)
    
    
    def get_neighbor_struct(self, single_atom_idx, lattice=False):
        """
        Returns the neighborhood around the given atom index
        
        """
        temp_periodic = self._check_calculated("get_neighbor_struct")
        
        if temp_periodic:
            neigh_idx = self.sc_idx[single_atom_idx]
            return self.sc.get_sub(neigh_idx, lattice=lattice)
        else:
            neigh_idx = self.idx[single_atom_idx]
            return self.struct.get_sub(neigh_idx, lattice=False)
    
    
    def whole_molecules(self):
        """
        Returns the structure stored in the class with a whole molecule 
        representation
        
        This takes ~10 ms for a system with 313 atoms.
        
        This takes >1 second for a structure with 6,128 atoms. Not too bad I 
        think (in particular compared to my old method). 
        
        75% of the time of this function, for a system with 6,128 atoms, is 
        spent on get_molecule_idx, which is a function that I have not written
        myself. This depends on an extrenal library, either scipy or networkx, 
        therefore there's no more meaningful optimization I can perform. 
        
        """
        if not self.overlap:
            raise Exception("Argument overlap must be True for get_whole_mol_trans")
        
        temp_periodic = self._check_calculated("whole_molecules", 
                                               required_periodic=True)
        
        trans = self.get_whole_mol_trans()
        self.struct.geometry += trans
        
        ### Make sure mol COM is inside the unit cell
        move_mol_com(self.struct, self.molecule_idx)
        
        return self.struct
    
    
    def get_whole_mol_trans(self):
        """
        At the same time, it should be very fast to get the translation of the 
        original structure that provides whole molecules from the already 
        stored information
                
        Will fix this later because it's not critical. But this will provide the 
        fastest method for reconstructing molecules on the same order of time
        as constructing the cell list for all atoms themselves (~50 ms or about
        20 structures every second... That's a huge upgrade to current 
        performance of the mcse implementation). Also, should test how this 
        algorithm scales with system size...
        
        """
        if not self.overlap:
            raise Exception(
                "Argument overlap must be True for get_whole_mol_trans")
        
        temp_periodic = self._check_calculated("get_whole_mol_trans", 
                                               required_periodic=True)
            
        molecule_idx = self.get_molecule_idx()
        supercell_trans = self.sc.properties["original_supercell_trans"]
        reference_idx = self.sc.properties["original_idx"]

        def build_trans(idx, 
                added_idx,
                supercell_geo,
                trans, 
                supercell_neigh_idx, 
                supercell_trans, 
                reference_idx,
                num_atoms,
                adjust=np.array([0.,0.,0.])):
            """
            Recursive function for building translation vector for molecule split
            across periodic boundary conditions

            Arguments
            ---------
            idx: int
                Index of current atom. Will start at 0 and traverse the molecule.
            added_idx: list
                Exact list of atom indices whose trans has already been calculated
            geo: np.ndarray
                2D array of the positions of the atoms inside the unit cell
            trans: np.ndarray
                2D array describing the current translations that will be required for 
                each atoms in the molecule.
            supercell_neigh_idx: list
                List of integers for the neighbors of the current atom. Should be given 
                from the supercell structure.
            supercell_trans: np.ndarray
                Translations from the original unit cell for all atoms in the supercell
                structure.
            reference_idx: list
                List of indices for each atom in the supercell structure that describes
                which unit cell atom index it is derived from
            num_atoms: int
                Number of atoms in the structure. Used to keep track if a neighbor
                comes from the original structure or from the supercell construction. 
            
            """
            current_neigh = supercell_neigh_idx[idx]
            
            for neigh in current_neigh:
                temp_ref_idx = reference_idx[neigh]
                if added_idx[temp_ref_idx] == 1:
                    continue
                trans[temp_ref_idx] = supercell_trans[neigh]+adjust
            
            for neigh in current_neigh:
                temp_ref_idx = reference_idx[neigh]
                if added_idx[temp_ref_idx] == 1:
                    continue
                added_idx[temp_ref_idx] = 1
                if neigh >= num_atoms:
                    ### If neigh comes from supercell construction
                    new_adjust = adjust + supercell_trans[neigh] - supercell_trans[temp_ref_idx]
                else:
                    new_adjust = np.array(adjust)
                build_trans(temp_ref_idx, 
                        added_idx,
                        supercell_geo,
                        trans, 
                        supercell_neigh_idx, 
                        supercell_trans, 
                        reference_idx,
                        num_atoms,
                        new_adjust)
        
        if self.include_self_idx:
            start_idx = 1
        else:
            start_idx = 0
            
        num_atoms = len(self.struct.geometry)
        geo = self.sc.get_geo_array()
        supercell_trans = np.array(self.sc.properties["original_supercell_trans"])
        supercell_neigh = [self.sc_idx[x][start_idx:] for x in range(len(self.sc_idx))]
        added = np.zeros((len(self.sc_idx),)).astype(int)
        reference_idx = self.sc.properties["original_idx"]
                
        trans = np.zeros((len(self.idx), 3))
        for iter_idx,temp_idx in enumerate(molecule_idx):
            added[temp_idx[0]] += 1
            ### Adjust supercell trans based on atom 0 because it's translation should
            ### be considered to be 0, that is the current position of the unit cell
            ### atom for temp_idx[0]
            stored_trans = np.array(supercell_trans[temp_idx[0]])
            supercell_trans -= supercell_trans[temp_idx[0]]
            build_trans(temp_idx[0], 
                        added, 
                        geo,
                        trans, 
                        supercell_neigh, 
                        supercell_trans, 
                        reference_idx,
                        num_atoms)
            ### Undo adjustment
            supercell_trans += stored_trans
        
        return trans
    
    
    def get_molecule_idx(self):
        """
        Takes only 8.64 ms for a system with 300 atoms
        Takes 834 ms for a system with 6,000 atoms. Bad scaling of is due to the
        construction of the graph from the adjacency matrix. Nothing I can fix. 
        
        Networkx implementation is about 0.1 seconds faster than scipy 

        """
        if not self.overlap:
            raise Exception("Argument overlap must be True for get_molecule_idx")
        
        temp_periodic = self._check_calculated("get_molecule_idx", 
                                               required_periodic=True)
        
        if self.include_self_idx:
            start_idx = 1
        else:
            start_idx = 0
        
        ## Build connectivity matrix
        bonds = [x[start_idx:] for x in self.idx]
        num_atoms = len(self.struct.geometry)
        adjacency = np.zeros((num_atoms, num_atoms))
        for atom_idx,bonded_idx_list in enumerate(bonds):
            adjacency[atom_idx,bonded_idx_list] = 1
        self.adj = adjacency
        
        # ### THIS TAKES A SECOND FOR 6,000 ATOMS, THAT'S SOOOO SLOW
        # graph = csr_matrix(adjacency)
        # n_components, component_list = connected_components(adjacency)
        # molecule_idx = [
        #         np.where(component_list == x)[0] for x in range(n_components)]
        
        graph = nx.convert_matrix.from_numpy_matrix(adjacency)
        molecule_idx = [list(x) for x in 
                        nx.algorithms.components.connected_components(graph)]
        molecule_idx = [np.sort(x) for x in molecule_idx]
        
        self.molecule_idx = molecule_idx
        self.struct.properties["molecule_idx"] = molecule_idx
        
        return molecule_idx
    
    
    def get_interactions(self, bonds_kw={}):
        """
        Returns only the intermolecular interactions. In addition, returns the
        translation that's found for the interaction in the potentially periodic
        system. 
        
        """
        temp_periodic = self._check_calculated("get_interactions")
        
        if temp_periodic:
            ### If periodic, have to use the supercell indices
            idx = self.sc_idx
        else:
            ### If non-periodic, can use the final indices
            idx = self.idx
        
        molecule_idx = self.struct.get_molecule_idx(**bonds_kw)
        
        mol_lookup = {}
        for temp_mol_idx,atom_idx_list in enumerate(molecule_idx):
            for temp_atom_idx in atom_idx_list:
                mol_lookup[temp_atom_idx] = temp_mol_idx
        
        inter_list = []
        for atom_idx,neigh_list in enumerate(idx):
            temp_mol_idx = mol_lookup[atom_idx]
            for temp_neigh_idx in neigh_list:
                temp_neigh_mol_idx = mol_lookup[temp_neigh_idx]
                if temp_mol_idx == temp_neigh_mol_idx:
                    continue
                else:
                    inter_list.append([atom_idx, temp_neigh_idx])
        
        trans = np.zeros((len(inter_list), 3))
        if not temp_periodic:   
            ### If structure was non-periodic, then translations are zero
            return inter_list,trans

        ### Computing the translations for the given interactions for 
        ### a periodic system
        raise Exception("Not implemented yet")
        
        
        return inter_list,trans
        
        
    
    def store4acsf(self, struct=None, atom_idx=[], unique_ele=[]):
        """
        Storing information in the style that the ACSF calculator is expecting
        in order to optimize the ACSF calculations. 
        
        This has been verified to give exactly the same result as the ASE 
        implementation (where the ASE implementation always uses cutoff*2 for 
        the actual radius from any single atom)
        
        """
        if struct == None:
            struct = self.struct
        
        if len(atom_idx) == 0:
            atom_idx = self.atom_idx
            
        if len(unique_ele) == 0:
            unique_ele = self.unique_ele
            
        if len(atom_idx) < len(self.sc_idx):
            raise Exception("Cannot use more atom_idx in store4acsf "+
                    "than provided to MCSENeighborList.calc_struct") 
        
        neighborlist = [[] for x in range(len(struct.geometry))]
        
        if len(unique_ele) == 0:
            ### Takes ~70 us for 323 atoms
            ### Takes ~2 ms for 6,128 atoms
            unique_ele = np.unique(struct.elements)
        
        temp_neighborele = {}
        for entry in unique_ele:
            temp_neighborele[entry] = {}
        neighborele = [dict(temp_neighborele) for x in range(len(struct.geometry))]
        
        ### Now that data structures have been prepared, they may be filled
        sc_ele = self.sc.geometry["element"]
        sc_geo = self.sc.get_geo_array()
        sc_trans = self.sc.properties["original_supercell_trans"]
        for iter_idx,atom_idx in enumerate(atom_idx):
            ### Get neighbor info and skip the atom itself
            temp_sc_neigh_idx = self.sc_idx[iter_idx][1:]
            
            ### Each atom has been relatively moved by CenteredSupercell and 
            ### this has to be undone
            temp_sc_neigh_pos = sc_geo[temp_sc_neigh_idx]
            temp_sc_neigh_pos -= sc_trans[atom_idx]
            temp_sc_neigh_ele = sc_ele[temp_sc_neigh_idx]
            
            ### Populate data structures
            neighborlist[atom_idx] = temp_sc_neigh_pos
            temp_neighborele = neighborele[atom_idx]
            for temp_ele in temp_neighborele.keys():
                temp_ele_idx = np.where(temp_sc_neigh_ele == temp_ele)[0]
                temp_neighborele[temp_ele] = temp_ele_idx
        
        struct.properties["neighborlist"] = neighborlist
        struct.properties["neighborele"] = neighborele
        
        return struct
        
    
    def _get_cutoffs(self, struct=None):
        if struct == None:
            struct = self.struct
        
        if len(self.cutoffs) != 0:
            cutoffs = self.cutoffs
        else:
            ele = struct.elements
            if self.overlap:
                if not self.vdw: 
                    cutoffs = [covalent_radii[atomic_numbers[x]]*self.mult for x in ele]
                else:
                    cutoffs = [vdw_radii[atomic_numbers[x]]*self.mult for x in ele]
            else:
                cutoffs = [self.cutoff_radius for x in ele]
                
        return cutoffs
    
    
    def _check_calculated(self, calling_method, required_periodic=None):
        """
        Checks if the necessary calculations have been performed already and
        returns whether or not the calculation was periodic
        
        """
        if self.struct == None:
            raise Exception("Must use calc_struct before call {}"
                            .format(calling_method))
            
        periodic = None
        if self.sc == None:
            periodic = False
        else:
            periodic = True
        
        if required_periodic != None:
            if required_periodic:
                if periodic:
                    return periodic
            else:
                if not periodic:
                    return periodic
            
            ### If fell to here, then the requirement was not met
            if required_periodic:
                raise Exception("Must call {} with a periodic structure"
                                .format(calling_method))
            else:
                raise Exception("Must call {} with a non-periodic structure"
                                .format(calling_method))
            
        return periodic
        

def box_struct(struct, vacuum=40):
    """
    Puts structure in a large vacuum box. 
    
    """
    ### Need to put the supercell in a vacuum box in order for the 
    ### neighborlist to make use of the efficient cell-list method
    geo = struct.get_geo_array()
    geo_min = np.min(geo, axis=0)
    geo_max = np.max(geo, axis=0)
    
    ### Put minimum geometry of supercell at origin
    struct.geometry -= geo_min
    geo = struct.get_geo_array()
    geo_min = np.array([0,0,0])
    geo_max = list(np.max(geo, axis=0))
    
    ### Construct box surrounding the supercell
    lv = [
        [geo_max[0]+vacuum, 0, 0],
        [0, geo_max[1]+vacuum, 0],
        [0, 0, geo_max[2]+vacuum],
        ]
    struct.set_lattice_vectors(lv)
    
    return struct,-geo_min


def move_mol_com(struct, molecule_idx=[]):
    if len(molecule_idx) == 0:
        if "molecule_idx" in struct.properties:
            molecule_idx = struct.properties["molecule_idx"]
        else:
            raise Exception("Please input molecule_idx to move_mol_com")
    
    mol_list = [struct.get_sub(x, lattice=False) for x in molecule_idx]
    com_list = np.vstack([com(x) for x in mol_list])
    
    lv = np.vstack(struct.get_lattice_vectors())
    lv_inv = np.linalg.inv(lv.T)
    
    com_frac = np.dot(lv_inv, com_list.T).T
    target_com_frac = com_frac % 1
    frac_trans = target_com_frac - com_frac
    cart_trans = np.dot(frac_trans, lv)
    
    geo = struct.get_geo_array()
    trans = np.zeros((len(struct.geometry), 3))
    for iter_idx,atom_idx in enumerate(molecule_idx):
        trans[atom_idx] += cart_trans[iter_idx]
    geo += trans
    struct.from_geo_array(geo, struct.elements)
    
    return struct


def get_inter(struct, vdw=[], max_sr=1.30, bonds_kw={}, standardize=True,
              mol_idx=[]):
    """
    Get inter-molecular interactions for a given molecular crystal structure
    
    """
    struct = struct.get_sub(np.arange(0,len(struct.geometry)))
    
    if len(vdw) == 0:
        vdw = vdw_radii
    
    if len(struct.get_lattice_vectors()) == 0:
        raise Exception("This function is for periodic structures")
    
    lv = np.vstack(struct.get_lattice_vectors())
    lv_inv = np.linalg.inv(lv.T)
        
    ### Obtain non-periodic molecular bonding
    if standardize:
        crystal_standardize(struct)
    
    non_periodic = struct.get_sub(np.arange(0,len(struct.geometry)), 
                                    lattice=False)
    if len(mol_idx) == 0:
        non_periodic_mol_idx = non_periodic.get_molecule_idx()
    else:
        non_periodic_mol_idx = mol_idx
    mol_com_list = []
    rel_pos_lookup = {}     ### Store relative position of each atom wrt com
                            ###   which will be used later to construct the 
                            ###   correct molecule indexing in the supercell
    
    for mol_idx in non_periodic_mol_idx:
        temp_mol = non_periodic.get_sub(mol_idx)
        temp_com = com(temp_mol)
        mol_com_list.append(temp_com)
        
        ### For each atom, compute the relative position to the com of the molecule
        temp_geo = temp_mol.get_geo_array()
        temp_rel_pos = temp_com - temp_geo
        
        for iter_idx,atom_idx in enumerate(mol_idx):
            rel_pos_lookup[atom_idx] = temp_rel_pos[iter_idx]

    mol_com_list = np.vstack(mol_com_list)
    
    ### Prepare neighborlist to be using vdW type distance calculations
    nl = MCSENeighborList(
        mult=max_sr,
        skin=0,
        cutoffs=[], 
        overlap=True, 
        vdw=True,
        cutoff_radius=8, 
        atom_idx=[],
        cell_size=0.001, 
        approximate=False,
        include_self_idx=False,
        acsf=False,
        unique_ele=[])
    
    neigh_idx = nl.calc_struct(struct)

    ### Use mol_idx from supercell is correct. 
    ### This allows a molecule interacting with its own image to be included
    ###   in the minimum sr calculation, which is the physically correct thing
    ###   to do.
    ### However, this does not allow the case that the molecule is actually 
    ###   connected to its own image. Therefore, this is a problem.
    ###   Therefore, the molecule idx of the supercell needs to be built in 
    ###   reference to the non-periodic geometry given that the non-periodic 
    ###   geometry contains valid molecules as in the case of genarris/GAtor
    ###   or any construction of a unit cell from simple molecules. 
    sc_ele = nl.sc.geometry["element"]
    sc_geo = nl.sc.get_geo_array()

    ### Datatypes to make fast lookups to what molecule each atom belongs to
    ### based on the non-periodic geometry
    sc_mol_lookup = {}
    for atom_idx,temp_oidx in enumerate(nl.sc.properties["original_idx"]):
        ### Used stored relative vector to COM of corresponding molecule in order
        ### to obtain the COM of the molecule that each atom belongs to
        temp_geo = sc_geo[atom_idx]
        temp_rel_vec = rel_pos_lookup[temp_oidx]
        temp_com = temp_geo + temp_rel_vec
        
        ### Find the relationship betweeen computed COM and COM from non-periodic
        ### in the basis of the unit cell lattice
        temp_com_diff = temp_com - mol_com_list
        temp_frac = np.dot(lv_inv, temp_com_diff.T).T
        
        ### Find idx that provides closest to integer values
        temp_int_diff = np.linalg.norm(temp_frac - np.round(temp_frac), axis=-1)
        temp_min_idx = np.argmin(temp_int_diff)
        if temp_int_diff[temp_min_idx] > 0.1:
            raise Exception("This should not be possible")
        
        ### This minimum idx corresponds to the molecule image and the rounded
        ### entry to the fractional translation. This can be stored for future use. 
        temp_final_frac = tuple(np.round(temp_frac[temp_min_idx]).astype(int))
        temp_key = (temp_min_idx, temp_final_frac)
        sc_mol_lookup[atom_idx] = temp_key
    
    keep_inter = []
    keep_inter_ref = []
    keep_inter_ele = []
    ref_idx = nl.sc.properties["original_idx"]
    for atom_idx,inter_idx_list in enumerate(nl.sc_idx):
        temp_mol_idx = sc_mol_lookup[atom_idx]
        for temp_inter_idx in inter_idx_list:
            temp_inter_mol_idx = sc_mol_lookup[temp_inter_idx]
            if temp_mol_idx == temp_inter_mol_idx:
                pass
            else:
                keep_inter.append((atom_idx,temp_inter_idx))
                keep_inter_ref.append((ref_idx[atom_idx], 
                                        ref_idx[temp_inter_idx]))
                keep_inter_ele.append((sc_ele[atom_idx], sc_ele[temp_inter_idx]))
    
        ### Prepare final distances and sr for returning
    geo = nl.sc.get_geo_array()
    sr_list = []
    dist_list = []
    for idx1,idx2 in keep_inter:
        temp_ele1 = sc_ele[idx1]
        temp_ele2 = sc_ele[idx2]
        temp_vdw1 = vdw_radii[atomic_numbers[temp_ele1]]
        temp_vdw2 = vdw_radii[atomic_numbers[temp_ele2]]
        temp_vdw = temp_vdw1+temp_vdw2
        temp_dist = np.linalg.norm(geo[idx1]-geo[idx2])
        temp_sr = temp_dist / temp_vdw
        dist_list.append(temp_dist)
        sr_list.append(temp_sr)
    
    return keep_inter_ele,keep_inter_ref,dist_list,sr_list
    
    

def get_min_sr(struct, vdw=[], max_sr=1.30, bonds_kw={}):
    """
    Get minimum inter-molecular sr for a given molecular crystal structure
    """
    if len(vdw) == 0:
        vdw = vdw_radii
    
    keep_inter_ele,_,_,sr_list = get_inter(struct, vdw, max_sr, bonds_kw)
        
    min_idx = np.argmin(sr_list)
    min_sr = sr_list[min_idx]
    min_ele = keep_inter_ele[min_idx]
    return min_sr,min_ele



    