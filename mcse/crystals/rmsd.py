# -*- coding: utf-8 -*-

import copy,time
import numpy as np 

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation 

from ase.data import atomic_numbers,atomic_masses_iupac2016

from mcse import Structure
from mcse.core.driver import PairwiseDriver
from mcse.core.utils import combine
from mcse.crystals import com_supercell,standardize,SupercellSphere,Supercell
from mcse.molecules.align import *
from mcse.molecules.utils import rot_mol
from mcse.molecules.symmetry import get_mirror_symmetry
from mcse.molecules.rmsd import rmsd as molecule_rmsd        
        
        
class RMSD(PairwiseDriver):
    """
    Class for finding the best possible match between two molecular crystal 
    structures. Extracsts nearest neighbor molecular clusters structure 1 and 
    attempts to choose the molecular cluster from structure 2 that best matches.
    
    
    Arguments
    ---------
    nn: int
        Number of nearest neighbors to match from struct1 and struct2
    search_nn: int
        Number of nearest neighbors to use from struct2 to search for the best
        match to struct1 in terms of relative COM positions. 
    radius: float 
        Radius used to construct supercell to search over. 
    mult: int
        If greater than zero, used to compute the appropriate radius for the 
        structure. See SupercellSphere class. 
    stol: float
        Tolerance per site to allow for differences in COM matching. 
    stol_mult: float
        If used, this overrides stol if it's greater than stol. This will 
        cause the stol cutoff will be set based on the minimum stol found 
        during execution multiplied by this value. 
    H: bool
        If True, will include hydrogen positions when matching structures. 
        If False, matching structures will be performed without considering
        the hydrogen positions. 
    color: bool
        If True, struct1 elements will all be switched to nitrogen and the 
        struct2 elements will all be switched to oxygen. This can help aid in
        the visualization of the aligned structures. 
    use_first: bool
        If True, will just use the first, best matching COM from the candidate
        crystal rather than comparing all within the stol.
    
    """
    def __init__(self, nn=8, search_nn=10, radius=10, mult=0, 
                 stol=0.5, stol_mult=1.25, tol=0.2, use_first=False,
                 stol_cutoff = 2, 
                 H=False, color=False, **kwargs):
        if type(nn) != int:
            raise Exception("nn argument must be int")
        if type(search_nn) != int:
            raise Exception("search_nn argument must be int")
        if stol < 0:
            raise Exception("stol must be greater than zero")
        if nn < 0:
            raise Exception("nn argument must be greater than zero")
        
        ## If user makes a bad setting choice, correct it for them using 
        ## reasonable adjustments.
        if search_nn < nn:
            if nn < 10:
                search_nn = nn+2
            elif nn > 10 and nn <= 20:
                search_nn = nn+4
            else:
                search_nn = nn+6
            
        self.nn = nn
        self.search_nn = search_nn
        self.radius = radius
        self.mult = mult
        self.stol = stol
        self.stol_mult = stol_mult
        self.H = H
        self.color = color
        self.use_first = use_first
        self.stol_cutoff = stol_cutoff
        self.tol = tol
        
        super().__init__(**kwargs)
    
    
    def calc_struct(self, struct1, struct2, pre=False):
        """
        Main method for matching structures. Specifically, struct1 will not be 
        modified in any way. The best fit between the molecules in the unit
        cell of struct1 and the supercell of struct2 will be found. 
        
        Algorithm is as follows:
            1) Find the vectors between the COM in struct1 by transforming
                the nearest neighbor pairs into the unit cell of struct1.
            2) Build a center of mass supercell of struct2 in order to find
                the best matching to struct1.
        
        """
        self.struct1 = copy.deepcopy(struct1)
        self.struct2 = copy.deepcopy(struct2)
        
        if pre:
            self.struct1 = standardize(struct1)
            self.struct2 = standardize(struct2)
            
        if not self.H:
            keep_idx = np.where(self.struct1.elements != "H")
            self.struct1 = self.struct1.get_sub(keep_idx, lattice=True)
            self.struct1.get_molecule_idx(**{'mult': 1.2, 'skin': 0.0, 'update': True})
            struct1 = self.struct1
            
            keep_idx = np.where(self.struct2.elements != "H")
            self.struct2 = self.struct2.get_sub(keep_idx, lattice=True)
            self.struct2.get_molecule_idx(**{'mult': 1.2, 'skin': 0.0, 'update': True})
            struct2 = self.struct2
            
        self.target_com_vector = self.get_target_com_vector()
        
        com_list = np.vstack([com(x) for x in self.struct1.molecules.values()])
        sort_idx = np.argsort(np.linalg.norm(com_list, axis=-1))
        min_com = com_list[sort_idx[0]]
        self.struct1.translate(-min_com)
        
        ### At the same time, want to construct a supercell of struct2 that 
        ### can be references later for RMSD calculations once the best COM fit 
        ### can be found. 
        sphere = SupercellSphere(radius=self.radius, 
                                 mult=self.mult,
                                 correction=False)
        self.sphere = sphere
        
        self.struct2_supercell = sphere.calc_struct(struct2)
        self.struct2_supercell_mol = [x for x in 
                        self.struct2_supercell.molecules.values()]
        self.struct2_supercell_com = np.vstack([com(mol) for mol in 
                                      self.struct2_supercell_mol])     
        
        ### Build KDTree to match the COM distances observed in struct1 with 
        ### the COM supercell of struct2.
        ### Should only ever have to look up the nearest neighbors equal to 
        ## the number of molecules in struct1
        # data = self.com_supercell.get_geo_array()
        data = self.struct2_supercell_com
        self.com_tree = cKDTree(data,
                               leafsize=self.search_nn+2)    
        
        ## This returns all nearest neighbor distances 
        nn_dist,nn_idx = self.com_tree.query(self.com_tree.data, 
                                              self.search_nn+1)
        
        
        ## nn_vectors will refer to the difference in vectors between the 
        ## first COM for nn_idx and all other values
        nn_positions = self.com_tree.data[nn_idx]
        nn_vectors = []
        for entry in nn_positions:
            temp_nn_vector = entry[1:] - entry[0]
            nn_vectors.append([temp_nn_vector])
        nn_vectors = np.vstack(nn_vectors)
        
        ### Remove duplicate candidate clusters based on COM pairwise distances
        self.nn_vectors = nn_vectors
        diff_matrix_tensor = []
        for entry in nn_vectors:
            temp_dist = np.linalg.norm(entry[1:] - entry[0], axis=-1)
            sort_idx = np.hstack([[0],np.argsort(temp_dist)+1])
            entry = entry[sort_idx]
            diff_matrix_tensor.append(np.linalg.norm(entry[:,None] - entry, axis=-1))
        diff_matrix_tensor = np.array(diff_matrix_tensor)
        R,C = np.triu_indices(nn_vectors.shape[0],1)
        mask = (np.abs(diff_matrix_tensor[R] - diff_matrix_tensor[C]) < 1e-3).all(axis=(1,2))
        I,G = R[mask], C[mask]
        remove_idx = np.unique(G)
        original_idx = np.arange(0,nn_vectors.shape[0])
        final_idx = np.setdiff1d(original_idx,remove_idx)
        
        nn_dist = nn_dist[final_idx]
        nn_idx = nn_idx[final_idx]
        nn_vectors = np.linalg.norm(nn_vectors, axis=-1)
        nn_vectors = nn_vectors[final_idx]
        self.test_nn_vectors = nn_vectors
        
        ## Now match vectors and calculate difference
        self.nn_vectors = []
        self.nn_idx = []
        [self.calc_best_match(x) for x in nn_vectors]
        self.nn_vectors = np.vstack(self.nn_vectors)
        self.nn_idx = np.vstack(self.nn_idx)
        
        ## Calculate differences between target and matched vectors
        diff = self.target_com_vector - self.nn_vectors
        diff_norm = np.linalg.norm(diff,axis=-1)
        
        self.diff_norm = diff_norm
        min_diff_norm = np.min(self.diff_norm)
        self.test_list = []
        if self.stol_mult > 0:
            stol_cutoff = min_diff_norm * self.stol_mult
            if stol_cutoff > self.stol:
                temp_stol = stol_cutoff
            else:
                temp_stol = self.stol
        else:
            temp_stol = self.stol
        self.temp_stol = temp_stol
        self.stol_idx = np.where(diff_norm < temp_stol)[0]
        self.stol_idx = self.stol_idx[np.argsort(diff_norm[self.stol_idx])]
        
        if np.min(diff_norm) > self.stol_cutoff:
            cutoff = len(np.where(diff_norm - np.min(diff_norm) < 0.01)[0])
        else:
            cutoff = len(self.stol_idx)
        
        if len(self.stol_idx) == 0:
            raise Exception("No Match Found, "+
                            "try increasing stol")

        ## Construct and calculate RMSD for each within stol
        rmsd_list = []
        for iter_idx in range(len(self.stol_idx)):
            ### Dereference idx with respect to stol_idx
            idx = self.stol_idx[iter_idx]
            
            ### Reconstruct using stored supercell is more stable
            com_idx = nn_idx[idx][self.nn_idx[idx]]
            combined = self.struct2_supercell_mol[com_idx[0]]
            for temp_com_idx in com_idx[1:]:
                combined = combine(combined, 
                                   self.struct2_supercell_mol[temp_com_idx])
            
            combined.translate(-self.com_tree.data[com_idx[0]])
            self.combined = combined
            self.test_list.append(combined)
            
            rmsd_list.append(self.calc_rmsd(self.struct1, combined))
            
            if self.use_first:
                break
            
            if rmsd_list[-1] < self.tol:
                break
            
            ### If the stol is very large, there's no reason to evaluate so 
            ### many positions
            if iter_idx > cutoff:
                break
            
        self.rmsd_list = rmsd_list
        min_idx = self.stol_idx[np.argmin(rmsd_list)]
        
        com_idx = nn_idx[min_idx][self.nn_idx[min_idx]]
        cluster_from_2 = self.struct2_supercell_mol[com_idx[0]]
        for temp_com_idx in com_idx[1:]:
            cluster_from_2 = combine(
                cluster_from_2, self.struct2_supercell_mol[temp_com_idx])
        cluster_from_2.translate(-self.com_tree.data[com_idx[0]])
        cluster_from_2.struct_id = self.struct2.struct_id
        
        self.cluster_from_2 = cluster_from_2
        final_rmsd = self.calc_rmsd(self.struct1, cluster_from_2, store=True)
        if np.abs(final_rmsd - np.min(rmsd_list)) > 1e-3:
            print(np.min(rmsd_list), final_rmsd)
            raise Exception("This should not be able to happen")
        
        if self.color:
            self.struct1.elements = "O"
            self.cluster_from_2.elements = "N"
        
        self.combined = combine(self.struct1, self.cluster_from_2)
        
        ## Save RMSD in combined structure
        self.combined.properties["RMSD"] = final_rmsd
        
        self.struct = self.combined
        self.overlapping_clusters = self.combined
        
        return self.combined.properties["RMSD"]
            
            
    def calc_rmsd(self, struct1, struct2, store=False, symmetry=True, rematch=True):
        """
        Different here is that the molecules in the center are first aligned
        and then molecules from struct2 are removed to provide the best match. 
        Finally, a final unconstrained rotation is allowed to finalize the 
        the best match. 
        
        """
        if not store:
            temp1 = struct1.copy()
            temp2 = struct2.copy()
            # temp1 = copy.deepcopy(struct1)
            # temp2 = copy.deepcopy(struct2)
        else:
            temp1 = struct1
            temp2 = struct2
        
        geo1 = temp1.get_geo_array()
        geo2 = temp2.get_geo_array()
        ele1 = temp1.elements
        ele2 = temp2.elements
        
        #### Align the central molecules first
        mol_idx1 = temp1.get_molecule_idx(**{'mult': 1.2, 'skin': 0.0, 'update': False})
        mol_idx2 = temp2.get_molecule_idx(**{'mult': 1.2, 'skin': 0.0, 'update': False})
        
        self.mol_idx1 = mol_idx1
        self.mol_idx2 = mol_idx2
        
        com_list1 = []
        mol_list1 = []
        for mol_idx in mol_idx1:
            temp_mol = Structure.from_geo(geo1[mol_idx], ele1[mol_idx])
            temp_com = com(temp_mol)
            com_list1.append(temp_com)
            mol_list1.append(temp_mol)
        com_list1 = np.vstack(com_list1)
        self.com_list1 = com_list1
        
        com_list2 = []
        mol_list2 = []
        for mol_idx in mol_idx2:
            temp_mol = Structure.from_geo(geo2[mol_idx], ele2[mol_idx])
            temp_com = com(temp_mol)
            com_list2.append(temp_com)
            mol_list2.append(temp_mol)
        com_list2 = np.vstack(com_list2)
        self.com_list2 = com_list2
        
        central_mol_idx = np.argmin(np.linalg.norm(self.com_list2, axis=-1))
        
        ### Before rotation, temp1 & temp2 must be centered at the COM of the 
        ### given molecule
        temp1.translate(-com_list1[0])
        temp2.translate(-com_list2[central_mol_idx])
        
        if rematch:
            ### Want to skip this after appling symmetry operations
            pa1,pa2 = match_principal_axes(mol_list1[0], 
                                           mol_list2[central_mol_idx], 
                                           pre_align=False)
            temp1.rotate(pa1)
            temp2.rotate(pa2)
        
        ### Symmetry is check after already aligning the central molecules
        ### with the origin by rotating each by pa1 and pa2
        if symmetry:
            ### Call multiple times and account for all symmetry operations
            ### of the central molecule that are not rotations
            
            ### All symmetry operations of the central molecule should be 
            ### allowed in the analysis
            
            test_mol = mol_list2[central_mol_idx].copy()
            test_mol = fast_align(test_mol)
            sym_ops = get_mirror_symmetry(test_mol, tol=0.3)
            self.sym_ops = sym_ops
            
            result_list = []
            for sym_op in sym_ops:
                temp_temp2 = temp2.copy()
                temp_temp2 = rot_mol(sym_op, temp_temp2)
                temp_result = self.calc_rmsd(temp1, 
                                             temp_temp2, 
                                             store=False, 
                                             symmetry=False,
                                             rematch=False)
                result_list.append(temp_result)
            
            self.result_list = result_list
            
            if store:
                min_idx = np.argmin(result_list)
                best_sym_op = sym_ops[min_idx]
                temp_2 = rot_mol(best_sym_op, temp2)
                
                ### Now finish by doing the rest of the algorithm again
                pass
            else:
                return np.min(result_list)
        
        geo1 = temp1.get_geo_array()
        geo2 = temp2.get_geo_array()
        ele1 = temp1.elements
        ele2 = temp2.elements
        
        com_list1 = []
        mol_list1 = []
        for mol_idx in mol_idx1:
            temp_mol = Structure.from_geo(geo1[mol_idx], ele1[mol_idx])
            temp_com = com(temp_mol)
            com_list1.append(temp_com)
            mol_list1.append(temp_mol)
        com_list1 = np.vstack(com_list1)
        self.com_list1 = com_list1
        self.mol_list1 = mol_list1
        
        com_list2 = []
        mol_list2 = []
        for mol_idx in mol_idx2:
            temp_mol = Structure.from_geo(geo2[mol_idx], ele2[mol_idx])
            temp_com = com(temp_mol)
            com_list2.append(temp_com)
            mol_list2.append(temp_mol)
        com_list2 = np.vstack(com_list2)
        self.com_list2 = com_list2
        
        central_mol_idx = np.argmin(np.linalg.norm(self.com_list2))
        self.mol1 = mol_list1[0]
        self.mol2 = mol_list2[central_mol_idx]
        self.temp1 = temp1
        self.temp2 = temp2
        
        com_list2 = []
        mol_list2 = []
        for mol_idx in mol_idx2:
            temp_mol = Structure.from_geo(geo2[mol_idx], ele2[mol_idx])
            temp_com = com(temp_mol)
            com_list2.append(temp_com)
            mol_list2.append(temp_mol)
        com_list2 = np.vstack(com_list2)
        self.com_list2 = com_list2
        self.com_list2_vecs = com_list2[1:]
        self.mol_list2 = mol_list2
        
        ### Constrain the RMSD calculation to be only between molecules such that
        ### atoms from more than two molecules can never be compared 
        com_mask = np.ones((len(com_list2),))
        selected_rmsd = 0
        final_temp2_struct_geo = []
        final_temp2_struct_ele = []
        self.selected_com = []
        self.total_rmsd_list = []
        self.total_com_list = []
        self.total_com_list = []
        for idx,mol in enumerate(mol_list1):
            current_com = com(mol)
            current_geo = mol.get_geo_array()
            
            com_diff = np.linalg.norm(current_com - com_list2, axis=-1)
            com_diff = com_diff[com_mask.astype(bool)]
            com_idx = np.where(com_mask == 1)[0]
            
            sort_com_idx = np.argsort(com_diff)
            
            mol_rmsd_list = []
            idx2_list = []
            for temp_sort_mol2_idx in sort_com_idx:
                ### de-reference the sort_idx wrt com_idx
                temp_mol2_idx = com_idx[temp_sort_mol2_idx]
                temp_mol2 = mol_list2[temp_mol2_idx]
                temp_mol2_geo = temp_mol2.get_geo_array()
                
                dist = cdist(current_geo,temp_mol2_geo)
                idx1,idx2 = linear_sum_assignment(dist)
                
                temp_mol2_geo = temp_mol2_geo[idx2]
                
                temp_result = np.mean(np.linalg.norm(current_geo - temp_mol2_geo,
                                                axis=-1))
                
                mol_rmsd_list.append(temp_result)
                idx2_list.append(idx2)
        
            selected_mol_idx = np.argmin(mol_rmsd_list)
            selected_rmsd += mol_rmsd_list[selected_mol_idx]
            final_selected_mol_idx = com_idx[sort_com_idx[selected_mol_idx]]
            com_mask[final_selected_mol_idx] = 0
            
            ### Build final_temp2_struct
            selected_mol2 = mol_list2[final_selected_mol_idx]
            selected_mol2_geo = selected_mol2.get_geo_array()
            selected_mol2_ele = selected_mol2.elements
            selected_mol2_geo = selected_mol2_geo[idx2_list[selected_mol_idx]]
            selected_mol2_ele = selected_mol2_ele[idx2_list[selected_mol_idx]]
            
            final_temp2_struct_geo.append(selected_mol2_geo)
            final_temp2_struct_ele.append(selected_mol2_ele)
            
            self.selected_com.append(com_list2[com_idx[selected_mol_idx]])
            self.total_rmsd_list.append(mol_rmsd_list)
            self.total_com_list.append(com_idx[sort_com_idx])
        
        geo2 = np.vstack(final_temp2_struct_geo)
        ele2 = np.hstack(final_temp2_struct_ele)
        result = np.mean(np.linalg.norm(geo1 - geo2, axis=-1))
                
        if store:
            struct1.from_geo_array(geo1,ele1)
            struct2.from_geo_array(geo2,ele2)
            struct2.get_molecule_idx(**{'mult': 1.2, 'skin': 0.0, 'update': True})
            self.struct2 = struct2
            self.final_geo = geo2
            self.final_ele = ele2
            self.final_idx1 = idx1
            self.final_idx2 = idx2
        
        return result
    
    
    def get_target_com_vector(self):
        """
        Get the target COM vectors from struct1 by finding the nearest neighbor
        cluster based on COM in the supercell of struct1.
        
        """        
        sphere = SupercellSphere(radius=self.radius, 
                                 mult=self.mult,
                                 correction=False)
        supercell = sphere.calc_struct(self.struct1)
        self.supercell = supercell
        
        ## Get nearest COM neighbors
        # data = struct1_com_supercell.get_geo_array()
        geo = supercell.get_geo_array()
        ele = supercell.elements
        data = []
        for mol_idx in supercell.properties["molecule_idx"]:
            temp_mol = Structure.from_geo(geo[mol_idx], ele[mol_idx])
            data.append(com(temp_mol))
        data = np.vstack(data)
            
        temp_com_tree = cKDTree(data,
                                leafsize=self.nn+1)
        dist,idx = temp_com_tree.query(temp_com_tree.data, self.nn+1)
        dist_sum = np.sum(dist,axis=-1)
        dist_idx = np.argmin(dist_sum)
        
        ## Get COM positions and translate the first molecule to the origin of 
        ## the structure.
        com_positions = temp_com_tree.data[idx[dist_idx]]
                
        ### Modify struct1 to be the extracted molecular cluster
        kept_com_idx = idx[dist_idx]
        # kept_mol_idx = [supercell.properties["molecule_idx"][x] for x in kept_com_idx]
        # temp_mol_list = []
        
        # self.kept_com_idx = kept_com_idx
        # self.kept_mol_idx = kept_mol_idx
        
        
        # self.struct1 = Structure()
        # for mol_idx in kept_mol_idx:
        #     temp_mol = Structure.from_geo(geo[mol_idx], ele[mol_idx])
        #     temp_mol_list.append(temp_mol)
        #     self.temp_mol = temp_mol
        #     self.struct1 = combine(self.struct1, temp_mol)
        # self.struct1.translate(-com(temp_mol_list[0]))
        
        self.struct1 = Structure()
        mol_struct_list = [x for x in supercell.molecules.values()]
        mol_com_list = []
        for mol_idx in kept_com_idx:
            mol_com_list.append(com(mol_struct_list[mol_idx]))
            self.struct1.append(mol_struct_list[mol_idx])
        self.struct1.translate(-mol_com_list[0])
        
        # mol_struct_list = [x for x in self.struct1.molecules.values()]
        # mol_com_list = [com(x) for x in mol_struct_list]
        self.target_mol_com_list = np.vstack(mol_com_list)
        
        test_com_vector = np.array(mol_com_list[1:])
        
        ## Get COM vectors to match
        com_vector = com_positions[1:] - com_positions[0]
        
        self.com_vector = com_vector
        self.mol_com_list = np.vstack(mol_com_list)
        self.mol_com_vector = self.mol_com_list[1:] - self.mol_com_list[0]
        
        if np.linalg.norm(com_vector - self.mol_com_vector) > 1e-2:
            # print(np.vstack(mol_com_list))
            # print(com_vector)
            # print(test_com_vector)
            raise Exception("This shouldn't be able to happen")
            
        com_vector = np.sort(np.linalg.norm(com_vector, axis=-1))
        
        return com_vector
    
    
    def calc_best_match(self, temp_nn_vectors):
        """
        Finds best match between the com_vectors of struct1 and struct2.
        
        """
        target = self.target_com_vector
        dist = cdist(target[:,None], temp_nn_vectors[:,None])
        idx1,idx2 = linear_sum_assignment(dist)
        
        ## Keep only values in the original self.target
        idx1 = idx1[:self.nn]
        idx2 = idx2[:self.nn]
        
        ## Only want to keep the values in the original self.target
        target = target[idx1]
        nn_vectors = temp_nn_vectors[idx2]
        
        self.nn_vectors.append([nn_vectors])
        ## nn_idx is used later to index into the original nn_idx. So it is 
        ## saved accoridingly. Remember that nn_vectors implicitly includes 
        ## COM 0
        temp_nn_idx = np.zeros((idx2.shape[0]+1))
        temp_nn_idx[1:] = idx2+1
        self.nn_idx.append([temp_nn_idx.astype(int)])
    
        
if __name__ == "__main__":
    pass    
    
        
    
        
        
        
        
        
