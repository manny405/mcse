# -*- coding: utf-8 -*-


"""

Label the atoms of two molecules in a commensurate way. This will be implemented
using the atomic bonding neighborhoods of each atom and the molecular connectivity. 

In addition, it would be nice if there was a particular unique way of doing 
this such that it wouldn't require comparing two molecules, but that should
come later.

"""

import copy

import numpy as np 
import networkx as nx

from ase.data import vdw_radii,atomic_numbers
from ase.data.colors import jmol_colors


def label(mol, bonds_kw={"mult": 1.20, "skin": 0, "update": False}):
    """
    Orders the atoms in the molecule in a unique way using a graph
    representation. The initial order of the atoms will not impact the order
    of labelling of nodes in the molecular graph. Note that if a labelling is
    symmetrically equivalent based on a undirected and unweighted graphical
    representation of the molecule, then the label order may be dependent
    on the initial ordering of atoms. This is important for cases where side
    groups have the exact same chemical identity, but perhaps very different
    torsion angles. Unique labelling based on torsion angles, for example, 
    cannot be  handled directly by this function. In nearly all other
    cases, this method will provide a unique ordering of the atoms in the 
    molecule. 
    
    """
    return order_atoms(mol, bonds_kw=bonds_kw)


def order_atoms(mol, bonds_kw={"mult": 1.20, "skin": 0, "update": False}):
    """
    Purpose of this function is as follows . . . .
    The algorithm works as follows . . . .
    This is useful because . . . .
    
    Agruments
    ---------
    mol: Structure
        Structure object to order labels of
    bonds_kw: dict
        Dictionary for computing bonds
    
    """
    unique_path = traverse_mol(mol, bonds_kw, ret="idx")
    geo = mol.get_geo_array()
    ele = mol.elements
    mol.from_geo_array(geo[unique_path], ele[unique_path])
    
    ### Need to update bonds after changing the atomic ordering using lookup 
    ### dict of the bonds currently stored in the molecule
    bond_idx_lookup = {}
    for new_idx,former_idx in enumerate(unique_path):
        bond_idx_lookup[former_idx] = new_idx
    
    ### First the bonds have to be resorted
    mol.properties["bonds"] = [mol.properties["bonds"][x] for x in unique_path]
    ### Then indices have to change
    for idx1,bond_list in enumerate(mol.properties["bonds"]):
        for idx2,atom_idx in enumerate(bond_list):
            mol.properties["bonds"][idx1][idx2] = bond_idx_lookup[atom_idx]
        
    return mol


def match_labels(mol1, mol2, 
                 bonds_kw={"mult": 1.20,
                            "skin": 0,
                            "update": False},
                 warn=False):
    """
    
    Arguments
    ---------
    warn: bool
        If False, then an exception will be raised if the ordering of the 
        molecules cannot be matched exactly. If True, then only a warning will
        be raised. 
        
    """
    formula1 = mol1.formula
    formula2 = mol2.formula
    
    if formula1 != formula2:
        ### It might be possible to include some fuzzier implementation that 
        ### finds best match even if the labels are not identical
        ### But that should be done later
        raise Exception("Matching atom labels requires that the formulas "+
                "of the molecules is identical")
    
    ### Make sure that the correct bonds have been obtained based on input arg
    mol1.get_bonds(**bonds_kw)
    mol2.get_bonds(**bonds_kw)
    
    ### Just label the first molecule arbitrarily because the ordering of the
    ###     second molecule will be built to match this one
    ### Also, note that once the atom ordering is changed once, it will never
    ###     change again thus this is a very safe thing to do even if mol1 is 
    ###     fed into this function many multiples of times. There is just some
    ###     argument against this for the sake of inefficiency, but the label
    ###     algorithm should be very fasty anyways...
    mol1 = label(mol1)
    
    ### Any neighborhood list that matches this target list will be exactly
    ###     an identical ordering up-to the symmetry of the molecular graph
    target_list = traverse_mol(mol1, bonds_kw, ret="neigh")
    
    ### Start with label mol2 and then start traversing by possible starting pos
    mol2 = label(mol2)
    ### Just need to iterate over the possible starting positions for the 
    ###    second molecule until an exact match is found
    ele2 = mol2.elements
    ele_numbers = [atomic_numbers[x] for x in ele2]
    max_idx_list = np.where(ele_numbers == np.max(ele_numbers))[0]
    
    final_idx_list = []
    for start_idx,_ in enumerate(max_idx_list):
        candidate_list = traverse_mol(mol2, 
                                      start_idx=start_idx, 
                                      bonds_kw=bonds_kw, 
                                      ret="neigh")
        
        if candidate_list == target_list:
            final_idx_list = traverse_mol(mol2, 
                                      start_idx=start_idx, 
                                      bonds_kw=bonds_kw, 
                                      ret="idx")
            break
    
    if len(final_idx_list) == 0:
        if not warn:
            raise Exception("Atom ordering could not be matched for {} {}"
                            .format(mol1.struct_id, mol2.struct_id))
        else:
            print("Atom ordering could not be matched for {} {}"
                            .format(mol1.struct_id, mol2.struct_id))
            ### Shouldn't make any modification if this is the case
            return mol1,mol2
    
    ### Reorder mol2 before returning
    geo = mol2.get_geo_array()
    ele = mol2.elements
    mol2.from_geo_array(geo[final_idx_list], ele[final_idx_list])
    
    ### Need to update bonds after changing the atomic ordering using lookup 
    ### dict of the bonds currently stored in the molecule
    bond_idx_lookup = {}
    for new_idx,former_idx in enumerate(final_idx_list):
        bond_idx_lookup[former_idx] = new_idx
    
    ### First the bonds have to be resorted
    mol2.properties["bonds"] = [mol2.properties["bonds"][x] for x in final_idx_list]
    ### Then indices have to change
    for idx1,bond_list in enumerate(mol2.properties["bonds"]):
        for idx2,atom_idx in enumerate(bond_list):
            mol2.properties["bonds"][idx1][idx2] = bond_idx_lookup[atom_idx]
    
    return mol1,mol2


def mol_to_graph(mol, 
                 bonds_kw={"mult": 1.20,"skin": 0,"update": False}):
    ele = mol.elements
    bonds = mol.get_bonds(**bonds_kw)
    
    g = nx.Graph()
    ### Add node to graph for each atom in struct
    g.add_nodes_from(range(len(ele)))
    ### Add edges
    for i,bond_list in enumerate(bonds):
        [g.add_edge(i,x) for x in bond_list]
        
    ### Add neighborhoods to each atom of molecule
    neighbors = [[[]] for x in g.nodes]
    for i,idx_list in enumerate(neighbors):
        neighbor_list = [x for x in g.adj[i]]
        neighbor_list.append(i)
        idx_list[0] += neighbor_list
    sorted_list = _sort_neighborlist(mol, g, neighbors)
    mol.properties["neighbors"] = sorted_list
    
    fragment_list = [mol.elements[tuple(x)] for x in sorted_list]
    fragment_list = [''.join(x) for x in fragment_list]
    mol.properties["neighbor_str"] = fragment_list
    
    attr_dict = {}
    for idx in range(len(g)):
        attr_dict[idx] = {"ele": ele[idx], "neighborhood": fragment_list[idx]}
    nx.set_node_attributes(g, attr_dict)
    
    return g


def traverse_mol(mol,
                 bonds_kw={"mult": 1.20,"skin": 0,"update": False},
                 start_idx=0,
                 ret="idx",
                 ):
    """
    
    Arguments
    ---------
    start_idx: int
        This is used to index into the indices of atoms in the molecule that 
        all have the same maximum atomic number.
    
    """
    g = mol_to_graph(mol, bonds_kw)
    ele = mol.elements
    ele_numbers = [atomic_numbers[x] for x in ele]
    max_idx_list = np.where(ele_numbers == np.max(ele_numbers))[0]
    
    if start_idx == 0:
        start_idx = max_idx_list[0]
    else:
        if start_idx >= len(max_idx_list):
            start_idx = max_idx_list[-1]
        else:
            start_idx = max_idx_list[start_idx]
    
    # print("START IDX: {}".format(start_idx))
    
    path = ordered_traversal_of_molecule_graph(g, start_idx)
    
    if ret == "idx":
        return path
    elif ret == "neigh" or \
         ret == "neighbor" or \
         ret == "neighborhood" or \
         ret == "neighborhoods":
             return [mol.properties["neighbor_str"][x] for x in path]
    else:
        raise Exception("ret not known")
        
    return path


def basic_traversal_of_molecule_graph(graph, 
                                        node_idx, 
                                        current_path=None):
    """
    This recursive function returns a basic traversal of a molecular graph. 
    This traversal obeys the following rules:
        1. Locations may only be visited once
        2. All locations must be visted
        
    """
    if current_path == None:
        current_path = []
        
    path = [node_idx]
    current_path += [node_idx]
    neighbors = graph.adj[node_idx]
    
    ### Need to take care of cycles
    for entry in neighbors:
        if entry in current_path:
            continue
        neigh_path = basic_traversal_of_molecule_graph(graph, entry, current_path)
        if len(neigh_path) > 0:
            path += neigh_path
            current_path += path
    
    return path
    

def ordered_traversal_of_molecule_graph(graph, 
                                        node_idx, 
                                        current_path=None):
    """
    This algorithm works as follows:
        1. 
    
    """
    path = [node_idx]
    current_path = [node_idx]
    current_node_idx = node_idx
    current_path_idx = 0
    while True:
        add_site = add_one_for_ordered_traversal(graph, current_node_idx, current_path)
        # print("Path: {}; Add Site: {}".format(path, add_site))
        
        ### Check for the case where no new site has been added
        ### If no new site has been added go backwards in path list and try
        ### to add a new site again
        if len(add_site) == 1:
            current_path_idx -= 1
            ### Completion condition
            if current_path_idx < 0:
                if len(path) == len(graph):
                    # print("FINISH")
                    break
                else:
                    raise Exception("Error")
            current_node_idx = path[current_path_idx]
            continue
            
        path += [add_site[1]]
        current_path += [add_site[1]]
        current_path_idx = len(path)
        current_node_idx = path[current_path_idx - 1]
        
        
        if len(path) == len(graph):
            # print("NORMAL FINISH")
            break
    
    return path



def add_one_for_ordered_traversal(graph, 
                                        node_idx, 
                                        current_path=None):
    """
    This recursive function returns an ordered traversal of a molecular graph. 
    This traversal obeys the following rules:
        
        1. Locations may only be visited once
        2. All locations must be visted
        3. Locations are visited in the order in which the shortest path is 
            followed
            - If potential paths are identical in length, then the one that
                provides lightest total weight is followed
            - If the total weight of each path is identical (which would be
                the case for a molecule that contains any cycle) then the 
                path the provides the lightest first atom is chosen
            - If the lightest first atom is identical, then.............
            
    Recursive algorithm works as follows:
        1. Go from node to node until reaching a node that has no neighbors. 
        2. Once this node is reached, it returns itself back up the stack. 
        3. If a node only has a single path, this is also immediately returned
             up the stack. 
        4. Once a node is reach that has two possible paths, a choice is made 
            between the two competing paths. The path that is the shortest is
            automatically chosen... But this is actually not what I want. 
            
            What I want is that the path leading down is fully traversed and 
            then the path that provides the lightest direction is gone down first
            
            If both paths are then equal in weight (such as should be the case 
            for a cycle) then the the path that provides the most direct route
            to the heaviest group will be prefered. 
            
            If the paths are completely identical, then it should not matter 
            which one is chosen first from the perspective of a graph.
                
        
    """
    if current_path == None:
        current_path = []
    
    ### Make copy of input current_path
    current_path = [x for x in current_path]
        
    path = [node_idx]
    current_path += [node_idx]
    neighbors = graph.adj[node_idx]
    
    ### Build entire traversal list
    neigh_path_list = []
    for entry in neighbors:
        # print(node_idx, entry)
        if entry in current_path:
            continue
        neigh_path = add_one_for_ordered_traversal(graph, entry, current_path)
        if len(neigh_path) > 0:
            neigh_path_list.append(neigh_path)
            # print(node_idx, entry, neigh_path)
    
    ### Only a single option
    if len(neigh_path_list) == 1:
        if len(neigh_path_list[0]) == 1:
            path += neigh_path_list[0]
            return path
    elif len(neigh_path_list) == 0:
        return [node_idx]
    
    ### If there's more than single option, then an algorithm that seeks 
    ### to stich together the neighbor paths in a reasonable and unique way
    ### should be used
    
    neigh_list_sorted = _sort_neighbor_path_list(graph, neigh_path_list)
    
    # print("SORTED: ", neigh_list_sorted)
    path += neigh_list_sorted
    
    return path


def _sort_neighbor_path_list(graph, neigh_path_list):
    """
    Recursive algorithm for sorting the neigh_path_list. Sorts by following
    criteria:
        1. 
        
        X. For a cycle, find the largest detour from the most direct path in 
            the cycle
    
    """
    # print(neigh_path_list)
    
    if len(neigh_path_list) == 0:
        return []
    
    neigh_path_list = copy.deepcopy(neigh_path_list)
    final_path_list = []
    
    path_length = np.zeros((len(neigh_path_list))).astype(int)
    for idx,entry in enumerate(neigh_path_list):
        path_length[idx] = len(entry)
        
    min_idx = np.argwhere(path_length == np.min(path_length)).ravel()
    if len(min_idx) == 1:
        final_path_list += neigh_path_list[min_idx[0]]
        del(neigh_path_list[min_idx[0]])
    else:
        ### Take into account atomic weights
        path_weight = np.zeros((len(min_idx))).astype(int)
        path_weight_list = []
        for idx,entry in enumerate(min_idx):
            temp_path = neigh_path_list[entry]
            temp_weight_list = [atomic_numbers[graph.nodes[x]["ele"]] for x in temp_path]
            temp_weight = np.sum(temp_weight_list)
            path_weight[idx] = temp_weight
            path_weight_list.append(temp_weight_list)
        
        min_idx = np.argwhere(path_weight == np.min(path_weight)).ravel()
        if len(min_idx) == 1:
            final_path_list += neigh_path_list[min_idx[0]]
            del(neigh_path_list[min_idx[0]])
        
        ### If absolutely identical, then it should matter which is used first
        elif all(x==path_weight_list[0] for x in path_weight_list):
            final_path_list += neigh_path_list[min_idx[0]]
            del(neigh_path_list[min_idx[0]])
        else:
            # ### FOR TESTING
            # path_weight_list = [[6, 0, 1, 6, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1], 
            #                     [6, 1, 6, 1, 6, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 1],
            #                     [6, 1, 6, 1, 6, 1, 6, 1, 6, 6, 1, 6, 6, 1, 6, 1]]
            
            ### Total path weights are identical but not ordered the same
            ### Likely, this is because there's a cycle in the molecule
            # print("CYCLES?: {}".format(neigh_path_list))
            # print("TOTAL PATH WEIGHTS: {}".format(path_weight))
            # print("PATH WEGITH LIST: {}".format(path_weight_list))
            
            ### Just choose the one that encounters the heaviest atom first
            keep_mask = np.ones((len(path_weight_list,))).astype(int)
            chosen_idx = -1
            for idx in range(len(path_weight_list[0])):
                current_pos = np.array([x[idx] for x in path_weight_list])
                
                ### Mask off entries that have already been discarded
                current_pos = current_pos[keep_mask.astype(bool)]
                
                ### Keep track of idx that are now described by current_pos
                keep_idx = np.where(keep_mask == 1)[0]
                
                ### Get mask that describes which paths have maximum at this 
                ### position in the path
                max_avail_mask = current_pos == np.max(current_pos)
                max_avail_idx = np.where(max_avail_mask > 0)[0]
                
                ### Get those that need to be discarded
                add_to_discard_idx = np.where(max_avail_mask < 1)[0]
                ### De-reference add_to_discard_idx wrt the keep_idx and 
                ### add these to the keep_mask
                add_to_discard_idx = keep_idx[add_to_discard_idx]
                keep_mask[add_to_discard_idx] = 0
                
                ### If there's only one option left, then this is the optimal path
                if len(max_avail_idx) == 1:
                    ### De-reference max idx with the paths that have been 
                    ### kept, described by keep_idx
                    chosen_idx = keep_idx[max_avail_idx[0]]
                    break
             
            ### The only possible case that a path was not chosen, is that
            ### two of the paths in the lists are identical and optimal. 
            ### Therefore, just choose one of them (the first one)
            if chosen_idx < 0:
                keep_idx = np.where(keep_mask > 0)[0]
                chosen_idx = keep_idx[0]
                # print("MORE THAN ONE OPTIMAL PATH")
            
            # print("CHOSEN DUE TO INDEX: ", idx)
            # print("CHOSEN PATH IDX: ", chosen_idx)
            # print("KEEP MASK: ", keep_mask)
            # print("MAX IDX: ", max_idx)
            
            
            # raise Exception("NEED TO IMPLEMENT HERE")
            final_path_list += neigh_path_list[chosen_idx]
            del(neigh_path_list[chosen_idx])
            
    
    if len(neigh_path_list) != 0:
        final_path_list += _sort_neighbor_path_list(graph, neigh_path_list)
    
    return final_path_list


def _sort_neighborlist(mol, g, neighbor_list):
    """
    Sorts neighborlist according to definition in _calc_neighbors. Only 
    works for a radius of 1.
    
    Arguments
    ---------
    g: nx.Graph
    neighbor_list: list of int
        List of adjacent nodes plus the node itself as i
    
    """
    ele = mol.elements
    sorted_list_final = [[[]] for x in g.nodes]
    for i,temp in enumerate(neighbor_list):
        # Preparing things which aren't writting well 
        idx_list = temp[0]
        current_node = i
        
        terminal_groups = []
        bonded_groups = []
        for idx in idx_list:
            if g.degree(idx) == 1:
                terminal_groups.append(idx)
            else:
                bonded_groups.append(idx)
        
        terminal_ele = ele[terminal_groups]
        alphabet_idx = np.argsort(terminal_ele)
        terminal_groups = [terminal_groups[x] for x in alphabet_idx]
        
        sorted_list = terminal_groups
        if current_node not in terminal_groups:
            sorted_list.append(current_node)
            remove_idx = bonded_groups.index(current_node)
            del(bonded_groups[remove_idx])
        
        bonded_ele = ele[bonded_groups]
        alphabet_idx = np.argsort(bonded_ele)
        bonded_groups = [bonded_groups[x] for x in alphabet_idx]
        
        sorted_list += bonded_groups
        
        sorted_list_final[i][0] = sorted_list
    
    return sorted_list_final


def get_bond_fragments(mol, 
                       bonds_kw={"mult": 1.20,"skin": 0,"update": False},
                       return_counts=True):
    g = mol_to_graph(mol, bonds_kw=bonds_kw)
    neigh_list = []
    for node_idx in g.nodes:
        neigh_list.append(g.nodes[node_idx]["neighborhood"])
    return np.unique(neigh_list, return_counts=return_counts)


def draw_mol_graph(mol,  bonds_kw={"mult": 1.20, "skin": 0, "update": False}):   
    g = mol_to_graph(mol, bonds_kw=bonds_kw)
    ele = mol.elements
    idx = [atomic_numbers[x] for x in ele]
    colors = jmol_colors[idx]
    nx.draw(g, node_color=colors, with_labels=True)









    