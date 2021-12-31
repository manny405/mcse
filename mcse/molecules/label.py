# -*- coding: utf-8 -*-


"""

Label the atoms of two molecules in a commensurate way. This will be implemented
using the atomic bonding neighborhoods of each atom and the molecular connectivity. 

In addition, it would be nice if there was a particular unique way of doing 
this such that it wouldn't require comparing two molecules, but that should
come later.

"""

import copy,itertools

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


def mol2graph(mol, bonds_kw={"mult": 1.20,"skin": 0,"update": False}):
    return mol_to_graph(mol, bonds_kw)


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



def label_mp(mol, sort_method="hash"):
    """
    Will label the atoms of the molecule using the new message passing 
    method
    
    """
    g = mol2graph(mol)
    message_descriptors = message_passing_atom_descriptors(g)
    
    ### Now, just need to decide order based on the message_descriptors
    
    ### Simple method is just based on sorting resulting hashes to provide 
    ### a truely unique ordering (up to graph symmetry)
    if sort_method != "hash":
        raise Exception("Only the hash sort method is implemented")
   
    hash_list = []
    for _,message in message_descriptors.items():
        hash_list.append(hash(message))
    
    unique_path = np.argsort(hash_list)
    
    geo = mol.get_geo_array()
    ele = mol.geometry["element"]
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
    
    ### Recompute properties from graph
    mol_to_graph(mol)
        
    return mol
        

def get_hash_order(mol):
    g = mol2graph(mol)
    message_descriptors = message_passing_atom_descriptors(g)
    hash_list = []
    for _,message in message_descriptors.items():
        hash_list.append(hash(message))
    sort_idx = np.argsort(hash_list)
    return sort_idx


def match2graph(mol,target_graph):
    """
    Return whether or not the given molecule matches the target_graph. If it
    does not match, a empty list will be returned. If it does match, the 
    index that matches the atoms from the molecule to the target_graph will be
    given
    
    This new implementation will work much better for matching two molecules. 
    The performance should be greatly increased. 
    
    """
    ### First check formula
    target_graph_formula = {}
    target_graph_ele = []
    for node_idx in target_graph.nodes:
        target_graph_ele.append(target_graph.nodes[node_idx]["ele"])
    ele,counts = np.unique(target_graph_ele, return_counts=True)
    target_graph_formula = dict(zip(ele,counts))
    
    mol_formula = mol.formula()
    if mol_formula != target_graph_formula:
        return []
    
    ### Then check neigh strings
    target_graph_neigh = []
    for node_idx in target_graph.nodes:
        target_graph_neigh.append(target_graph.nodes[node_idx]["neighborhood"])
    neigh,counts = np.unique(target_graph_neigh, return_counts=True)
    target_graph_neigh = dict(zip(neigh,counts))
    
    input_graph = mol_to_graph(mol)
    input_neigh = []
    for node_idx in input_graph.nodes:
        input_neigh.append(input_graph.nodes[node_idx]["neighborhood"])
    neigh,counts = np.unique(input_neigh, return_counts=True)
    input_neigh = dict(zip(neigh,counts))
    
    if input_neigh != target_graph_neigh:
        return []
    
    ### If neighborhoods & counts match find the ordering that matches the 
    ### neighborhood orderings of the two graph
    
    ### Start by getting the message descriptors for each atom given the index
    target_md = message_passing_atom_descriptors(target_graph)
    input_md = message_passing_atom_descriptors(input_graph)

    ### Now, is as simple as building a lookup table that will take the input_md
    ### and transform to the target_md
    input_lookup_dict = {}
    for idx,message in input_md.items():
        ### Account for atoms that may have the same messages
        if message in input_lookup_dict:
            input_lookup_dict[message].append(idx)
        else:
            input_lookup_dict[message] = [idx]

    ### Now iterate through target, matching each target to an index in the 
    ### input molecule
    matching_indices = []
    for idx,message in target_md.items():
        if message not in input_lookup_dict:
            raise Exception("This should not be possible.")

        input_idx_list = input_lookup_dict[message]
        
        if len(input_idx_list) == 0:
            raise Exception("This should not be possible")

        matching_indices.append(input_idx_list.pop())
    
    return matching_indices


def message_passing_atom_descriptors(mol_graph):
    """
    Message passing is utilized to provide a unique identification of each 
    node in the graph and where it is with respect to the rest of the molecule.
    This is accomplished by iteratively collecting messages from atoms of 
    increasing bond distance until all possible messages have been received. 
    The messages that are passed by each atom are the sorted atom neighborhood 
    strings. Messages are passed by and from every atom in the molecule, 
    thereby collecting at each atom a distinctly unique set of messages that 
    details is exact chemical environment within the molecule. Sorting of the 
    atoms is then the task of providing a unique ordering for the messages of 
    a given molecule. One simple way is to hash all of message descriptors that
    are collected by each atom into a integer. Then a unique ordering of the 
    atoms in the molecule is provided by ordering this integer hash. 
    
    """
    message_dict = {}
    for node_idx in mol_graph.nodes:
        temp_neigh = mol_graph.nodes[node_idx]["neighborhood"]
        message_dict[node_idx] = {
            "messages": {0: {"message_list": [temp_neigh], "from_list": [node_idx]}},
            "used_idx": {node_idx: True}
            }
    
    message_order = 1
    while True:
        test_converged = _get_message(mol_graph, message_dict, message_order)
        message_order += 1
        if not test_converged:
            break
    
    message_descriptors = {}
    for node_idx in mol_graph.nodes:
        temp_full_message = ""
        for key,value in message_dict[node_idx]["messages"].items():
            sorted_message = np.sort(value["message_list"])
            temp_full_message += "_".join(sorted_message)
            temp_full_message += " | "
        message_descriptors[node_idx] = temp_full_message
        
    return message_descriptors


def _get_message(mol_graph, message_dict, message_order):
    ### Store whether or not a meaningful message has been passed in order to 
    ### check that algorithm has converged
    message_passed = False
    
    # ### Testing writing next messages
    # message_order = 2
    
    for node_idx in mol_graph.nodes:
        message_dict[node_idx]["messages"][message_order] = {"message_list": [],
                                                             "from_list": []}
        temp_dict = message_dict[node_idx]["messages"][message_order]
        prev_nodes = message_dict[node_idx]["messages"][message_order-1]["from_list"]
        for temp_idx in prev_nodes:
            adj_list = [x for x in mol_graph.adj[temp_idx]]
            for next_node_idx in adj_list:
                ### Check that it has not already been visited
                if next_node_idx in message_dict[node_idx]["used_idx"]:
                    continue
                
                ### Otherwise, collect the message
                temp_message = mol_graph.nodes[next_node_idx]["neighborhood"]
                temp_dict["message_list"].append(temp_message)
                temp_dict["from_list"].append(next_node_idx)
                message_dict[node_idx]["used_idx"][next_node_idx] = True
                message_passed = True
        
    return message_passed


def get_symmetric_ordering(mol, global_min_result=False):
    """
    Get all possible atom orderings of the given molecule that provide a 
    graphically symmetric result. Using this method, RMSD calculations can 
    occur between two graphically symmetry molecules returning the global 
    minimum result in an efficient way, performing the smallest possible 
    number of calculations. 
    
    The only assumption made here is that the graph should start from the atom
    type that has the smallest number of graphically unique locations in the 
    molecule. If there is a tie, then the graphically unique location with the 
    smallest hash is used. This does not lead to the literal smallest number 
    of calculations. This is because all starting locations would have to be 
    searched over in order to provide the smallest possible of symmetric orderings. 
    However, it's likely that this would increase the computational cost beyond. 
    Therefore, this is the purpose of the argument global_min_result. 

    If global_min_result is True, then all possible starting hashes are searched
      over to provide the literal minimum possible number of symmetric orderings. 
    If global_min_result is False, then a reasonable assumption is used, as 
      explained above, to provide a very good estimate for the minimum possible 
      number of symmetry orderings at a much smaller computational cost.
    Certain algorithms will benefit from an efficient estimate, while certain
      algorithms will benefit from the global minimum at increased initial 
      computational cost. Therefore, this is left to the user/developer to decide. 
      
    Timing Info:
        global_min_result=False
            TATB    :   96     ms,   384    paths,    tatb
            QQQBRD02:   179    ms,   4608   paths,    hexanitroethane
            1142965 :   2.1    ms,   16     paths,    p-nitrobenzene
            1211485 :   4.6    ms,   144    paths,    N,N-dimethylnitramine
            1142948 :   3.37   ms,   8      paths,    m-Dinitrobenzene
            ZZZMUC  :   21.5   ms,   96     paths,    TNT
            ZZZFYW  :   14.3   ms,   8      paths,    1,2-dinitrobenzene 
            CUBANE  :   627    ms,   240    paths,    cubane
            220699  :   4.31   ms,   288    paths,    N,N-dimethyl-4-nitroaniline 
            DATB    :   11.9   ms,   64     paths,    datb
            HEVRUV  :   9.31   ms,   48     paths,    trinitromethane
            PERYTN12:   485    ms,   6144   paths,    pentaerythritol tetranitrate 
            ZZZQSC  :   19.2   ms,   48     paths,    2,6-Dinitrotoluene 
            CTMTNA14:   13.2   ms,   384    paths,    rdx
            MNTDMA  :   4172   ms,   144    paths,    m-Nitro-N,N-dimethylaniline 
            DNOPHL01:   11.8   ms,   4      paths,    2,4-Dinitrophenol 
            TNPHNT  :   27     ms,   192    paths,    2,4,6-Trinitrophenetole 
            HIHHAH  :   15.6   ms,   122    paths,    1-(3-Nitrophenyl)ethanone 
            SEDTUQ10:   5.42   ms,   64     paths,    fox-7
            SEDTUQ01:   8.65   ms,   64     paths,    fox-7
            PUBMUU  :   782    ms,   512    paths,    CL-20
            PICRAC12:   14.2   ms,   16     paths,    Picric acid 
            CAXNIY  :   6170   ms,   2048   paths,    2,2-Dinitroadamantane 
            HOJCOB  :   702    ms,   2048   paths,    HMX
            OCHTET03:   650    ms,   2048   paths,    HMX
            MNPHOL02:   4.65   ms,   2      paths,    m-Nitrophenol 
            TNIOAN  :   9.77   ms,   32     paths,    2,4,6-Trinitroaniline
            EJIQEU01:   2.76   ms,   2      paths,    5-Amino-1H-tetrazole 
            TNBENZ13:   2.39   ms,   48     paths,    1,3,5-trinitrobenzene 
            
        global_min_result=True
            TATB    :   96     ms,   384    paths,    tatb
            QQQBRD02:   1000   ms,   4608   paths,    hexanitroethane
            1142965 :   77.3   ms,   16      paths,    p-nitrobenzene
            1211485 :   87.6   ms,   144    paths,    N,N-dimethylnitramine
            1142948 :   92.9   ms,   8      paths,    m-Dinitrobenzene
            ZZZMUC  :   513    ms,   96     paths,    TNT
            ZZZFYW  :   140    ms,   8      paths,    1,2-dinitrobenzene 
            CUBANE  :   2000   ms,   240    paths,    cubane
            220699  :   2150   ms,   288    paths,    N,N-dimethyl-4-nitroaniline 
            DATB    :   358    ms,   64     paths,    datb
            HEVRUV  :   41.5   ms,   48     paths,    trinitromethane
            PERYTN12:   5730   ms,   6144   paths,    pentaerythritol tetranitrate 
            ZZZQSC  :   234    ms,   48     paths,    2,6-Dinitrotoluene 
            CTMTNA14:   832    ms,   384    paths,    rdx
            MNTDMA  :   1690   ms,   144    paths,    m-Nitro-N,N-dimethylaniline 
            DNOPHL01:   93.4   ms,   4      paths,    2,4-Dinitrophenol 
            TNPHNT  :   927    ms,   192    paths,    2,4,6-Trinitrophenetole 
            HIHHAH  :   123    ms,   12     paths,    1-(3-Nitrophenyl)ethanone 
            SEDTUQ10:   68.9   ms,   64     paths,    fox-7
            SEDTUQ01:   49.5   ms,   64     paths,    fox-7
            PUBMUU  :   25060  ms,   512    paths,    CL-20
            PICRAC12:   125    ms,   16     paths,    Picric acid 
            CAXNIY  :     ms,      paths,    2,2-Dinitroadamantane 
            HOJCOB  :     ms,   2048   paths,    HMX
            OCHTET03:     ms,   2048   paths,    HMX
            MNPHOL02:     ms,   2      paths,    m-Nitrophenol 
            TNIOAN  :     ms,   32     paths,    2,4,6-Trinitroaniline
            EJIQEU01:     ms,   2      paths,    5-Amino-1H-tetrazole 
            TNBENZ13:     ms,   48     paths,    1,3,5-trinitrobenzene 
        
    As one can see above, for all the molecules in the dataset, the global 
    minimum number of paths is found using the simple assumptions described 
    above. Although, it may be possible to conceive of a molecule that would 
    break this observation. However, the default option will be set to 
    False for global minimum searching because it is unlikely that the global
    minimum will not be found using the simple implemented assumptions.
    
    PAHs Dataset
    --------------------------
    0/89: BZPHAN
    NUM ATOMS: 30
    UNIQUE PATHS: 2
    24.3 ms ± 3.84 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    1/89: CORANN12
    NUM ATOMS: 30
    UNIQUE PATHS: 20
    489 ms ± 72.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    2/89: DBNTHR02
    NUM ATOMS: 36
    UNIQUE PATHS: 2
    65 ms ± 879 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    3/89: CENYAV
    NUM ATOMS: 66
    UNIQUE PATHS: 8
    156 ms ± 866 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    4/89: DBPHEN02
    NUM ATOMS: 36
    UNIQUE PATHS: 4
    69.9 ms ± 1.59 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    5/89: ZZZNTQ01
    NUM ATOMS: 62
    UNIQUE PATHS: 128
    88.7 ms ± 1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    6/89: KUBWAF01
    NUM ATOMS: 46
    UNIQUE PATHS: 8
    107 ms ± 897 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    7/89: PERLEN07
    NUM ATOMS: 32
    UNIQUE PATHS: 8
    93.7 ms ± 1.17 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    8/89: ABECAL
    NUM ATOMS: 46
    UNIQUE PATHS: 4
    110 ms ± 2.44 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    9/89: BIFUOR
    NUM ATOMS: 44
    UNIQUE PATHS: 8
    46.6 ms ± 460 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    10/89: CENXUO
    NUM ATOMS: 66
    UNIQUE PATHS: 8
    174 ms ± 1.41 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    11/89: HPHBNZ03
    NUM ATOMS: 72
    UNIQUE PATHS: 768
    253 ms ± 3.34 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    12/89: PEZPUG
    NUM ATOMS: 52
    UNIQUE PATHS: 2
    83.1 ms ± 683 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    13/89: HBZCOR
    NUM ATOMS: 60
    UNIQUE PATHS: 24
    1min 23s ± 15.5 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    14/89: KUBVUY
    NUM ATOMS: 66
    UNIQUE PATHS: 32
    897 ms ± 138 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    15/89: ANTCEN
    NUM ATOMS: 24
    UNIQUE PATHS: 4
    65.2 ms ± 10.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    16/89: BIPHNE01
    NUM ATOMS: 20
    UNIQUE PATHS: 8
    77.7 ms ± 2.44 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    17/89: TRIPHE12
    NUM ATOMS: 30
    UNIQUE PATHS: 12
    221 ms ± 2.85 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    18/89: SANQII
    NUM ATOMS: 36
    UNIQUE PATHS: 2
    174 ms ± 18.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    19/89: VEHCAM
    NUM ATOMS: 96
    UNIQUE PATHS: 128
    822 ms ± 20.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    20/89: HPHBNZ02
    NUM ATOMS: 72
    UNIQUE PATHS: 768
    371 ms ± 6.65 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    21/89: FLUANT02
    NUM ATOMS: 26
    UNIQUE PATHS: 2
    34.1 ms ± 1.57 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    22/89: RAKGOA
    NUM ATOMS: 42
    UNIQUE PATHS: 2
    439 ms ± 4.44 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    23/89: KAGGEG
    NUM ATOMS: 38
    UNIQUE PATHS: 4
    310 ms ± 6.84 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    24/89: QQQCIG04
    NUM ATOMS: 70
    UNIQUE PATHS: 64
    338 ms ± 5.86 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    25/89: WOQPAT
    NUM ATOMS: 54
    UNIQUE PATHS: 2
    1.75 s ± 56 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    26/89: TERPHE02
    NUM ATOMS: 32
    UNIQUE PATHS: 16
    23.6 ms ± 415 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    27/89: XECJIZ
    NUM ATOMS: 50
    UNIQUE PATHS: 8
    151 ms ± 1.74 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    28/89: NAPHTA04
    NUM ATOMS: 18
    UNIQUE PATHS: 4
    13.5 ms ± 287 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    --------------------------
    29/89: BEANTR
    NUM ATOMS: 30
    UNIQUE PATHS: 1
    28.8 ms ± 898 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    30/89: VUFHUA
    NUM ATOMS: 66
    UNIQUE PATHS: 2
    3.81 s ± 68.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    31/89: VEBJIW
    NUM ATOMS: 56
    UNIQUE PATHS: 16
    157 ms ± 2.63 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    32/89: REBWIE
    NUM ATOMS: 48
    UNIQUE PATHS: 4
    197 ms ± 1.86 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    33/89: QUATER10
    NUM ATOMS: 60
    UNIQUE PATHS: 4
    6.39 s ± 173 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    34/89: VEBKAP
    NUM ATOMS: 46
    UNIQUE PATHS: 4
    106 ms ± 1.27 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    35/89: DPANTR01
    NUM ATOMS: 44
    UNIQUE PATHS: 16
    102 ms ± 2.22 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    36/89: DUPHAX
    NUM ATOMS: 62
    UNIQUE PATHS: 1
    5.6 s ± 95.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    37/89: DNAPAN
    NUM ATOMS: 48
    UNIQUE PATHS: 2
    347 ms ± 3.75 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    38/89: TBZPYR
    NUM ATOMS: 44
    UNIQUE PATHS: 2
    498 ms ± 2.98 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    39/89: ZZZOYC01
    NUM ATOMS: 36
    UNIQUE PATHS: 2
    88.6 ms ± 2.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    40/89: FOVVOB
    NUM ATOMS: 52
    UNIQUE PATHS: 32
    50.1 ms ± 414 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    41/89: PUTVUV
    NUM ATOMS: 102
    UNIQUE PATHS: 512
    177 ms ± 1.6 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    42/89: IZUCIP
    NUM ATOMS: 52
    UNIQUE PATHS: 8
    307 ms ± 1.53 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    43/89: TETCEN01
    NUM ATOMS: 30
    UNIQUE PATHS: 4
    48.2 ms ± 1.63 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    44/89: PYRCEN
    NUM ATOMS: 26
    UNIQUE PATHS: 64
    216 ms ± 5.32 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    45/89: PUNVEA
    NUM ATOMS: 132
    UNIQUE PATHS: 16384
    3.08 s ± 136 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    46/89: BNPERY
    NUM ATOMS: 34
    UNIQUE PATHS: 2
    293 ms ± 4.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    47/89: TERPHO02
    NUM ATOMS: 32
    UNIQUE PATHS: 1
    7.01 ms ± 180 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    --------------------------
    48/89: KIDJUD03
    NUM ATOMS: 52
    UNIQUE PATHS: 64
    66.9 ms ± 792 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    49/89: PENCEN
    NUM ATOMS: 36
    UNIQUE PATHS: 4
    108 ms ± 2.57 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    50/89: QUPHEN01
    NUM ATOMS: 42
    UNIQUE PATHS: 32
    89.1 ms ± 2.28 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    51/89: PHENAN
    NUM ATOMS: 24
    UNIQUE PATHS: 2
    30.8 ms ± 536 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    52/89: TBZPER
    NUM ATOMS: 52
    UNIQUE PATHS: 2
    1.64 s ± 16.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    53/89: KAFVUI
    NUM ATOMS: 38
    UNIQUE PATHS: 1
    124 ms ± 1.77 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    54/89: PEZPIU
    NUM ATOMS: 52
    UNIQUE PATHS: 1
    95.2 ms ± 1.32 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    55/89: KAGFUV
    NUM ATOMS: 56
    UNIQUE PATHS: 4
    876 ms ± 9.36 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    56/89: BOXGAW
    NUM ATOMS: 56
    UNIQUE PATHS: 1
    4.21 s ± 67 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    57/89: PUNVIE
    NUM ATOMS: 62
    UNIQUE PATHS: 64
    58.9 ms ± 582 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    58/89: FANNUL
    NUM ATOMS: 28
    UNIQUE PATHS: 28
    88.6 ms ± 2.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    59/89: VEBJAO
    NUM ATOMS: 136
    UNIQUE PATHS: 4096
    1min 2s ± 11.9 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    60/89: TEBNAP
    NUM ATOMS: 42
    UNIQUE PATHS: 4
    229 ms ± 15.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    61/89: LIMCUF
    NUM ATOMS: 72
    UNIQUE PATHS: 256
    649 ms ± 6.43 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    62/89: CORXAI10
    NUM ATOMS: 52
    UNIQUE PATHS: 1
    1.44 s ± 23.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    63/89: CEQGEL
    NUM ATOMS: 32
    UNIQUE PATHS: 2
    186 ms ± 3.82 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    64/89: CORONE01
    NUM ATOMS: 36
    UNIQUE PATHS: 96
    1.79 s ± 122 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    65/89: SURTAA
    NUM ATOMS: 36
    UNIQUE PATHS: 1
    241 ms ± 1.94 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    66/89: ZZZNKU01
    NUM ATOMS: 52
    UNIQUE PATHS: 64
    66.3 ms ± 1.96 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    67/89: JULSOY
    NUM ATOMS: 64
    UNIQUE PATHS: 2
    5.2 s ± 23 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    68/89: KAGFOP
    NUM ATOMS: 58
    UNIQUE PATHS: 8
    656 ms ± 8.55 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    69/89: VEBJES
    NUM ATOMS: 76
    UNIQUE PATHS: 64
    369 ms ± 2.69 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    70/89: BIGJUX
    NUM ATOMS: 120
    UNIQUE PATHS: 256
    2 s ± 8.52 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    71/89: NAPANT01
    NUM ATOMS: 52
    UNIQUE PATHS: 4
    2.06 s ± 335 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    72/89: YITJAN
    NUM ATOMS: 40
    UNIQUE PATHS: 2
    62.9 ms ± 5.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    73/89: PUJQIV
    NUM ATOMS: 34
    UNIQUE PATHS: 2
    32.9 ms ± 1.53 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    74/89: YOFCUR
    NUM ATOMS: 68
    UNIQUE PATHS: 8
    1min 46s ± 8.95 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    75/89: FACPEE
    NUM ATOMS: 68
    UNIQUE PATHS: 16
    1.1 s ± 149 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    76/89: TPHBEN01
    NUM ATOMS: 42
    UNIQUE PATHS: 48
    70.7 ms ± 806 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    77/89: CRYSEN01
    NUM ATOMS: 30
    UNIQUE PATHS: 2
    58.2 ms ± 361 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    78/89: POBPIG
    NUM ATOMS: 48
    UNIQUE PATHS: 4
    1.4 s ± 30.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    79/89: PYRPYR10
    NUM ATOMS: 46
    UNIQUE PATHS: 4
    601 ms ± 18.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    80/89: PERLEN05
    NUM ATOMS: 32
    UNIQUE PATHS: 8
    119 ms ± 2.75 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    81/89: TBZHCE
    NUM ATOMS: 64
    UNIQUE PATHS: 4
    2.72 s ± 12.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    82/89: KEGHEJ01
    NUM ATOMS: 22
    UNIQUE PATHS: 4
    44.7 ms ± 206 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    83/89: BIPHEN
    NUM ATOMS: 22
    UNIQUE PATHS: 8
    16 ms ± 70.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    --------------------------
    84/89: PHNAPH
    NUM ATOMS: 46
    UNIQUE PATHS: 4
    663 ms ± 5.01 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    85/89: DBZCOR
    NUM ATOMS: 48
    UNIQUE PATHS: 4
    1.63 s ± 32.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    86/89: BNPYRE10
    NUM ATOMS: 32
    UNIQUE PATHS: 1
    67.2 ms ± 523 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    --------------------------
    87/89: VEBJOC
    NUM ATOMS: 96
    UNIQUE PATHS: 256
    1.26 s ± 5.21 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    --------------------------
    88/89: BENZEN
    NUM ATOMS: 12
    UNIQUE PATHS: 12
    14.6 ms ± 275 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) 
    
    Based on the PAHs dataset, it becomes clear the the scaling of the algorithm
    has very little to do with the number of atoms in the molecule or even the 
    number of unique paths. The scaling of the algorithm is primairly concerned
    with the number of connected cycles in the molecule. Such chemical groups are 
    relatively uncommon in all molecular classes besides organic electronics. 
    Particularly interesting for this class of systems is that despite that 
    they are only made up of fused rings of hydrogen and carbon, for some
    very large and complex molecules, there may only be one graphically 
    symmetric path for the molecule (BOXGAW) demonstrating the success of the 
    general approach applied here. 
    
    """
    gmol = mol2graph(mol)
    
    ### Take care of the possibility of multiple molecules
    component_idx_lists = []
    
    for sub_idx in nx.components.connected_components(gmol):
        sub_idx = [x for x in sub_idx]
        sub_g = gmol.subgraph(sub_idx)
        
        #### Reorder nodes in g according to sub_idx
        g = nx.Graph()
        g.add_nodes_from(range(len(sub_g)))
        attr_dict = {}
        lookup = {}
        rlookup = {}
        for iter_idx,temp_node_idx in enumerate(sub_g):
            temp_attr = sub_g.nodes[temp_node_idx]
            attr_dict[iter_idx] = temp_attr
            lookup[iter_idx] = temp_node_idx
            rlookup[temp_node_idx] = iter_idx
        nx.set_node_attributes(g, attr_dict)
        for temp_edge in sub_g.edges:
            g.add_edge(rlookup[temp_edge[0]], rlookup[temp_edge[1]])
        
        hash_list = [g.nodes[x]["hash"] for x in g.nodes]
        unique_hash,counts = np.unique(hash_list, return_counts=True)
        count_sort_idx = np.argsort(counts)

        all_paths_list = []
        for iter_idx,temp_idx in enumerate(count_sort_idx):
            all_paths_list.append([])
            temp_hash = unique_hash[temp_idx]
            temp_hash_idx = np.where(hash_list == temp_hash)[0]
            
            for temp_atom_idx in temp_hash_idx:
                temp_paths = ordered_descriptor_traversal(g,temp_atom_idx)
                all_paths_list[iter_idx] += temp_paths
            
            if not global_min_result:
                break
        
        ### Use the hash the provides the minimum number of unique orderings
        num_paths = np.array([len(x) for x in all_paths_list])
        min_paths_idx = np.argmin(num_paths)
        min_paths = all_paths_list[min_paths_idx]
        
        ### Need to de-reference each of the paths wrt the subgraph
        temp_final_paths = []
        for temp_path in min_paths:
            temp_deref_path = [lookup[x] for x in temp_path]
            temp_final_paths.append(temp_deref_path)
        component_idx_lists.append(temp_final_paths)
    
    ### Assume that connected components are not identical. This assumption 
    ### may be incorrect for some systems. However, that may be updated later. 
    final_paths = [list(itertools.chain.from_iterable(x)) 
                   for x in itertools.product(*component_idx_lists)]

    return final_paths
    


def ordered_descriptor_traversal(
            g, 
            node_idx, 
            current_path=None):
    """
    Uses the message descriptors to make and ordered traversal of the molecule. 
    Traversal means that the path must be connected. Ordered means that provided
    the same graph and starting position, the same traversal will always be 
    returned. 
    
    IN ADDITION, will return all possible graphically symmetric traversals. 
    
    This function will now keep track and append all symmetric traversals all
    at once. This will create a significantly easier to use API compared to 
    what was built above. 
    
    Symmetry switches are no longer returned. This is because they will be 
    handled by create NEW PATH lists for every symmetry switch. Then each 
    path that is created from a symmetry switch will always be acted upon 
    independently as it moves up the stack. THEREBY, all symmetrically 
    equivalent paths will be built and returned all at once. 
    
    Timing info:
        - TATB: 48 ms

    """
    
    if current_path == None:
        current_path = []
    
    ### Make copy of input current_path
    current_path = [x for x in current_path]
    
    ### Get neighbors and sort them by their hash value, thereby neighbors
    ### with smallest hash values are always visted first
    path = [node_idx]
    current_path += [node_idx]
    neigh = [x for x in g.adj[node_idx]]
    neigh_hash = [g.nodes[x]["hash"] for x in neigh]
    sorted_neigh_idx = np.argsort(neigh_hash)
    neigh = [neigh[x] for x in sorted_neigh_idx]

    ### Build entire traversal list
    neigh_path_list = []
    for entry in neigh:
        # print(node_idx, entry)
        if entry in current_path:
            continue
        neigh_path = ordered_descriptor_traversal(
                                g, 
                                entry, 
                                current_path)
        if len(neigh_path) > 0:
            neigh_path_list.append(neigh_path)  
            
    ### Only a single option so safe to add
    if len(neigh_path_list) == 0:
        return [[node_idx]]
    
    ### Returned is a list of lists
    symmetry_switches = combine_neigh_path_lists(
            neigh_path_list,
            g, 
            current_path)
    
    sym_paths = []
    for neigh_list_sorted in symmetry_switches:
        sym_paths.append(path+neigh_list_sorted)
        
    # print("-----------------------------------------")
    # print("NODE", node_idx)
    # print("NEIGH", neigh)
    # if node_idx == 3:
    #     print("NEIGH PATH", 
    #           neigh_path_list[0][0], 
    #           neigh_path_list[1][1],
    #           neigh_path_list[2][2])
    # print("CURRENT PATH", current_path)
    # print("NUM PATHS", len(sym_paths), len(np.unique(np.vstack(sym_paths), axis=0)))
    # # print("SYM", sym_paths)
    
    # if len(sym_paths) > 64:
    #     raise Exception()
    
    return sym_paths

def combine_neigh_path_lists(sym_neigh_path_list, 
                             g,
                             current_path):
    """
    Returned is a list of lists which describes all possible symmetry 
    switches for the given neigh_path_lists
    
    Input is a list like this [[[6]],[[7]]] that must be combined together and
    returned providing all symmetric orders as separate paths like this
    [[6,7],[7,6]] if 6 and 7 are graphically symmetric. If 6 and 7 are not 
    graphically symmetric, then the one that has a smaller hash will always 
    appear first. For example [[6,7]] if 6 has the smaller hash. 
    
    Symmetry switches, like railroad switches on a track, provide the ordering
    with a different path around the molecule. Symmetry refers to that this 
    different order provides a graphically symmetry path, thus even though 
    a switch occurs it provides a symmetric outcome. 
    
    """
    ### First build data structures 
    all_path_list = []             ### Concatenated list of input 
    neigh_ref_idx_list = []        ### Stores which list each path comes from
    lookup = {}                    ### Stores lookup for each ref
    for outer_idx,sym_path_list in enumerate(sym_neigh_path_list):
        lookup[outer_idx] = []
        for temp_path in sym_path_list:
            all_path_list.append(np.array(temp_path))
            neigh_ref_idx_list.append(outer_idx)
            lookup[outer_idx].append(len(all_path_list)-1)
    
    ### Sort the neighbor lists by the hash value of the first entry. In 
    ### addition, find which neighbors share this minimum hash value, which 
    ### are graphically symmetry atoms thus indicating a symmetry switch
    hash_ref = np.array([g.nodes[x[0][0]]["hash"] for x in sym_neigh_path_list])
    hash_ref_sort_idx = np.argsort(hash_ref)
    hash_list = np.array([g.nodes[x[0]]["hash"] for x in all_path_list])
    hash_sort_idx = np.argsort(hash_list)
    unique_hash,counts = np.unique(hash_list,return_counts=True)
    min_hash = hash_list[hash_sort_idx[0]]
    
    ### In addition, build a data-structure that will be able to identify 
    ### which ref actually add atoms to the path for each possible permutation
    meaningful_dict = {}
    for ref_list in itertools.permutations(list(range(len(sym_neigh_path_list)))):
        used_idx = {}
        temp_use_ref_list = []
        keep_idx = []
        for iter_idx,temp_ref in enumerate(ref_list):
            meaningful = False
            temp_idx_list = sym_neigh_path_list[temp_ref][0]
            for temp_idx in temp_idx_list:
                if temp_idx not in used_idx:
                    meaningful = True
                    used_idx[temp_idx] = True
            if meaningful:
                temp_use_ref_list.append(temp_ref)
                keep_idx.append(iter_idx)
        meaningful_dict[tuple(ref_list)] = {"use": temp_use_ref_list,
                                            "keep_idx": keep_idx}
    
    ### Make list by combining all unique hashes together while also ensuring
    ### all neigh_ref are included
    all_idx_list = [[]]
    for iter_idx,temp_hash in enumerate(unique_hash):
        ### Find which neighbor lists are possible for this hash
        temp_hash_idx = np.where(hash_list == temp_hash)[0]
        
        ### Find which ref this neighbor list comes from
        temp_ref_idx = np.array([neigh_ref_idx_list[x] for x in temp_hash_idx])
        temp_unique_ref = np.unique(temp_ref_idx)
        temp_ref_loc_idx = [np.where(temp_ref_idx==x)[0] for x in temp_unique_ref]
        
        ### If there are multiple ref, every ordering of the ref needs to be 
        ### considered, because one list from each ref needs to be added 
        ### to construct all possibilities
        temp_idx_product_list = list(itertools.product(*temp_ref_loc_idx))
        ### In addition, the order in which the ref is added needs to be 
        ### permuted
        temp_ref_order_list = itertools.permutations(list(range(len(temp_unique_ref))))
        
        ### Using all this information, append this hash to the final lists
        temp_all_idx_list = []
        for temp_ref_order in temp_ref_order_list:
            for temp_idx_prod in temp_idx_product_list:
                ### Reorder based on temp_ref_order
                temp_idx_prod = [temp_idx_prod[x] for x in temp_ref_order]
                ### De-reference temp_idx_prod to the current hash_idx
                temp_idx_prod = [temp_hash_idx[x] for x in temp_idx_prod]
                                
                ### Append to each existing entry in all_idx_list
                for entry in all_idx_list:
                    temp_all_idx_list.append(entry+temp_idx_prod)
                    
        ### Set all_idx_list for the next hash loop
        all_idx_list = temp_all_idx_list
    
    ### Now, using the meaningful dict information, reduce the total number of
    ### entries in all_idx_list that have to be used
    use_idx_dict = {}
    neigh_ref_idx_list = np.array(neigh_ref_idx_list)
    for entry in all_idx_list:
        temp_ref_idx = neigh_ref_idx_list[entry]
        keep_idx = meaningful_dict[tuple(temp_ref_idx)]["keep_idx"]
        temp_use = [entry[x] for x in keep_idx]
        use_idx_dict[tuple(temp_use)] = True
    all_idx_list = [list(x) for x in list(use_idx_dict.keys())]
        
    ### Build data structure to make path building much more efficient
    all_mask_list = []
    for temp_path in all_path_list:
        temp_mask = np.zeros(shape=(len(g),)).astype(int)
        for iter_idx,temp_idx in enumerate(temp_path):
            ### Because iter_idx stored here, this stores the information 
            ### of where the temp_idx actually appears in the given path
            ### referenced later by temp_add_order_list
            temp_mask[temp_idx] = iter_idx+1
            ### +1 used because don't want to ever store 0
        all_mask_list.append(temp_mask)
        
    ### Build paths and store
    symmetry_switched_paths = {}
    for iter_idx,temp_idx_list in enumerate(all_idx_list):
        path_mask = np.zeros(shape=(len(g),)).astype(int)
        path = [x for x in all_path_list[temp_idx_list[0]]]
        path_mask[path] = 1
        for temp_idx in temp_idx_list[1:]:
            ### Get data
            temp_path = all_path_list[temp_idx]
            temp_path_mask = all_mask_list[temp_idx]
            
            ### This gets only the mask for the entries in the temp path
            ### that have not yet been added
            temp_add_mask = np.logical_and(
                            np.logical_not(path_mask), temp_path_mask)
            ### Add these to path_mask
            path_mask[temp_add_mask] = 1
            
            ### This is numeric value of atom_idx that needs to be added
            temp_add_idx_list = np.where(temp_add_mask)[0]
            
            ### This is the relative order with which these atom_idx appear in path
            temp_add_order_list = np.argsort(temp_path_mask[temp_add_mask])
            
            for temp_add_order_idx in temp_add_order_list:
                path.append(temp_add_idx_list[temp_add_order_idx])
                
        symmetry_switched_paths[tuple(path)] = path
    
    ###### SLOWER older method that can be used for validating results
    # ### Build paths and store
    # symmetry_switched_paths = {}
    # for iter_idx,temp_idx_list in enumerate(all_idx_list):
    #     temp_combined_path = []
        
    #     temp_currently_used = {}
    #     for x in current_path:
    #         temp_currently_used[x] = True
            
    #     for temp_idx in temp_idx_list:
    #         temp_path = all_path_list[temp_idx]
    #         for temp_atom_idx in temp_path:
    #             if temp_atom_idx in temp_currently_used:
    #                 continue
    #             else:
    #                 temp_currently_used[temp_atom_idx] = True
    #                 temp_combined_path.append(temp_atom_idx)
        
    #     symmetry_switched_paths[tuple(temp_combined_path)] = temp_combined_path
    
    ### By nature of the algorithm, some symmetry_switched_paths that are 
    ### built in the above loop will be duplicates. The paths that are built
    ### are all unique, until one considers that an atom cannot be visited twice. 
    ### Thereby, two paths that are unique become duplicates when what makes 
    ### the path unique is removed because it's already been visited. 
    symmetry_switched_paths = [x for x in symmetry_switched_paths.values()]
    
    ### Add all to current_path
    for atom_idx in symmetry_switched_paths[0]:
        if atom_idx not in current_path:
            current_path.append(atom_idx)
    
    # print(len(sym_neigh_path_list),
    #       [len(x) for x in sym_neigh_path_list],
    #       len(all_idx_list), 
    #       len(symmetry_switched_paths),
    #       counts)
    
    return symmetry_switched_paths





    