
import os,copy,itertools

import numpy as np

from ase.data import chemical_symbols,vdw_radii,atomic_numbers,covalent_radii

from mcse.molecules.label import mol2graph



def get_dimer_cutoff_matrix(dimer, vdw=[], bonds_kw={}, hbond=True):
    mol_idx = dimer.get_molecule_idx(**bonds_kw)
    mol1 = dimer.get_sub(mol_idx[0], struct_id="mol1")
    mol2 = dimer.get_sub(mol_idx[1], struct_id="mol2")
    return get_cutoff_matrix(mol1,mol2,vdw,bonds_kw,hbond)


def get_cutoff_matrix(mol1, 
                      mol2, 
                      vdw=[], 
                      bonds_kw={},
                      hbond=True):
    """
    Return the vdW distance matrix for mol1 and mol2 considering the 
    possible intermolecular hydrogen bonds that may form between the two.
    
    """
    ele1 = mol1.elements
    if mol2 == None:
        mol2 = mol1
    ele2 = mol2.elements
    
    ### First build basic cutoff matrix by vdW
    if len(vdw) == 0:
        vdw = vdw_radii
    
    a1 = np.array([vdw[atomic_numbers[x]] for x in ele1])
    a2 = np.array([vdw[atomic_numbers[x]] for x in ele2])
    cutoff_matrix = a1[:,None] + a2[None,:]
    
    if not hbond:
        return cutoff_matrix

    ### Now, need to augment by possible h-bond donors and acceptors
    g1 = mol2graph(mol1, bonds_kw=bonds_kw)
    g2 = mol2graph(mol2, bonds_kw=bonds_kw)
    
    ### Get donors and acceptors
    d1 = donors(g1)
    a1 = acceptors(g1)
    d2 = donors(g2)
    a2 = acceptors(g2)

    ### Populate cutoff_matrix at donor,acceptor indices using consistent definition
    ### for radii relevant for h-bonds. Consistent in the sense that a sr value of 
    ### 0.85 gives the values listed in the Genarris 2.0 publication for h-bonding
    ### which are as follows:
    h_bond = {
            "OH-O": 1.5,
            "OH-N": 1.6,
            "NH-O": 1.6,
            "NH-N": 1.75}

    for i,j in itertools.product(d1,a2):
        ele_a = ele2[j]
        temp_donor = [x for x in g1.neighbors(i)]
        if len(temp_donor) != 1:
            raise Exception("Hydrogen detected with two bonds. That's impossible.")
        ele_d = ele1[temp_donor[0]] 
        dist = h_bond["{}H-{}".format(ele_d, ele_a)] / 0.85
        cutoff_matrix[i,j] = dist
        
    for i,j in itertools.product(d2,a1):
        ele_a = ele1[j]
        temp_donor = [x for x in g2.neighbors(i)]
        if len(temp_donor) != 1:
            raise Exception("Hydrogen detected with two bonds. That's impossible.")
        ele_d = ele2[temp_donor[0]] 
        dist = h_bond["{}H-{}".format(ele_d, ele_a)] / 0.85
        cutoff_matrix[j,i] = dist
    
    return cutoff_matrix
    

def donors(graph):
    """ Returns donor idx from molecule graph """
    idx = []
    for node_idx in graph:
        temp_node = graph.nodes[node_idx]
        temp_neigh = temp_node["neighborhood"]
        temp_ele = temp_node["ele"]
        if temp_ele != "H":
            continue
        if temp_neigh in ["HN", "HO"]:
            idx.append(node_idx)
    return idx
            

def acceptors(graph):
    """ Returns acceptor idx from molecule graph """
    idx = []
    for node_idx in graph:
        temp_node = graph.nodes[node_idx]
        temp_neigh = temp_node["neighborhood"]
        temp_ele = temp_node["ele"]
        if temp_ele not in ["N", "O"]:
            continue
        if temp_neigh[0] in ["N","O"]:
            idx.append(node_idx)
        elif temp_neigh[:2] == "HO":
            idx.append(node_idx)
    return idx