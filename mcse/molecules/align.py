
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

from ase.data import atomic_numbers,atomic_masses_iupac2016
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as PGA

from mcse import Structure   
from mcse.core.utils import com,center_com
from mcse.molecules.symmetry import get_symmetry 


def align(struct, sym_kw={"max_rot": 4, "tol": 0.3}, recursion_count=0):
    """
    Aligns the molecule such that the COM is at the origin and the axes defined 
    by the moment of inertia tensor are oriented with the origin. In addition, 
    the operation is safe with respect to the symmetry of the molecule and the
    symmetry of the moment of inertia tensor with respect to the geometry of
    the molecule. 
    
    """
    ### First move COM of molecule to the origin
    trans = com(struct)
    geo = struct.get_geo_array()
    ele = struct.elements
    geo = geo - trans
    struct.from_geo_array(geo, ele)
    
    principal_axes = get_principal_axes(struct)
    principal_axes,symmetric = _correct_principal_axes_symmetry(
                                    struct, principal_axes, sym_kw)
    rot = principal_axes
    
    ### If principal axes don't pass any symmetric checks, that it can be 
    ### known that the rotation must be applied to the geometry to properly 
    ### align it with the cartesian coordinate system. 
    if symmetric == False:
        principal_axes = get_principal_axes(struct)
        rot = principal_axes
        geo =  np.dot(rot, geo.T).T
        struct.from_geo_array(geo, ele)
    
        ### Check if the newly aligned structure has actually aligned the 
        ### molecule with the origin. 
        ### Multiple alignments may need to be made if the molecule is highly 
        ### symmetric to converge the alignment process. 
        ### Recursion is an easy way to do this. 
        
        ### First check if more alignment is necessary
        new_principal_axes = get_principal_axes(struct)
        new_principal_axes,new_symmetric = _correct_principal_axes_symmetry(
                                    struct, new_principal_axes, sym_kw)
        
        ### Aligned structure did not lead to symmetric alignment with the
        ### origin
        if new_symmetric == False:
            #### Break recursion if tried too many times
            if recursion_count == 10:
                raise Exception("Alignment Failed. Is {} highly symmetric?"
                                .format(struct.struct_id))
            else:
                # print("Recurring: {}".format(recursion_count+1))
                align(struct, sym_kw, recursion_count+1)
    
    return struct


def align_molecules(mol1, mol2, sym_kw={"max_rot": 6, "tol": 0.1}):
    """
    Aligns two molecules to minimize rmsd. Note that the molecules need have 
    their center of masses at the origin for the algorithm to properly align
    their orientation in space. 
    
    Arguments
    ---------
    mol1: Structure
        Molecule that will remain stationary for alignment
    mol2: Structure
        Molecule that will
    
    """
    rot = align_molecules_rot(mol1, mol2, sym_kw)
    
    ### Move COM to origin before performing rotation
    com1 = com(mol1)
    com2 = com(mol2)
    mol1.translate(-com1)
    mol2.translate(-com2)
    
    ### Rotate mol2 as defined by align_molecules_rot algorithm
    mol2 = rot_mol(rot,mol2)
    
    ### Move back to original COM position
    mol1.translate(com1)
    mol2.translate(com2)
    
    return mol1,mol2


def align_molecules_rot(mol1, mol2, sym_kw={"max_rot": 6, "tol": 0.1}, 
                        recursion_count=0, 
                        current_rot=[np.array([[1,0,0],[0,1,0],[0,0,1]])]):
    """
    Returns the rotation matrix that can be used to align the principal axes 
    of two molecules taking into account the symmetry of the given molecules. 
    For example, if the molecule has two-fold rotational symmetry and the 
    alignment computed is the same 180 degree rotation, then the identity 
    matrix is returned. 
    
    Arguments
    ---------
    mol1: Structure
        Molecule that will remain stationary for alignment
    mol2: Structure
        This is the molecule that should be rotated by the returned rotation
        matrix.
    sym_kw: dict
        Dictionary of arguments used for identifying the molecular symmetry 
        of each molecule. 
    
    """
    ### Get COM so that they can be reset in the end
    com1 = com(mol1)
    com2 = com(mol2)
          
    mol1.translate(-com1)
    mol2.translate(-com2)
    
    ### First get principal axes of the molecules
    pa1 = get_principal_axes(mol1)
    pa2 = get_principal_axes(mol2)
     
    ### Find rotation that aligns these principal components
    Rot,temp_rmsd = R.align_vectors(pa1, pa2)
    rot = Rot.as_matrix()
    if temp_rmsd > 0.1:
        raise Exception("Alignment not found for {} and {}"
                        .format(mol1.struct_id, mol2.struct_id))
    
    ### Test solution
    ### Perform rotation wrt principal axes of the molecule
    geo2 = mol2.get_geo_array()
    ele2 = mol2.elements
    temp_mol2 = Structure.from_geo(geo2, ele2)
    rot_mol(rot, temp_mol2)
    pa2 = get_principal_axes(temp_mol2)
    
    if np.linalg.norm(pa1 - pa2) > 1e-3:
        recursion_count += 1
        if recursion_count > 10:
            raise Exception("Two molecule alignment Failed. Is {} highly symmetric?"
                            .format(mol1.struct_id))
        else:
            ### Might take multiple tries if molecule is high symmetric
            rot = np.dot(rot, align_molecules_rot(mol1, temp_mol2))
    
    ### TEST COMPUTE RMSD
    # from mcse.molecules.rmsd import calc_rmsd_ele,rmsd
    # geo2 = mol2.get_geo_array()
    # ele2 = mol2.elements
    # temp_mol2 = Structure.from_geo(geo2, ele2)
    # rot_mol(rot, temp_mol2)
    # _,_,_,temp_rmsd = rmsd(mol1, temp_mol2)
    # if temp_rmsd > 0.1:
    #     raise Exception(temp_rmsd)
            
    mol1.translate(com1)
    mol2.translate(com2)
    
    return rot


def fast_align(struct, center=True):
    """
    Performs fast alignment by just rotating the system wrt the principal 
    axes of the input Structure. This will not take into account potential 
    inconsistencies due to molecular symmetry for calculation of the principal 
    axis. Also, will center the molecule by default. 

    """
    if center:
        struct.translate(-com(struct))
    pa = get_principal_axes(struct)
    ### Rotation that leads to alignment of geometry is just the principal axes 
    ### themselves, not their inverse (as far as I know...)
    rot_mol(pa, struct)
    return struct


def get_principal_axes(struct):
    """
    Obtain principal axes of inertia for the given structure. 
    
    In addition, an algorithm has been implemented that
    corrects for small numerical differences in the SVD algorithm  
    due to symmetry of the inertial tensor, and not necessarily due to 
    symmetry of the molecule, This corrects a huge problem wherein the 
    principal axis for different initial positions of the same molecule will
    result in different internal axis. Correction to these numerical 
    differences comes in the form of ensuring that the projection of the 
    maximum mass is a positive value. This ensures that the chosen axes are 
    independent of the initial position.
    
    Small differences in the atomic positions, particularly of heavy 
    atoms, for the same molecule may still lead to issues for this algorithm.
    However, encourtering these cases should be unlikely...
    
    """
    inertial_tensor = get_inertia_tensor(struct)
    ### SVD Decomposition to get axes of principal axes
    eig_vals, vecs = np.linalg.eig(inertial_tensor)
    sort_idx = np.argsort(eig_vals)[::-1]
    ### Should be transpose of u because eig vectors are stored in column major
    principal_axes = vecs.T
    principal_axes = principal_axes[sort_idx]
    
    ### Ensure that the projection of mass onto the vector is always positive
    ### thereby standardizing principal axes even for highly symmetry molecules
    geo = struct.get_geo_array()
    ele = struct.elements
    mass = np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele])
    proj = []
    for entry in principal_axes:
        ### Find maximum value for each axes
        ### In this way, the direction of the principal axes should always be
        ### in the direction with the greatest combined mass and distance in the molecule
        ### Therefore, making this method independent of numerical inconsistencies
        ### But not conformation differences...
        temp_proj = np.dot(entry, geo.T)*mass
        max_idx = np.argmax(np.abs(temp_proj))
        proj.append(temp_proj[max_idx])
    neg_idx = np.where(np.array(proj) < 0)[0]
    
    if len(neg_idx) > 0:
        diag = np.zeros((3,)) + 1
        diag[neg_idx] = -1
        op = np.zeros((3,3))
        np.fill_diagonal(op, diag)
        principal_axes = np.dot(op, principal_axes)
     
    ### Finally, check that the principal_axes obey the right-hand rule
    if np.linalg.det(principal_axes) < -0.99:
        principal_axes = np.dot(np.array([[1.,0,0], [0,1,0], [0,0,-1]]), 
                                principal_axes)
    
    return principal_axes


def match_principal_axes(struct1,struct2,pre_align=False):
    """
    Matches the principal axes of the structures so as to remove the influence 
    of small numerical differences between identical molecules that cannot 
    otherwise be accounted for. Will modify the principal axes of struct2 so 
    as to match will the principal axes of struct1. 
    
    """
    if pre_align:
        fast_align(struct1)
        fast_align(struct2)
    else:
        if np.linalg.norm(com(struct1)) > 1e-3:
            raise Exception("Structures must be pre-centered and aligned")
        if np.linalg.norm(com(struct2)) > 1e-3:
            raise Exception("Structures must be pre-centered and aligned")
    
    if struct1.formula != struct2.formula:
        raise Exception("That's not what this is for")
    
    pa1 = get_principal_axes(struct1)
    pa2 = get_principal_axes(struct2)
    
    geo1 = struct1.get_geo_array()
    ele1 = struct1.elements
    mass1 = np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele1])
    
    geo2 = struct2.get_geo_array()
    ele2 = struct2.elements
    mass2 = np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele2])
    
    ### Find the mass projection onto the PA for each current molecule
    proj1 = np.zeros((3,))
    for idx,entry in enumerate(pa1):
        temp_proj = np.dot(entry, geo1.T)*mass1
        max_idx = np.argmax(np.abs(temp_proj))
        proj1[idx] = temp_proj[max_idx]
    
    proj2 = np.zeros((3,))
    for idx,entry in enumerate(pa2):
        temp_proj = np.dot(entry, geo2.T)*mass2
        max_idx = np.argmax(np.abs(temp_proj))
        proj2[idx] = temp_proj[max_idx]
        
    
    if np.linalg.norm(proj1 - proj2) > 0.1:
        neg_options = np.array([[1,1,1],
                       [-1,1,1],[1,-1,1],[1,1,-1],
                       [-1,1,-1],[-1,-1,1],[1,-1,-1],
                       [-1,-1,-1]
                       ])
        proj2 = np.zeros((8,3))
        for row_idx,neg_idx in enumerate(neg_options):
            temp_pa = pa2 * neg_idx[:,None]
            for idx,entry in enumerate(temp_pa):
                temp_proj = np.dot(entry, geo2.T)*mass2
                max_idx = np.argmax(np.abs(temp_proj))
                proj2[row_idx][idx] = temp_proj[max_idx]
        
        final_idx = np.argmin(np.linalg.norm(proj2 - proj1, axis=-1))
        pa2 = pa2 * neg_options[final_idx][:,None]
        
    return pa1,pa2
    
    
    
def _correct_principal_axes_symmetry(struct, principal_axes, sym_kw):
    geo = struct.get_geo_array()
    ele = struct.elements
    
    ### Get symmetry of the molecule
    sym_ops = get_symmetry(struct, **sym_kw)
    
    ### Correct operation to apply to align axes with the origin is the inverse
    ### of the principals axes, which is just equal to its transpose. 
    rot = principal_axes.T
    
    ### Before applying rotation, check that this is not just a symmetry 
    ### operation of the molecule or a symmetry operation of the inertia
    ### tensor construction
    symmetric = False
    ### First check it's not identity already
    identity_check = np.linalg.norm(principal_axes - 
                                    np.array([[1.,0,0], [0,1,0], [0,0,1]]))
    if identity_check < 1e-4:
        symmetric = True
    
    ### Check principal axes against all molecular symmetry
    if symmetric == False:
        for entry in sym_ops:
            diff = np.linalg.norm(principal_axes - entry)
            if diff < 1e-4:
                principal_axes = np.array([[1.,0,0], [0,1,0], [0,0,1]])
                symmetric = True
                break
    
    if symmetric == False:
        #### Check for symmetry of principal axis under the operation of the 
        #### principal axes rotation. 
        rot_geo =  np.dot(rot, geo.T).T
        temp_struct = Structure.from_geo(rot_geo, ele)
        rot_principal_axes = get_principal_axes(temp_struct)
        
        diff = np.linalg.norm(principal_axes - rot_principal_axes)
        if diff < 1e-4:
            principal_axes = np.array([[1.,0,0], [0,1,0], [0,0,1]])
            symmetric = True
            # print("@@@@@@@@@@@ Priniciple Axis Symmetry @@@@@@@@@@@")
    
    #### Check for symmetry of principal axis with respect to the 
    #### molecular inversion symmetry
    if symmetric == False:
        identity_check = np.linalg.norm(np.abs(principal_axes) - 
                                        np.array([[1.,0,0], [0,1,0], [0,0,1]]))
        if identity_check < 1e-4:
            #### Check for inversion matrix in symmetry ops. Inversion symmetry
            #### can cause the principal axes operation to really be the 
            #### identity operation, but this is also not a symmetry operation 
            #### of the molecule. This is because the principal axes are 
            #### symmetric under more operations than the molecule itself. 
            inv = np.array([[-1,0,0], [0,-1,0], [0,0,-1]])
            for entry in sym_ops:
                diff = np.linalg.norm(inv - entry)
                if diff < 1e-8:
                    principal_axes = np.array([[1.,0,0], [0,1,0], [0,0,1]])
                    symmetric = True
                    # print("@@@@@@@@@@@ INVERSION @@@@@@@@@@@")
    
    return principal_axes,symmetric


def get_inertia_tensor(struct):
    """
    Calculates inertia tensor of the given structure. 
    
    """
    geo = struct.get_geo_array()
    ele = struct.elements
    mass =  np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele])
    
    inertial_tensor = np.zeros((3,3))
    ## Handle diangonal calculations first
    inertial_tensor[0,0] = np.sum(mass*(np.square(geo[:,1]) + np.square(geo[:,2])))
    inertial_tensor[1,1] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,2])))
    inertial_tensor[2,2] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,1])))
    
    ## Handle off diagonal terms
    inertial_tensor[0,1] = -np.sum(mass*geo[:,0]*geo[:,1])
    inertial_tensor[1,0] = inertial_tensor[0,1]
    
    inertial_tensor[0,2] = -np.sum(mass*geo[:,0]*geo[:,2])
    inertial_tensor[2,0] = inertial_tensor[0,2]
    
    inertial_tensor[1,2] = -np.sum(mass*geo[:,1]*geo[:,2])
    inertial_tensor[2,1] = inertial_tensor[1,2]
    
    total_inertia = np.dot(mass,np.sum(np.square(geo), axis=-1))
    inertial_tensor = inertial_tensor / total_inertia
    
    return inertial_tensor


def moit_pymatgen(struct):
    """
    Calculates the moment of inertia tensor for the system using Pymatgen. 
    However, this method is very slow due to Pymatgen performing many other 
    operations through the point group analyzer. It may still be used for
    validation purposes. Otherwise, use the high-performance moit function. 

    """
    mol = struct.get_pymatgen_structure()
    pga = PGA(mol)
    ax1,ax2,ax3 = pga.principal_axes
    return np.vstack([ax1,ax2,ax3])


def moit(struct):
    """
    Obtain the principal axes of the molecule using the moment of inertial 
    tensor.
    
    """
    inertial_tensor = np.zeros((3,3))
    
    geo = struct.get_geo_array()
    ele = struct.elements
    mass =  np.array([atomic_masses_iupac2016[atomic_numbers[x]] for x in ele])
    
    ### First center molecule
    total = np.sum(mass)
    com = np.sum(geo*mass[:,None], axis=0)
    com = com / total
    geo = geo - com
    
    ## Handle diangonal calculations first
    inertial_tensor[0,0] = np.sum(mass*(np.square(geo[:,1]) + np.square(geo[:,2])))
    inertial_tensor[1,1] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,2])))
    inertial_tensor[2,2] = np.sum(mass*(np.square(geo[:,0]) + np.square(geo[:,1])))
    
    ## Handle off diagonal terms
    inertial_tensor[0,1] = -np.sum(mass*geo[:,0]*geo[:,1])
    inertial_tensor[1,0] = inertial_tensor[0,1]
    
    inertial_tensor[0,2] = -np.sum(mass*geo[:,0]*geo[:,2])
    inertial_tensor[2,0] = inertial_tensor[0,2]
    
    inertial_tensor[1,2] = -np.sum(mass*geo[:,1]*geo[:,2])
    inertial_tensor[2,1] = inertial_tensor[1,2]
    
    total_inertia = np.dot(mass,np.sum(np.square(geo), axis=-1))
    
    inertial_tensor = inertial_tensor / total_inertia
    eigvals, eigvecs = np.linalg.eig(inertial_tensor)
    
    ax1,ax2,ax3 = eigvecs.T
    principal_axes = np.vstack([ax1,ax2,ax3])
    
    return principal_axes


def orientation(struct):
    """
    Returns a rotation matrix for the moment of inertial tensor 
    for the given molecule Structure. 
    
    Upon closer inspection, this orientation function is literlly just an 
    identity operation for rotation matrices...

    """
    axes = moit(struct)
    return np.linalg.inv(axes.T)
    

def show_axes(struct, ele=["He", "F", "Cl", "Br"]):
    """
    Visualize the COM of the molecule and the axes defined
    by the moment of inertial tensor of the molecule by adding
    an atom of type ele to the structure.

    """
    if len(ele) != 4:
        raise Exception("Must supply 4 element types for show_axes")
        
    com_pos = com(struct) 
    struct.translate(-com_pos)
    
    axes = get_principal_axes(struct)
    # axes = moit_pymatgen(struct)
    struct.append(com_pos[0],com_pos[1],com_pos[2],ele[0])
    for idx,row in enumerate(axes):
        row += com_pos
        struct.append(row[0],row[1],row[2],ele[idx+1])
    
    return struct


def rot_mol(rot, struct):
    """
    Applys rotation matrix to the geometry of the structure. 
    
    Arguments
    ---------
    frac: bool
        If frac is True, then the lattice of the Structure will be considered
        and the rotation will take place in fractional space before being
        converted back to cartesian space. 
    
    """
    geo = struct.get_geo_array()
    rot_geo = np.dot(rot, geo.T).T
    
    struct.from_geo_array(rot_geo, struct.elements)
    
    return struct



if __name__ == "__main__":
    pass
    

    
