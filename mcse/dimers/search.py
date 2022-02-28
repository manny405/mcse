

from mcse.core.driver import BaseDriver_



class DimerGridSearch(BaseDriver_):
    """
    Generates all dimer structures that should be considered for a grid search
    to find the best dimer arangements. Grid search is performed over all 
    x,y,z positions for the COM and all orientations of the molecule. Only
    dimers with physically relevant intermolecular distances are kept for the  
    user by providing maximum and minimum scaled vdW distances as max_sr and 
    min_sr. Grid search can be performed using a a single unqiue molecule or 
    two distinct molecules as input. 
    
    This method is parallelized using MPI. The user may launch as many MPI ranks
    as they would like in order to reduce the computational burden for each 
    rank and speedup the time-to-solution. 
    
    Arguments
    ---------
    min_sr: float
        Minimum specific radius to use for dimer distance checks.
    max_sr: float
        Maximum specific radius multiplier that is allowed to be the minimum 
        distance between two dimers, thereby removing dimers formed from molecules
        that are far away. 
    box: float,list
        Box size to search over for x,y,z positions. It's assumes that first 
        molecule of the dimer will be placed at 0,0,0. If the box size is a 
        float, a box will be placed at 0,0,0 and will extend by this value in
        all directions. If a list is provided, the box will only extend by these
        lengths in the x,y,z directions respectively, and due to symmetry, in 
        the -x,-y,-z directions. Default behavior is that the box size will
        automatically be detected based on the size of the input molecules.
    grid_spacing: float
        Grid spacing to use for x,y,z position spacing
    angle_spacing: float
        Spacing of orientation angles to use for every x,y,z position. Assumed
        to be in degrees. 
    cutoff: float
        Distance between COM to neglect from dimer grid search. 
    tol: float
        Tolerance used for the rmsd comparison. If the difference between the
        structures is less than tol, then they are considered duplicates. 
    vdw: list
        List of all vdw radii for all elements in periodic table
    bonds_kw: dict
        Keyword arguments for Structure.get_bonds method. This is used
        for recognizing molecular connectivity. 
    inter_list: list
        List of tuples of elements that should be considered for the distance
        calculations. For example ("Li", "O"). Then, if the distance between the
        Li in one molecule and the O in another molecule is outside the min_sr
        to max_sr range then the dimer system will be removed. This is helpful
        to reduce the search space based on chemical intution.
    
    """
    def __init__(self, 
                 folder="",
                 min_sr=0.75,
                 max_sr=1.30,
                 box=-1, 
                 grid_spacing=2.5, 
                 angle_spacing=30, 
                 inter_list=[],
                 tol=0.1,
                 vdw=[],
                 bonds_kw={},
                 comm=None):
        raise Exception()