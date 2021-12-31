
#%%

import numpy as np
from scipy.spatial.transform import Rotation as R

from mcse.io import read,write
from mcse.molecules import rot_mol
from mcse.molecules.align import fast_align
from mcse.molecules.compare import compare_rmsd_pa,generate_combined

#%%

### Read in file
ben = read("benzene.xyz")

### Align principal axes of inertia with the origin
fast_align(ben)

### Can save this for visualization purposes
write("aligned_benzene", ben, file_format="xyz", overwrite=True)

# %%

### Prepare example rotation matrix
rot = R.from_euler("xyz", [90,0,0], degrees=True).as_matrix()

### Apply rotation to copy of benzene
rben = rot_mol(rot, ben.copy())
### Apply global translation just for fun
rben.geometry += np.array([10,0,0])
### Apply translation to first atom just for fun
rben.geometry[0] += np.array([0.25,0,0])

### Use compare function to get rotation matrix
results = compare_rmsd_pa(ben,rben)

### Boolean if the comparison determined geometries are duplicates
duplicates = results[0]
### RMSD calculated between molecules
rmsd = results[1]
### Tuple of translates and rotations found to best match molecules
trans_tuple = results[2]
rot_tuple = results[3]
### Reordering of atoms to provide minimum RMSD
idx_tuple = results[4]
### If rotation matrix found would break chirality of molecule. Not important
###   for benzene
chiral = results[-1]

# %%

### Use this function to easily visualize the molecules overlapped on one 
### another after apply the rotations & translations found by compare_rmsd_pa
combined = generate_combined(ben,rben,results)

write("combined_benzene", combined, file_format="xyz", overwrite=True)

# %%

### Can use build in visualizer to visualized combined geometry more clearly.
###   Commented out 
# from mcse.molecules.render import OverlappingClusters
# ar = OverlappingClusters(dpa=50, interactive=True)
# ar.calc_struct(combined)

# %%
