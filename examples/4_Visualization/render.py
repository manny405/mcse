
"""
This file covers the same examples demonstred in the README in the form of a 
python script

"""

from mcse.io import read,write

##############################################################################
#### Molecule Examples                                                    ####  
##############################################################################
# from mcse.molecules.render import AlignedRender,AreaMaximizedRender,AreaMinimizedRender

# struct_dict = read("Example_Structures")
# interactive = False
# dpa = 100

# ### AlignedRender aligns molecule's principal axis with 
# ar = AlignedRender(dpa=dpa, interactive=interactive)
# ar.calc(struct_dict["cl20"])
# ar.write("Aligned_Images")
# ar.calc(struct_dict["benzene"])
# ar.write("Aligned_Images")

# ### Use viewing angle that maximizes visible area of molecule
# ar = AreaMaximizedRender(dpa=dpa, interactive=interactive)
# ar.calc(struct_dict["cl20"])
# ar.write("Area_Maximized_Render")
# ar.calc(struct_dict["benzene"])
# ar.write("Area_Maximized_Render")
                                               
# ### Use viewing angle that minmizes visible area of molecule
# ar = AreaMinimizedRender(dpa=dpa, interactive=interactive)
# ar.calc(struct_dict["cl20"])
# ar.write("Area_Minimized_Render")
# ar.calc(struct_dict["benzene"])
# ar.write("Area_Minimized_Render")


##############################################################################
#### Crystal Examples                                                     ####  
##############################################################################
# from mcse.crystals.render import AlignedRender,SupercellRender,MotifRender
# from mcse.crystals import standardize

# struct_dict = read("Example_Structures")   
# interactive = False
# dpa = 50
# supercell_mult=(2,2,2)

# standardize(struct_dict["TATNBZ"])
# standardize(struct_dict["DATNBZ01"])

# ### Standard viewing angle for crystal is the (1,1,0) direction
# ar = AlignedRender(dpa=dpa, interactive=interactive)
# # ar.calc(struct_dict["TATNBZ"])
# ar.write("Aligned_Crystal_Images")
# ar.calc(struct_dict["DATNBZ01"])
# ar.write("Aligned_Crystal_Images")

# ### Visualization for (2,2,2) supercell
# ar = SupercellRender(dpa=dpa, supercell_mult=supercell_mult, interactive=interactive)
# ar.calc(struct_dict["TATNBZ"])
# ar.write("Supercell_Images")
# ar.calc(struct_dict["DATNBZ01"])
# ar.write("Supercell_Images")
                                               
# ### Viewing angle is calculated from supercell for visualization of the 
# ### motif of the crystal
# ar = MotifRender(dpa=dpa, supercell_mult=supercell_mult, interactive=interactive)
# ar.calc(struct_dict["TATNBZ"])
# ar.write("Motif_Images")
# ar.calc(struct_dict["DATNBZ01"])
# ar.write("Motif_Images")

##############################################################################
#### RMSD Example                                                         ####  
##############################################################################
# from mcse.crystals import RMSD
# from mcse.molecules.render import OverlappingClusters

# dpa = 50
# interactive = True

# rmsd = RMSD(nn=12, search_nn=16)
# struct_dict = read("RMSD_Example_Structures")
# oc = OverlappingClusters(dpa=dpa, interactive=interactive)

# rmsd.calc_struct(struct_dict["3ddf4a302a"], struct_dict["8aeae1272d"])
# oc.calc_struct(rmsd.overlapping_clusters)
# oc.write("RMSD_Images")

# rmsd.calc_struct(struct_dict["3ddf4a302a"], struct_dict["fafdb8075b"])
# oc.calc_struct(rmsd.overlapping_clusters)
# oc.write("RMSD_Images")

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,1))
# ax = fig.add_subplot(111)
# oc.matplotlib_colorbar(ax=ax)
# plt.tight_layout()
# fig.savefig("RMSD_Images/colorbar.pdf")



