
"""
This file covers the same examples demonstred in the README in the form of a 
python script

"""

from mcse.io import read,write

##############################################################################
#### FindMolecules Detailed Example                                       ####  
##############################################################################
# from mcse.crystals.find_molecules import FindMolecules

# fm = FindMolecules(residues=0, mult=1.05, conformation=True)
# struct = read("PUBMUU02.cif")
# fm.calc(struct)

# struct_dict = read("Example_Structures")
# fm.calc(struct_dict)

# fm.calc(struct_dict, write=True)
# for struct_id,struct in struct_dict.items():
#     fm.calc(struct)
#     fm.write("Molecules", file_format="xyz", overwrite=True)
    
    
##############################################################################
#### Packing Factor Example                                               ####  
##############################################################################
# from mcse.crystals.packing_factor import PackingFactor

# pf =  PackingFactor(spacing=0.25)
# struct = read("PUBMUU02.cif")
# result = pf.calc(struct)


##############################################################################
#### Motif Example                                                        ####  
##############################################################################
# from mcse.crystals.motif import Motif

# struct_dict = read("Example_Structures")
# m = Motif()
# m.calc(struct_dict)

# for struct_id,struct in struct_dict.items():
#     print(struct_id,struct.properties["Motif"])
    
    
    
##############################################################################
#### Duplicate Check Example                                              ####  
##############################################################################
# from mcse.crystals import DuplicateCheck

# struct_dict = read("DuplicateCheck_Example_Structures")
# dc = DuplicateCheck()
# dc.calc(struct_dict, write=True)


##############################################################################
#### RMSD Example                                                         ####  
##############################################################################
# from mcse.crystals.rmsd import RMSD

# struct_dict = read("RMSD_Example_Structures")
# rmsd = RMSD(nn=12, search_nn=16, verbose=True)    
# rmsd.calc(struct_dict)

