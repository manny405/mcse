
"""
This file covers the same examples demonstrated in the README in the form of a 
python script

"""

import time,os

from mcse.io import read,write
from mcse.libmpi import ParallelCalc

# ##############################################################################
# #### Packing Factor Example                                               ####  
# ##############################################################################
# from mcse.crystals.packing_factor import PackingFactor
# from mcse.crystals.motif import Motif

# pf =  PackingFactor(spacing=0.25)
# pc = ParallelCalc(
#                   struct_path="Parallel_Example_Structures", 
#                   output_path="Packing_Factor_Calculated", 
#                   driver=pf,
#                   overwrite=True,
#                   verbose=False)

# start = time.time()
# pc.calc()
# end = time.time()
# print("{}: {}".format(pc.rank, end-start))


##############################################################################
#### DuplicateCheck Example                                               ####  
##############################################################################
# from mcse.crystals import DuplicateCheck

# struct_dict = read("DuplicateCheck_Example_Structures")
# dc = DuplicateCheck()
# pc = ParallelCalc(
#                   struct_path="Parallel_Example_Structures", 
#                   output_path="Duplicate_Check_Calculated", 
#                   driver=dc,
#                   overwrite=True,
#                   verbose=False)

# start = time.time()
# pc.calc()    
# end = time.time()
# print("{}: {}".format(pc.rank, end-start))
    
    


