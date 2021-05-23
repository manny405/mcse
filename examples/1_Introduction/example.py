
"""
This file covers the same examples demonstred in the README in the form of a 
python script

"""

from mcse.io import read,write


### Reading single files
crystal = read("BENZEN.cif")
molecule = read("benzene.xyz")
print(crystal)
print(molecule)
write("BENZEN", crystal)
write("molecule", molecule, file_format="xyz")

### Cannot overwrite existing files without overwrite=True
write("molecule", molecule, file_format="xyz", overwrite=True)

### Reading and writing a directory of files
struct_dict = read("Example_Structures")
write("Structure_Directory", struct_dict)

### Standardizing a Crystal Structure
from mcse.crystals import standardize
struct = read("BENZEN.cif")
standardize(struct)

### Writing molecules that make up a structure
write("BENZENE_Molecules", struct.molecules)



### From ASE
from mcse import Structure
from ase import Atoms
h2_atoms = Atoms('H2',positions=[[0, 0, 0],[0, 0, 0.7]])
print(h2_atoms)
h2_mcse = Structure.from_ase(h2_atoms)
print(h2_mcse)



### From Pymatgen
import pymatgen.core as pmg
lattice = pmg.Lattice.cubic(4.2)
cscl_pmg = pmg.Structure(lattice, ["Cs", "Cl"],[[0, 0, 0], [0.5, 0.5, 0.5]])
print(cscl_pmg)
cscl_mcse = Structure.from_pymatgen(cscl_pmg)
print(cscl_mcse)