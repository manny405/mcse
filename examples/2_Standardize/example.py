
"""
This file covers the same examples demonstred in the README in the form of a 
python script

"""

from mcse import com
from mcse.io import read,write
from mcse.crystals import standardize,FindMolecules
from mcse.molecules.standardize import standardize as standardize_mol
from mcse.molecules import get_principal_axes

### Standardizing crystal structure
print("------------- Crystal Standardize -------------")
struct = read("PUBMUU02.cif")
print(struct.elements)
standardize(struct)
struct.elements


### Closer look at FindMolecule details as first example of analysis Driver
struct = read("PUBMUU02.cif")
fm = FindMolecules()
fm.calc(struct)


### Standardizing molecules
print("------------- Molecule Standardize -------------")
mol = read("rdx.xyz")
print("------------- Before Standardize -------------")
print(mol.elements)
print(com(mol))
print(get_principal_axes(mol))
standardize_mol(mol)
print("------------- After Standardize -------------")
print(mol.elements)
print(com(mol))
print(get_principal_axes(mol))