
import json,copy,datetime,numbers

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation as R

import ase
from ase import Atoms
from ase.data import atomic_numbers,atomic_masses_iupac2016,chemical_symbols
from ase.formula import Formula
from ase.neighborlist import NeighborList,natural_cutoffs

from pymatgen.core import Lattice as LatticeP
from pymatgen.core import Structure as StructureP
from pymatgen.core import Molecule


class Structure(object):
    """
    Structure object is used for representation of molecules and crystals. This
    is the core object used for describing the geometry and properties in the 
    mcse library. 
    
    Structures may also be constructed from Pymatgen or ASE by called 
    Structure.from_ase and Structure.from_pymatgen respectively. This will 
    convert, in memory, from either the ase.atoms or pymatgen.structure/molecule
    into the mcse Structure object. 
    
    
    Arguments
    ---------
    struct_id: str
        String that is unique for the given structure
    
    """
    def __init__(self, 
                 struct_id="", 
                 geometry=[],
                 elements=[], 
                 lattice=[],
                 bonds=[],
                 properties={},
                 bonds_kw={"mult": 1.20, "skin": 0.0, "update": False},
                 trusted=False):
        
        bonds_kw = dict(bonds_kw)
        
        ### Call set methods that handle the appropriate type transformations
        self.set_geometry(geometry)
        self.set_elements(elements)
        self.set_lattice(lattice)
        self.set_properties(properties)
        self.set_bonds(bonds, bonds_kw=bonds_kw)
        self.set_id(struct_id)
        self.bonds_kw = bonds_kw
        self._molecules = []
        
        if not trusted:
            ### Can skip some slow checks if inputs can be trusted. 
            ###   This is mostly for use by developers, not users
            self.check_valid_struct()
        
    
    def __str__(self):
        if len(self.struct_id) == 0:
            self.get_struct_id(update=True)
        if len(self.get_lattice_vectors()) > 0:
            struct_type = "Crystal"
        else:
            struct_type = "Molecule"
        return "{}: {} {}".format(self.struct_id, 
                                  struct_type,
                                  self.formula)
    
    def __repr__(self):
        return self.__str__()
    
    
    def __iter__(self):
        return iter(zip(self.geometry, self.elements))
    
    def __getitem__(self, idx):
        return tuple([self.geometry[idx], self.elements[idx]])
    
    def __len__(self):
        return len(self.geometry)
    
    
    def set_id(self, struct_id):
        if len(struct_id) == 0:
            self.get_struct_id(update=True)
        else:
            self.struct_id = struct_id
    
    
    def set_geometry(self, geometry):
        if isinstance(geometry, (list)):
            geometry = np.array(geometry)
            
        if isinstance(geometry, (np.ndarray)):
            if len(geometry) > 0:
                if len(geometry.shape) == 1:
                    geometry = geometry[None,:]
            
            if len(geometry) == 0:
                pass
            elif geometry.shape[1] != 3:
                raise Exception("Geometry {} not in x,y,z format"
                                .format(geometry))
                
        else:
            raise Exception("Input geometry of type {} not recognized"
                            .format(type(geometry)))
                
        self.geometry = geometry
        
    
    def set_elements(self, elements):
        if isinstance(elements, (list)):
            elements = np.array(elements)
        
        if isinstance(elements, (np.ndarray)):
            if len(elements.shape) == 2:
                elements = elements.ravel()
        else:
            raise Exception("Input elements of type {} not recognized"
                            .format(type(elements)))
        
        for idx,entry in enumerate(elements):
            if isinstance(entry, (str, np.str_)):
                pass
            elif np.isscalar(entry):
                elements[idx] = chemical_symbols[int(entry)]
            else:
                raise Exception("Input element {} of type {} not recognized"
                            .format(entry, type(entry)))
        
        self.elements = elements
        
    
    def set_lattice(self, lattice):
        if isinstance(lattice, (list)):
            lattice = np.array(lattice)
        
        if isinstance(lattice, (np.ndarray)):
            if len(lattice) == 0:
                ### Non-periodic molecule
                pass
            elif lattice.shape != (3,3):
                raise Exception("Lattice {} not recognized as a (3,3) "
                    +"array of lattice vectors")
        else:
            raise Exception("Input lattice of type {} not recognized"
                            .format(type(lattice))) 
        
        self.lattice = lattice
        
    
    def set_bonds(self, bonds, bonds_kw={"mult": 1.20, "skin": 0.0, "update": False}):
        if len(bonds) > 0:
            self.properties["bonds_kw"] = bonds_kw
            self.properties["bonds"] = bonds
            self.bonds = bonds
            self.bonds_kw = bonds_kw
        else:
            self.bonds = [] 
            self.properties["bonds_kw"] = bonds_kw
            self.bonds_kw = bonds_kw
            self.get_bonds(**bonds_kw)
        
    
    def set_properties(self, properties):
        if len(properties) == 0:
            properties = dict(properties)
        self.properties = properties
                
    
    def check_valid_struct(self):
        """
        Checks if the values that are currently stored in the Structure 
        object constitute a valid structure. Lightweight function that only
        checks the shape of Structure attributes. 
        
        """
        if self.elements.shape[0] != self.geometry.shape[0]:
            if self.geometry.shape == (1,0) and self.elements.shape[0] == 0:
                ### Empty structure
                pass
            else:
                raise Exception("Number of elements {} does not match the number "
                            .format(self.elements.shape[0])+
                            "of positions {}"
                            .format(self.geometry.shape[0]))
                
                
    def get_struct_id(self, update=False):
        """
        Get the id for the structure. If a struct_id has already been stored, 
        this will be returned. Otherwise, a universal struct_id will be 
        constructed. If update is True, then the current struct_id
        will be discarded and a universal struct_id will be constructed.

        """
        if not update and len(self.struct_id) > 0:
            return self.struct_id
        else:
            ## Get type 
            name = ""
            if len(self.get_lattice_vectors()) > 0:
                name = "Structure"
            else:
                name = "Molecule"
            
            ## Get formula
            formula = Formula.from_list(self.elements) 
            ## Reduce formula, which returns formula object
            formula = formula.format("hill")
            ## Then get string representation stored in formula._formula
            formula = str(formula)
            ## Add formula to name
            name += "_{}".format(formula)

            ## Add Date
            today = datetime.date.today()
            
            year = str(today.year)
            month = str(today.month)
            month_fill = month.zfill(2)
            day = str(today.day)
            day_fill = day.zfill(2)
            name += "_"+year+month_fill+day_fill
            # name += "_{}{}{}".format(today.year,today.month,today.day)

            ## Add random string
            name += "_{}".format(rand_str(10))

            self.struct_id = name
            return self.struct_id
    
    @property
    def molecules(self):            
        mol_idx = self.get_molecule_idx(**self.bonds_kw)
        self._molecules = {}
        for idx,temp_idx in enumerate(mol_idx):
            temp_mol = self.get_sub(temp_idx, lattice=False)
            temp_mol.struct_id = "{}_Molecule_{}".format(self.struct_id,
                                                         idx)
            self._molecules[temp_mol.struct_id] = temp_mol
        
        return self._molecules
    
    
    @molecules.setter
    def molecules(self, obj):
        raise Exception("Cannot change molecules manually. First update "+
                "bonds_kw then call Structure.molecules again.")
            
    
    def translate(self, trans):
        """
        Translate the structure by the given translation vector. 
        
        Arguments
        ---------
        trans: iterable
            Iterable of shape (3,)
            
        """
        if isinstance(trans, (list)):
            trans = np.array(trans)[None,:]
        elif isinstance(trans, (np.ndarray)):
            if len(trans.shape) == 1:
                trans = trans[None,:]
        self.geometry += trans
        
    
    def rotate(self, rot, wrt="origin", frac=True, degrees=True, seq="xyz"):
        """
        Rotate molecule using rotation matrix. 
        
        Arguments
        ---------
        rot: array 
            Can be either a list of 3 euler angles in the given order or a 3,3 
            rotation matrix. 
        frac: bool
            If True and the Structure is a crystal, will rotate including the 
            rotation of the lattice vectors. 
        wrt: str
            Rotation performed with respect to any of the following options,
                ["origin", "com"]. Although, only origin implemented now.
        order: str
            Order for euler angles if rotation is given as euler angles. 
        """
        if wrt != "origin":
            raise Exception("Not Implemented")
        
        rot = np.array(rot)
        if rot.shape == (3,3):
            pass
        elif rot.ravel().shape == (3,):
            ### Assume euler angles
            Rot = R.from_euler(seq, rot.ravel(), degrees=degrees)
            rot = Rot.as_matrix()
        else:
            raise Exception(
                "Only rotation matrices and euler angles are currently implemented.")
        
        if len(self.lattice) == 0 or frac == False:
            self.geometry = np.dot(rot, self.geometry.T).T
        else:
            self.lattice = np.dot(rot, self.lattice.T).T
            frac_coords = self.cart2frac()
            frac_coords = np.dot(rot, frac_coords.T).T
            cart = self.frac2cart(frac_coords)
    
    
    def append(self, *args):
        """
        Append to the current Structure. Three input formats are accepted:
            
            Structure
                Input a Structure to combine with the current Structure
            
            geometry,elements
                Input a geometry array and element array to append
            
            x,y,z,element
                Adding a single new coordinate and element
            
        """
        if len(args) == 1:
            if isinstance(args[0], Structure):
                self.combine(args[0])
            else:
                raise Exception("Only one argument provided to append must "+
                    "be a Structure")
        elif len(args) == 2:
            if len(self.geometry) > 0:
                new_geo = np.vstack([self.geometry, args[0]])
                new_ele = np.hstack([self.elements, args[1]])
            else:
                new_geo = args[0]
                new_ele = args[1]
            self.set_geometry(new_geo)
            self.set_elements(new_ele)
        elif len(args) == 4:
            new_pos = np.array([args[0:3]])[None,:]
            if len(self.geometry) > 0:
                new_geo = np.vstack([self.geometry, new_pos])
                new_ele = np.hstack([self.elements, args[-1]])
            else:
                new_geo = new_pos
                new_ele = np.array([args[-1]])
            self.set_geometry(new_geo)
            self.set_elements(new_ele)
        else:
            raise Exception("Append arguments {} not recognized".format(args))
        
        self.check_valid_struct()
                
    
    def combine(self, struct):
        """
        Combine input Structure with current Structure. The lattice of the 
        current Structure will always be kept by default.
        
        """
        if len(self.geometry) == 0:
            geo = struct.geometry
            ele = struct.elements
        elif len(struct.geometry) == 0:
            geo = self.geometry
            ele = self.elements
        else:
            geo = np.vstack([self.geometry, struct.geometry])
            ele = np.hstack([self.elements, struct.elements])
            
        self.set_geometry(geo)
        self.set_elements(ele)
        
        if len(self.lattice) == 0:
            if len(struct.lattice) > 0:
                self.lattice = struct.lattice
        
        self.check_valid_struct()
        
    
    def from_geo_array(self, geometry, elements):
        """  Set geometry of structure to the input array and elements
        
        Arguments
        ---------
        Array: np.matrix of numbers
          Atom coordinates should be stored row-wise
        Elements: np.matrix of strings
          Element strings using shorthand notations of same number of rows 
          as the array argument
        """
        self.set_geometry(geometry)
        self.set_elements(elements)
        self.check_valid_struct()
    
    
    @classmethod 
    def from_geo(cls, array, elements, lat=[], struct_id=""):
        """
        Construction method of Structure object. 
        
        """
        struct = cls()
        struct.from_geo_array(array, elements)
        if len(struct_id) == 0:
            struct.get_struct_id(update=True)
        else:
            struct.struct_id = struct_id
        if len(lat) > 0:
            struct.set_lattice_vectors(lat)
        return struct
    
    
    def get_sub(self, idx, lattice=True, struct_id=""):
        """
        Returns the sub-structure with respect to provided indices. 
        
        Argumnets
        ---------
        idx: iterable
            Iterable of indices to construct the molecule structure.
        lattice: bool
            If True, will include the original lattice vectors
        """
        geo = self.get_geo_array()
        sub = Structure.from_geo(geo[idx], 
                                 self.elements[idx], 
                                 struct_id=struct_id)
        sub.properties["Parent_ID"] = self.struct_id
        if lattice:
            if len(self.lattice) > 0:
                sub.lattice = self.lattice
        sub.get_struct_id()
        return sub
        
    
    def set_property(self, key, value):
        self.properties[key] = value
        
    def delete_property(self, key):
        try: self.properties.pop(key)
        except: pass
    
    
    def get_property(self, key):
        try: return self.properties.get(key)
        except:
            try: self.reload_structure()  # may not have properly read property
            except Exception as e: print(e); return None
            
            
    def get_lattice_vectors(self):
        return self.lattice
    
    
    def get_geo_array(self):
        return self.geometry
    
    
    def get_ase_atoms(self):
        """ Works for periodic and non-periodic systems
        
        Purpose: Returns ase atoms object
        """
        symbols = self.elements
        positions = self.geometry
        cell = np.array(self.lattice)
        
        if len(symbols) == 0 or len(positions) == 0:
            raise Exception("Empty ase.Atoms object cannot be constructed")
        
        if len(cell) == 3:
            pbc = (1,1,1)
            return ase.Atoms(symbols=symbols, positions=positions,
                         cell=cell, pbc=pbc)
        else:
            pbc = (0,0,0)
            return ase.Atoms(symbols=symbols, positions=positions)
    
    @classmethod
    def from_ase(cls, atoms):
        """ 
        Construction classmethod for Structure from ase Atoms object. 
        
        """
        symbols = atoms.get_chemical_symbols()
        geo_array = atoms.get_positions()
        pbc = atoms.get_pbc()

        struct = cls()
        
        if pbc.any() == True:
            cell = atoms.get_cell()
            struct.lattice = np.vstack([cell[0], cell[1], cell[2]])
                
        struct.from_geo_array(geo_array,symbols)
        struct.molecules
        struct.get_struct_id(update=True)
        return struct
  
    
    def get_pymatgen_structure(self):
        """
        Inputs: A np.ndarry structure with standard "elements format
        Outputs: A pymatgen core structure object with basic geometric properties
        """
        if len(self.get_lattice_vectors()) > 0:
            lattice = LatticeP(self.lattice)
            structp = StructureP(lattice, self.elements, self.geometry,
                                 coords_are_cartesian=True)
            return structp
            
        else:
            coords = self.get_geo_array()
            symbols = self.geometry['element']
            molp = Molecule(symbols, coords)
            return molp
        
    @property
    def frac(self):
        """
        Returns fractional coordinates of the current geometry
        
        """
        return self.cart2frac()
    
    
    def cart2frac(self, pos=[], move_inside=False):
        """
        Get fraction coordinates of the input positions. If no input is given,
        then the fraction coordinates of the structure's geometry is given. 
        Result is always returned as a 2D array.
        
        """
        lv = np.array(self.lattice)
        if len(lv) == 0:
            raise Exception("Cannot get Fractional Coordinates for Structure "
                +"{} which has no lattice vectors.".format(self.struct_id))
            
        if len(pos) == 0:
            pos = self.geometry
        else:
            pos = np.array(pos)
            if len(pos.shape) == 1:
                pos = pos[None,:]
            elif len(pos.shape) > 2:
                raise Exception("Input positions must be a 2D array")
        
        lv_inv = np.linalg.inv(lv.T)
        frac = np.dot(lv_inv, pos.T).T
        
        if move_inside:
            frac %= 1
        
        return frac
    
    
    def frac2cart(self, frac=[], move_inside=False):
        lv = np.array(self.get_lattice_vectors())
        if len(lv) == 0:
            raise Exception("Cannot get Fractional Coordinates for Structure "
                +"{} which has no lattice vectors.".format(self.struct_id))
            
        frac = np.array(frac)
        if len(frac.shape) == 1:
            frac = frac[None,:]
        elif len(frac.shape) > 2:
            raise Exception("Input fractional positions must be a 2D array")
        
        if move_inside:
            offset = frac.astype(int)
            neg_offset_idx = np.argwhere(frac < -0.0001)
            offset[neg_offset_idx[:,0],neg_offset_idx[:,1]] -= 1
            frac -= offset
        
        cart = np.dot(frac, lv)
        
        return cart

    @classmethod
    def from_pymatgen(cls, pymatgen_obj):
        """ 
        Construction classmethod for Structure by converting pymatgen 
        Lattice/Molecule to Structure.
        
        """
        struct = cls()
        
        geometry = np.array([site.coords for site in pymatgen_obj])
        species = np.array([site.specie.name for site in pymatgen_obj])
        if type(pymatgen_obj) == Molecule:
            struct.from_geo_array(geometry,species)
            
        elif type(pymatgen_obj) == LatticeP:
            raise Exception('Lattice conversion not implemented yet')
        
        elif type(pymatgen_obj) == StructureP:
            struct.from_geo_array(geometry,species)
            struct.set_lattice(pymatgen_obj.lattice.matrix)
        struct.molecules
        struct.get_struct_id(update=True)
        return struct

    
    @property
    def formula(self):
        formula_dict = {}
        ele_list,count = np.unique(self.elements, return_counts=True)
        for idx,ele in enumerate(ele_list):
            ## Conversion to int to be JSON serializable
            formula_dict[ele] = int(count[idx])
        self.properties["formula"] = formula_dict
        return formula_dict

    
    @property
    def density(self):
        if self.lattice.shape != (3,3):
            raise Exception("Cannot get density of a molecule")
            
        volume = self.get_unit_cell_volume()
        mass = np.sum([atomic_masses_iupac2016[atomic_numbers[x]]
                       for x in self.elements])
    
        ## Conversion factor for converting amu/angstrom^3 to g/cm^3
        ## Want to just apply factor to avoid any numerical errors to due float 
        factor = 1.66053907
    
        self.properties["density"] = (mass / volume)*factor
        return self.properties["density"]

    
    @property
    def spg(self):
        return self.get_space_group()
    
    
    def get_space_group(self,  symprec=0.1, angle_tolerance=5.0, update=True):
        if self.lattice.shape != (3,3):
            raise Exception("Cannot get space group of a molecule")
        
        if update == False:
            if "space_group" in self.properties:
                return self.properties["space_group"]
            else:
                pass
        
        pstruct = self.get_pymatgen_structure()
        spg_symbol,spg_internation_number = \
                pstruct.get_space_group_info(symprec=symprec,
                                    angle_tolerance=angle_tolerance)
    
        self.properties["space_group"] = spg_internation_number
        return self.properties["space_group"]


    def get_lattice_angles(self):
        if len(self.lattice) == 0:
            raise Exception("Tried to get_lattice_angles for empty lattice")
        if self.lattice.shape != (3,3):
            raise Exception("Tried to call get_lattice_angles for "+
                "improper lattice {}".format(self.lattice))
        A = self.lattice[0]
        B = self.lattice[1]
        C = self.lattice[2]
        alpha = self.angle(B, C)
        beta = self.angle(C, A)
        gamma = self.angle(A, B)
        return alpha, beta, gamma


    def get_lattice_magnitudes(self):
        return np.linalg.norm(self.lattice, axis=-1)


    def get_unit_cell_volume(self, update=False):
        if update == False:
            if "cell_vol" in self.properties:
                self.properties["unit_cell_volume"] = self.properties["cell_vol"]
                return self.properties["cell_vol"]
            if "unit_cell_volume" in self.properties:
                return self.properties["unit_cell_volume"]

        self.properties["unit_cell_volume"] = np.linalg.det(self.lattice)
        return self.properties["unit_cell_volume"]
    
    
    def set_lattice_vectors(self, lat):
        self.set_lattice(lat)


    def angle(self, v1, v2):
        numdot = np.dot(v1,v2)
        anglerad = np.arccos(numdot/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        angledeg = anglerad*180/np.pi
        return angledeg
    
    
    def document(self, _id=""):
        """
        Turn Structure object into a document for MongoDB.
        
        Arguments
        ---------
        _id: str
           The _id for the document in the MongoDB. Default behavior is to
           use the struct_id as the _id.
        """
        struct_doc = dict(self.__dict__)
        struct_doc["geometry"] = self.geometry.tolist()
        struct_doc["elements"] = self.elements.tolist()
        struct_doc["lattice"] = self.lattice.tolist()
        
        if len(_id) == 0:
            struct_doc["_id"] = self.struct_id
        else:
            struct_doc["_id"] = _id
        return struct_doc
    

    # json data handling packing
    def dumps(self):
        data_dictionary = {}
        data_dictionary['properties'] = dict(self.properties)
        data_dictionary['struct_id'] = self.struct_id
        data_dictionary['geometry'] = self.geometry.tolist()
        data_dictionary["elements"] = self.elements.tolist()
        data_dictionary["lattice"] = self.lattice.tolist()
        
        data_dictionary['properties'] = self._properties_to_json(data_dictionary['properties'])
        
        return json.dumps(data_dictionary, indent=4)
    
    
    def _properties_to_json(self, properties):
        for iter_idx,value in enumerate(properties):
            if isinstance(properties, dict):
                key = value
                value = properties[value]
            else:
                key = iter_idx
            if isinstance(value, np.ndarray):
                properties[key] = value.tolist()
            elif isinstance(value, list):
                for idx,entry in enumerate(value):
                    value[idx] = self._properties_to_json(entry)
            elif isinstance(value, dict):
                properties[key] = value
            elif isinstance(value, np.integer):
                properties[key] = int(value)
            elif isinstance(value, (np.bool_)):
                properties[key] = bool(value)
            elif type(value).__module__ == np.__name__:
                properties[key] = float(value)
            else:
                pass
        return properties
        
    
    def loads(self, json_string):
        data_dictionary = json.loads(json_string)
        
        try: self.struct_id = data_dictionary['struct_id']
        except: pass
    
        self.geometry = np.array(data_dictionary['geometry'])
        self.elements = np.array(data_dictionary["elements"])
        self.lattice = np.array(data_dictionary["lattice"])
        
        if "bonds_kw" in data_dictionary["properties"]:
            self.bonds_kw = data_dictionary["properties"]["bonds_kw"]
        
        ## Delete data used from data_dictionary and move everything that's
        ## left into the properties section
        self.properties = data_dictionary['properties']
        del(data_dictionary["properties"])
        if "struct_id" in data_dictionary:
            del(data_dictionary["struct_id"])
        del(data_dictionary["elements"])
        del(data_dictionary["geometry"])
        del(data_dictionary["lattice"])

        for key,value in data_dictionary.items():
            self.properties[key] = value
        
        self.molecules
            
            
    def copy(self):
        return Structure(
                    struct_id=copy.deepcopy(self.struct_id), 
                    geometry=copy.deepcopy(self.geometry),
                    elements=copy.deepcopy(self.elements), 
                    lattice=copy.deepcopy(self.lattice),
                    bonds=copy.deepcopy(self.bonds),
                    properties=dict(self.properties),
                    bonds_kw=dict(self.bonds_kw)
                    )
    
    
    @classmethod
    def from_dict(cls,dictionary):
        struct = cls()
        properties = dictionary["properties"]
        struct.struct_id = dictionary["struct_id"]
        geometry = dictionary["geometry"]
        elements = dictionary["elements"]
        lattice = dictionary["lattice"]
        
        ### Call set methods that handle the appropriate type transformations
        struct.set_geometry(geometry)
        struct.set_elements(elements)
        struct.set_lattice(lattice)
        struct.set_properties(properties)
        
        return struct
        
    
    def get_bonds(self, mult=1.20, skin=0.0, update=False):
        """
        Returns array of covalent bonds in the molecule. In addition, these
        are stored in the Structure properties for future reference. 
        
        Arguments
        ---------
        mult: float
            For ASE neighborlist
        skin: float
            For ASE neighborlist
        update: bool
            If True, will force an update of bond information.
            
        Returns
        -------
        list
            The index of the list corresponds to the atom the bonds are 
            describing. Inside each index is another list. This is the indices
            of the atoms the atom is bonded. Please keep in mind that Python
            iterables are zero indexed whereas most visualization softwares
            will label atoms starting with 1. 
        
        """
        temp_bonds = []
        if "bonds" in self.properties:
            temp_bonds = self.properties["bonds"]
        elif len(self.bonds) > 0:
            temp_bonds = self.bonds
        if update == False and len(temp_bonds) > 0:
            pass
        else:
            if len(self.geometry) > 0:
                atoms = self.get_ase_atoms()
                cutOff = natural_cutoffs(atoms, mult=mult)
                neighborList = NeighborList(cutOff, self_interaction=False,
                                            bothways=True, skin=skin)
                neighborList.update(atoms)
                
                # Construct bonding list indexed by atom in struct
                bonding_list = [[] for x in range(self.geometry.shape[0])]
                for i in range(self.geometry.shape[0]):
                    temp_list = list(neighborList.get_neighbors(i)[0])
                    if len(temp_list) > 0:
                        temp_list.sort()
                    bonding_list[i] = [int(x) for x in temp_list]
            else:
                bonding_list = []
            
            self.properties["bonds"] = bonding_list
            self.bonds = bonding_list
            
        return self.properties["bonds"]
    
    
    def get_molecule_idx(self, mult=1.20, skin=0.0, update=False):
        if update == False:
            if "molecule_idx" in self.properties:
                return self.properties["molecule_idx"]
            
        bonds = self.get_bonds(mult, skin, update)
        
        ## Build connectivity matrix
        graph = np.zeros((self.geometry.shape[0],self.geometry.shape[0]))
        for atom_idx,bonded_idx_list in enumerate(bonds):
            for bond_idx in bonded_idx_list:
                graph[atom_idx][bonded_idx_list] = 1
        
        graph = csr_matrix(graph)
        n_components, component_list = connected_components(graph)
        molecule_idx_list = [list(np.where(component_list == x)[0])
                                for x in range(n_components)]
        
        self.properties["molecule_idx"] = molecule_idx_list
        
        return self.properties["molecule_idx"]
    
    
def rand_str(length):
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    np_alphabet = np.array([x for x in alphabet])
    rand = np.random.choice(np_alphabet, size=(length,), replace=True)
    return "".join(rand)



if __name__ == '__main__':
    pass
