
import os
import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist,pdist,squareform

from ase.data import atomic_numbers,covalent_radii
from ase.data.vdw_alvarez import vdw_radii
from ase.data.colors import jmol_colors

from mcse import Structure
from mcse.io import check_dir,check_overwrite
from mcse.core.driver import BaseDriver_


all_radii = []
for idx,value in enumerate(vdw_radii):
    if np.isnan(value):
        value = covalent_radii[idx]
    all_radii.append(value)
all_radii = np.array(all_radii)

### Change from jmol's red bromine color to a copper color
jmol_colors[35] = jmol_colors[29]


class Render(BaseDriver_):
    """
    Renders a extremely high quality image of an atomic system using VTK. 
    All settings for image rendering are passed in during instantiation of 
    the class. Default values that achieves a visually appealing image have
    been set. 
    
    Argumnets
    ---------
    dpa: int
        Dots per angstrom. The implemented method for determining the 
        resolution of the image is based on this value. The number of pixels
        of the molecule is scaled by its length in an effort to construct 
        the render window that is commensurate with the shape of the molecule. 
    vdw: float
        Coefficient to multiply the van der Waals radii by when drawing the 
        atomic spheres. 
    bonds: float
        Radius for the cylinders used to plot bonds. 
    unit_cell: float
        Radius for the cylinders used for the unit cell. 
    interactions: float
        Radius for the cylinders used to plot interactions. 
    interaction_length: float
        Units of length for the dotted cylinder line
    interaction_gap: float
        Ratio of the length value. Should be given as a value less than 1.0. 
    interaction_color: iterable
        RGB value for the ineraction cylinders.
    bond_color: iterable
        RGB value for the color of the cylinders drawn for covalent bonds. If
        no value is provided, then the default algorithm for bond coloring will 
        be used.
    plot_bonds: bool
        Determines if covalent bonds will be drawn for the molecule. 
    periodic_bonds: bool
        If False, then no bonds that go across the periodic boundaries of the 
        unit cell will be drawn. 
    bond_kw: dict
        Dictionary of keyword arguments that are pass to the 
        mcse.Structure.get_bonds method.
    ele_colors: iterable
        Iterable containing RGB values for the colors to use for each element. 
        Iterable should be ordered according to atomic numbers. 
    vdw_radii: iterable
        Iterable containing floats that describe the van der Waals radii for 
        each element. Iterable should be ordered according to atomic numbers. 
    atom_metallic: float
        VTK material metallic value for atomic spheres. 
    atom_roughness: float
        VTK material roughness value for atomic spheres. 
    *_resolution: int
        Resolution used to create the sphere/cylinder for atoms, bonds, and 
        intermolecular interactions. 
    
    """
    def __init__(self, 
                 dpa=50, 
                 vdw=0.30,
                 bonds=0.15,
                 unit_cell=0.075,
                 interactions=0.075,
                 plot_bonds=True,
                 periodic_bonds=False,
                 depth_of_field=False, 
                 bonds_kw={"mult": 1.20, "skin": 0.0, "update": False},
                 ele_colors=jmol_colors,
                 vdw_radii=all_radii, 
                 atom_metallic=0.05,
                 atom_roughness=0.2,
                 atom_resolution=100,
                 bond_metallic=0.05,
                 bond_roughness=0.15,
                 bond_color=(),
                 bond_resolution=100,
                 default_lattice_vector_color=(0,0,0),
                 color_abc = [(1.,0,0), 
                              (0,1.,0),
                              (0,0,1.)
                              ],
                 lattice_vector_labels = ("a", "b", "c"),
                 lattice_vector_label_color=(0.11,0.11,0.11),
                 lattice_vector_label_size = 55,
                 unit_cell_resolution = 100,
                 unit_cell_metallic=0.0,
                 unit_cell_roughness=0.1,
                 interaction_length=0.2, 
                 interaction_gap=0.4, 
                 interaction_color=(0.09019607843137255, 
                                    0.7450980392156863,
                                    0.8117647058823529), 
                 interaction_metallic=0.0,
                 interaction_roughness=0.5,
                 interaction_resolution=100,
                 background_color = (1,1,1),
                 file_name="",
                 num_close_interactions=0,
                 interactive=True,
                 ):
        
        self.dpa = dpa
        
        ### These names are not exactly descriptive of what they control. 
        ### More descriptive names would be vdw_coef and bond_radius. However, 
        ### the names have been simplified in favor of creating a simpler API
        ### for the user. 
        self.vdw = vdw
        self.bonds = bonds
        self.unit_cell = unit_cell
        self.interactions = interactions
        
        self.ele_colors = ele_colors
        self.vdw_radii = vdw_radii
        self.plot_bonds = plot_bonds
        self.periodic_bonds = periodic_bonds
        self.depth_of_field = depth_of_field
        self.bonds_kw = bonds_kw
        
        ### Settings for atomic spheres
        self.atom_metallic = atom_metallic
        self.atom_roughness = atom_roughness
        self.atom_resolution = atom_resolution
        
        ### Settings for bond cylinders
        self.bond_roughness = bond_roughness
        self.bond_metallic = bond_metallic
        self.bond_color = bond_color
        self.bond_resolution = bond_resolution
        
        ### Settings for unit cell cylinders
        self.color_abc = color_abc
        self.default_lattice_vector_color = default_lattice_vector_color
        self.lattice_vector_labels = lattice_vector_labels
        self.lattice_vector_label_color = lattice_vector_label_color
        self.lattice_vector_label_size = lattice_vector_label_size
        self.unit_cell_resolution = unit_cell_resolution
        self.unit_cell_metallic = unit_cell_metallic
        self.unit_cell_roughness = unit_cell_roughness
        
        ### Settings for interaction cylinders
        self.num_close_interactions = num_close_interactions
        self.interaction_length = interaction_length
        self.interaction_gap =  interaction_gap
        self.interaction_color = interaction_color
        self.interaction_metallic = interaction_metallic
        self.interaction_roughness = interaction_roughness
        self.interaction_resolution = interaction_resolution
        
        ### Other image settings
        self.background_color = background_color
        
        self.file_name = file_name
        self.interactive = interactive
        
        self.window_size = (600,600)
        self.scale = 0
        self.extent = []
        self.initialize_render(self.window_size,
                               self.scale,
                               self.extent)
        
        if self.depth_of_field == True:
            raise Exception("Depth of field is not implemented")
        
        
    
    def initialize_render(self,
                          window_size=(600,600),
                          scale=0, 
                          extent=[]):
        """
        Initializes everything need for a new render. Everytime a new image is
        to be made, this method needs to be called in order to remove all 
        previous sources and settings from the previous render. 
        
        """
        self.renderer = vtk.vtkRenderer()
        self.renderer.UseImageBasedLightingOn()
        self.renderer.SetBackground(self.background_color)
        self.renderer.SetLayer(0)
        
        if len(extent) > 0:
            self.renderer.GetActiveCamera().SetFocalPoint(
                          0.5*(extent[0]+extent[1]),
                          0.5*(extent[2]+extent[3]),
                          0.5*(extent[4]+extent[5]))
            self.renderer.GetActiveCamera().SetPosition(
                                0.5*(extent[0]+extent[1]),
                                0.5*(extent[2]+extent[3]),
                                (extent[1]-extent[0])*(extent[3]-extent[2])*1.25)
                                ### Location in the Z direction should be 
                                ### greater than the width of the structure to
                                ### avoid any clipping
                                # np.max([extent[1] - extent[0],
                                #         extent[3] - extent[2]])*1.25)
            ### Make sure clipping range respects the camera position
            max_range = (extent[1]-extent[0])*(extent[3]-extent[2])*1.25
            self.renderer.GetActiveCamera().SetClippingRange(0.01, max_range*2)
                            
            
            if scale > 0:
                self.renderer.GetActiveCamera().SetParallelProjection(1)
                self.renderer.GetActiveCamera().SetParallelScale(0.5*
                                    (extent[3] - extent[2]))
        
        ### For user controlled lighting
        ### However, the user will have to overwrite this add_lighting method
        ### using class inheritance.
        self.add_lighting(self.renderer)
        
        ### Unit cell label renderer will be in a different layer
        self.ucv_label_renderer = vtk.vtkRenderer()
        self.ucv_label_renderer.UseImageBasedLightingOn()
        self.ucv_label_renderer.SetLayer(1)
        ### Make sure both renderers get the same camer
        self.ucv_label_renderer.SetActiveCamera(
            self.renderer.GetActiveCamera())
        self.add_lighting(self.ucv_label_renderer)
        
        ### Initialize render window
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetWindowName("mcse")
        self.renderWindow.SetSize(window_size)
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.AddRenderer(self.ucv_label_renderer)
        self.renderWindow.SetNumberOfLayers(2)
        
        
        if not self.interactive:
            self.renderWindow.SetOffScreenRendering(1)
            
    
    # @classmethod
    # def render(self, struct, rotation=[0,0,0], filename="", interactive=True):
    #     """
    #     Simple method for users to use to quickly render an image of a 
    #     structure.
        
    #     """
    
    def calc_struct(self, struct):
        ### Decide if plotting unit cell
        if len(struct.get_lattice_vectors()):
            if self.unit_cell:
                unit_cell = True
                
        self.window_size = self.get_window_size(struct, self.dpa, self.vdw, unit_cell)
        self.scale = self.get_scale(struct, unit_cell=unit_cell)
        self.extent = self.get_extent(struct, unit_cell=unit_cell)
        self.initialize_render(self.window_size, self.scale, self.extent)
        self.struct = struct
        self.add_geometry(struct)
        self.add_close_interactions()
        
        if len(self.struct.get_lattice_vectors()) > 0:
            self.add_unit_cell(np.vstack(struct.get_lattice_vectors()))
            self.add_unit_cell_labels(np.vstack(struct.get_lattice_vectors()))
        
        if self.interactive == True:
            self.start_interactive()
    
    
    def start_interactive(self):
        self.renderWindow.Render()
        self.renderWindow.SetWindowName("mcse")
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.Start()
            
    
    def write(self, output_dir="./", file_format="png", overwrite=True):
        """
        Writes rendered image to a file. Note that the arguments to the 
        write class follow the mcse.Driver API. 
        
        """
        check_dir(output_dir)
        if len(self.file_name) == 0:
            temp_file_name = "{}.{}".format(self.struct.struct_id, 
                                            file_format)
        else:
            temp_file_name = self.file_name
            
        if file_format != "png":
            raise Exception("The only supported file format is png")
        
        file_path = os.path.join(output_dir, temp_file_name)
        check_overwrite(file_path, overwrite=overwrite)
        
        self.renderWindow.Render()
        
        image_filter = vtk.vtkWindowToImageFilter()
        image_filter.SetInput(self.renderWindow)
        image_filter.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(file_path)
        writer.SetInputConnection(image_filter.GetOutputPort())
        writer.Write()
        
    
    def get_window_size(self, struct, dpa, vdw=-1, unit_cell=False):
        """
        Obtains render window size that is commensurate with the shape of the 
        molecule. Scales the shape of the molecule by the dots per angstrom
        to obtain the final window size. 
        
        """
        if vdw < 0:
            vdw = self.vdw
            
        geo = struct.get_geo_array()
        ele = struct.elements
        temp_radii = np.array([self.vdw_radii[atomic_numbers[x]] for x in ele]
                              )[:,None]*vdw
        min_geo = geo - temp_radii
        max_geo = geo + temp_radii
        
        x_dim = np.max(max_geo[:,0]) - np.min(min_geo[:,0])
        y_dim = np.max(max_geo[:,1]) - np.min(min_geo[:,1])
        
        ### Account for position of the unit cell
        if unit_cell:
            lattice_sites = self.get_unit_cell_sites(struct)
            if len(lattice_sites) > 0:
                ### Account for radius of unit cell cylinders and for the 
                ### axis labels
                offset = self.unit_cell*1.01
                if len(self.lattice_vector_labels) > 0:
                    ### Fontsize given in points
                    ### There are 72 points per 1 tall
                    ### Therefore, treat as though each lattice site also has
                    ### sphere of radius given by the fontsize 
                    text_radius = self.lattice_vector_label_size / 24 * 0.5 
                    offset += text_radius
                
                min_lattice_sites = lattice_sites - np.array([
                                offset*0.4,self.unit_cell*1.01,0])
                max_lattice_sites = lattice_sites + np.array([offset*0.4, offset, 0])
                min_geo = np.vstack([min_geo, min_lattice_sites])
                max_geo = np.vstack([max_geo, max_lattice_sites])
            
        x_dim = np.max(max_geo[:,0]) - np.min(min_geo[:,0])
        y_dim = np.max(max_geo[:,1]) - np.min(min_geo[:,1])   
        x_dots = int(x_dim*dpa)
        y_dots = int(y_dim*dpa)
                
        return (x_dots,y_dots)
    
    
    def get_scale(self, struct, unit_cell=False):
        geo = struct.get_geo_array()
        ele = struct.elements
        temp_radii = np.array([self.vdw_radii[atomic_numbers[x]] for x in ele]
                              )[:,None]*self.vdw
        min_geo = geo - temp_radii
        max_geo = geo + temp_radii
        
        ### Account for position of the unit cell
        if unit_cell:
            lattice_sites = self.get_unit_cell_sites(struct)
            if len(lattice_sites) > 0:
                ### Account for radius of unit cell cylinders and for the 
                ### axis labels
                offset = self.unit_cell*1.01
                if len(self.lattice_vector_labels) > 0:
                    ### Fontsize given in points
                    ### There are 72 points per 1 tall
                    ### Therefore, treat as though each lattice site also has
                    ### sphere of radius given by the fontsize 
                    text_radius = self.lattice_vector_label_size / 24 * 0.5 
                    offset += text_radius
                
                min_lattice_sites = lattice_sites - np.array([
                                offset*0.4,self.unit_cell*1.01,0])
                max_lattice_sites = lattice_sites + np.array([offset*0.4, offset, 0])
                min_geo = np.vstack([min_geo, min_lattice_sites])
                max_geo = np.vstack([max_geo, max_lattice_sites])
            
        struct_extent = np.max(max_geo,axis=0) - np.min(min_geo,axis=0)
        
        ### Extent should only be used in the y-direction becuse it's what is parallel
        ### to the camera viewing angle
        # return 0.5*struct_extent[1]
        return 0.5*np.max(struct_extent)
    

    def get_extent(self, struct, unit_cell=False):
        geo = struct.get_geo_array()
        ele = struct.elements
        temp_radii = np.array([self.vdw_radii[atomic_numbers[x]] for x in ele]
                              )[:,None]*self.vdw
        min_geo = geo - temp_radii
        max_geo = geo + temp_radii
        
        extent = [
              np.min(min_geo[:,0]), np.max(max_geo[:,0]), 
              np.min(min_geo[:,1]), np.max(max_geo[:,1]),
              np.min(min_geo[:,2]), np.max(max_geo[:,2])]
        
        ### Account for position of the unit cell and its labels
        if unit_cell:
            lattice_sites = self.get_unit_cell_sites(struct)
            if len(lattice_sites) > 0:
                ### Account for radius of unit cell cylinders and for the 
                ### axis labels
                offset = self.unit_cell*1.01
                if len(self.lattice_vector_labels) > 0:
                    ### Fontsize given in points
                    ### There are 72 points per 1 tall
                    ### Therefore, treat as though each lattice site also has
                    ### sphere of radius given by the fontsize 
                    text_radius = self.lattice_vector_label_size / 24 * 0.5
                    offset += text_radius
                
                min_lattice_sites = lattice_sites - np.array([
                                offset*0.4,self.unit_cell*1.01,0])
                max_lattice_sites = lattice_sites + np.array([offset*0.4, offset, 0])
                min_geo = np.vstack([min_geo, min_lattice_sites])
                max_geo = np.vstack([max_geo, max_lattice_sites])
                
            
        extent = [
              np.min(min_geo[:,0]), np.max(max_geo[:,0]), 
              np.min(min_geo[:,1]), np.max(max_geo[:,1]),
              np.min(min_geo[:,2]), np.max(max_geo[:,2])]
        
        return extent
    
    
    def add_geometry(self, 
                     struct, 
                     plot_bonds=None,
                     periodic_bonds=None, 
                     vdw=None, 
                     colors=[]):
        """
        Adds atoms of given structure to the image.
        
        """
        if plot_bonds == None:
            plot_bonds = self.plot_bonds
        if vdw == None:
            vdw = self.vdw
        if periodic_bonds == None:
            periodic_bonds = self.periodic_bonds

        geo = struct.get_geo_array()
        ele = struct.elements
        
        for idx in range(len(geo)):
            if len(colors) > 0:
                color = colors[idx]
            else:
                color = []
            self.add_atom(geo[idx], ele[idx], vdw=vdw, color=color)
            
        if plot_bonds:
            if periodic_bonds:
                bonds = struct.get_bonds(**self.bonds_kw)
            else:
                temp_non_periodic = Structure.from_geo(struct.get_geo_array(),
                                                       struct.elements)
                bonds = temp_non_periodic.get_bonds(**self.bonds_kw)
                
            added = {}
            for idx1 in range(len(bonds)):
                bonded = bonds[idx1]
                for idx2 in bonded:
                    pos1 = geo[idx1]
                    pos2 = geo[idx2]
                    ele1 = ele[idx1]
                    ele2 = ele[idx2]
                    
                    temp_added = tuple(np.sort([idx1,idx2]))
                    if temp_added not in added:
                        if len(colors) > 0:
                            temp_colors = [colors[idx2], colors[idx1]]
                        else:
                            temp_colors = []
                        self.add_bond(pos1, pos2, ele1, ele2, color=temp_colors)
                        added[temp_added] = True
                    else:
                        continue
        
    
    def add_atom(self, pos, element, vdw=None, color=[], renderer=None):
        """
        Adds an atom to the image. 
        
        """
        if renderer == None:
            renderer = self.renderer
        if vdw == None:
            vdw = self.vdw
        
        if type(element) == str:
            temp_color = self.ele_colors[atomic_numbers[element]]
            radius = self.vdw_radii[atomic_numbers[element]]
        elif type(element) in [np.str, np.str_, np.string_]:
            temp_color = self.ele_colors[atomic_numbers[str(element)]]
            radius = self.vdw_radii[atomic_numbers[element]]
        elif type(element) == int or type(element) == float:
            temp_color = self.ele_colors[int(element)]
            radius = self.vdw_radii[int(element)]
        else:
            raise Exception("Type of {} cannot be used parsed as an element."
                .format(type(element)))
            
        if len(color) > 0:
            pass
        else:
            color = temp_color
            
        radius = radius * vdw
        
        # Create a sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(pos[0], pos[1], pos[2])
        sphereSource.SetRadius(radius)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(self.atom_resolution)
        sphereSource.SetThetaResolution(self.atom_resolution)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        actor.GetProperty().SetInterpolationToPBR()
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetMetallic(self.atom_metallic)
        actor.GetProperty().SetRoughness(self.atom_roughness)
        
        renderer.AddActor(actor)
        
    
    def add_bond(self, pos1, pos2, ele1, ele2, 
                 radius=0, color=(), resolution=None, renderer=None):
        """
        Adds a bond to the image. 
        
        """
        if radius == 0:
            radius = self.bonds
        if len(color) == 0:
            if len(self.bond_color) == 0:
                color = [self.ele_colors[atomic_numbers[ele2]],
                          self.ele_colors[atomic_numbers[ele1]]]
            else:
                color = [self.bond_color, self.bond_color]
        if resolution == None:
            resolution = self.bond_resolution
        if renderer == None:
            renderer = self.renderer
            
        #### Calculate angle between pos1 and pos2
        diff = np.array(pos1) - np.array(pos2)
        unit_diff = diff / np.linalg.norm(diff)
        
        try: 
            rot = R.align_vectors([unit_diff], [[0,1,0]])[0]
        except:
            #### This should only fail if the bond is between two atoms that
            #### lie directly on top of one another
            return
        euler_angles = rot.as_euler("xyz", degrees=True)
        
        #### Calculate translation
        trans_vec = [pos2 + diff / 4, pos2 + 3*diff / 4]
        
        #### Calculate the correct height
        height = np.linalg.norm(diff) / 2
        
        for i in range(2):
            transFilter = self.aligned_cylinder(radius, 
                                           height, 
                                           euler_angles, 
                                           trans_vec[i],
                                           resolution=resolution)
            
            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transFilter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.GetProperty().SetInterpolationToPBR()
            actor.GetProperty().SetMetallic(self.bond_metallic)
            actor.GetProperty().SetRoughness(self.bond_roughness)
            actor.GetProperty().SetColor(color[i])
            
            # dist = np.linalg.norm(trans_vec[i])
            # opacity = dist / 14
            # actor.GetProperty().SetOpacity(1-opacity)
            
            actor.SetMapper(mapper)
            renderer.AddActor(actor)
    
    
    def add_interaction(self, pos1, pos2, 
                        radius=0, length=0, gap=0, color=(), resolution=0,
                        renderer=None):
        """
        Arguments
        ---------
        length: float
            Units of length for the dotted cylinder line
        gap: float
            Ratio of the length value. Should be given as a value less than 1.0. 
        
        """
        if radius == 0:
            radius = self.interactions
        if length == 0:
            length = self.interaction_length
        if gap == 0:
            gap = self.interaction_gap
        if len(color) == 0:
            color = self.interaction_color
        if resolution == 0:
            resolution = self.interaction_resolution
        if renderer == None:
            renderer = self.renderer
        
        gap_length = length*gap
        total_length = length + gap_length
        
        #### Calculate angle between pos1 and pos2
        diff = np.array(pos1) - np.array(pos2)
        dist = np.linalg.norm(diff)
        unit_diff = diff / dist
        rot = R.align_vectors([unit_diff], [[0,1,0]])[0]
        euler_angles = rot.as_euler("xyz", degrees=True)
        
        #### Calculate all translations
        num_cylinders = int(dist / total_length)
        ## Offset from centers of atoms
        offset = dist - num_cylinders*total_length 
        ## Remaining length that is exactly divisible by cylinder total length
        remaining = dist - offset
        
        ## Get all starting position by adding half of offset to pos2
        trans_vec = []
        for i in range(num_cylinders):
            temp_trans = pos2 + unit_diff*(offset / 2) + unit_diff*total_length*i
            trans_vec.append(temp_trans)
        
        for i in range(len(trans_vec)):
            transFilter = self.aligned_cylinder(radius,
                                               length, 
                                               euler_angles, 
                                               trans_vec[i],
                                               resolution)
            
            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transFilter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.GetProperty().SetInterpolationToPBR()
            actor.GetProperty().SetMetallic(self.interaction_metallic)
            actor.GetProperty().SetRoughness(self.interaction_roughness)
            actor.GetProperty().SetColor(color)
            actor.SetMapper(mapper)
            renderer.AddActor(actor)
            
    
    def add_close_interactions(self, struct=None, num_close_interactions=-1):
        if num_close_interactions <= 0:
            num_close_interactions = self.num_close_interactions
            if num_close_interactions <= 0:
                return 
        if struct == None:
            struct = self.struct
            
        geo = struct.get_geo_array()
        ele = struct.elements
        vdw_array = np.array([vdw_radii[atomic_numbers[x]] for x in ele])
        vdw_matrix = vdw_array[:,None] + vdw_array[None,:]
        ### Use vdw matrix to calculate sr 
        all_dist = squareform(pdist(geo)) / vdw_matrix
        
        mol_idx_list = struct.get_molecule_idx()
        if len(mol_idx_list) == 1:
            ### Intermolecular contact implementation is for atoms that are 
            ### not bonded to one another
            bonds = struct.properties["bonds"]
            all_geo_idx = np.arange(0,len(geo)).astype(int)
            all_inter_idx = []
            for idx,bond_idx in enumerate(bonds):
                mask = np.ones((len(geo),)).astype(int)
                mask[idx] = 0
                mask[bond_idx] = 0
                
                ### Ssecond and third nearest neighbors should not be used
                for temp_bonded_idx in bond_idx:
                    temp_second_neigh_idx = bonds[temp_bonded_idx]
                    mask[temp_second_neigh_idx] = 0
                    for temp_bonded_idx_2 in temp_second_neigh_idx:
                        temp_third_neigh_idx = bonds[temp_bonded_idx_2]
                        mask[temp_third_neigh_idx] = 0
                
                temp_idx = np.hstack([np.zeros((np.sum(mask),1)).astype(int)+idx, 
                                      all_geo_idx[mask.astype(bool)][:,None]])
                if len(all_inter_idx) == 0:
                    all_inter_idx = temp_idx
                else:
                    all_inter_idx = np.vstack([all_inter_idx, temp_idx])
            
            ### Only get interactions going one direction
            all_inter_idx = np.unique(np.sort(all_inter_idx, axis=-1), axis=0)
            
        else:
            ### Should just build a list of pairwise moleucle indices that can 
            ### be taken from the all dist...
            ### Not hyper-efficient, but the overall number of molecules should
            ### always be relatively small and therefore it should be fine
            ### Will look like two forloops with np.meshgrid inside each for loop 
            ### where np.meshgrid indexes over the input mol_idx
            all_inter_idx = []
            for idx1,mol_idx1 in enumerate(mol_idx_list):
                range1 = np.arange(0,len(mol_idx1))
                for mol_idx2 in mol_idx_list[idx1+1:]:
                    range2 = np.arange(0,len(mol_idx2))
                    grid1,grid2 = np.meshgrid(range1, range2)
                    grid_idx = np.c_[grid1.ravel(), grid2.ravel()]
                    temp_dist_idx = np.hstack([mol_idx1[grid_idx[:,0]][:,None], 
                                               mol_idx2[grid_idx[:,1]][:,None]])
                    if len(all_inter_idx) == 0:
                        all_inter_idx = temp_dist_idx
                    else:
                        all_inter_idx = np.vstack([all_inter_idx, temp_dist_idx])
        
        all_inter_dist = all_dist[all_inter_idx[:,0], all_inter_idx[:,1]]
        sort_idx = np.argsort(all_inter_dist)
        keep_idx = sort_idx[0:num_close_interactions]
        geo_idx = all_inter_idx[keep_idx]
        
        for temp_geo_idx in geo_idx:
            pos1 = geo[temp_geo_idx[0]]
            pos2 = geo[temp_geo_idx[1]]
            self.add_interaction(pos1, pos2)
        
        return
        
    
    def aligned_cylinder(self, radius, height, euler_angles, translation, 
                         resolution):
        rot_x,rot_y,rot_z = euler_angles
        
        ### Cylinder between the two
        cylinderSource = vtk.vtkCylinderSource()
        cylinderSource.SetCenter(0.0, 0.0, 0.0)
        cylinderSource.SetRadius(radius)
        cylinderSource.SetHeight(height)
        cylinderSource.SetResolution(resolution)
        
        # Rotate cylinder to align with bond vector
        x_transform = vtk.vtkTransform()
        x_transform.RotateWXYZ(rot_x,1,0,0)
        x_transformFilter=vtk.vtkTransformPolyDataFilter()
        x_transformFilter.SetTransform(x_transform)
        x_transformFilter.SetInputConnection(cylinderSource.GetOutputPort())
        x_transformFilter.Update()
        
        y_transform = vtk.vtkTransform()
        y_transform.RotateWXYZ(rot_y,0,1,0)
        y_transformFilter=vtk.vtkTransformPolyDataFilter()
        y_transformFilter.SetTransform(y_transform)
        y_transformFilter.SetInputConnection(x_transformFilter.GetOutputPort())
        y_transformFilter.Update()
        
        z_transform = vtk.vtkTransform()
        z_transform.RotateWXYZ(rot_z,0,0,1)
        z_transformFilter=vtk.vtkTransformPolyDataFilter()
        z_transformFilter.SetTransform(z_transform)
        z_transformFilter.SetInputConnection(y_transformFilter.GetOutputPort())
        z_transformFilter.Update()
        
        trans = vtk.vtkTransform()
        trans.Translate(translation)
        transFilter = vtk.vtkTransformPolyDataFilter()
        transFilter.SetTransform(trans)
        transFilter.SetInputConnection(z_transformFilter.GetOutputPort())
        transFilter.Update()
        
        return transFilter
        
    
    def add_lighting(self, renderer):
        """
        Adds good default lighting settings for atomic systems. If the user 
        would like to change the letting settings, the easiest way is to 
        inherit from the Render class and overwrite the add_lighting method. 
        
        """
        light = vtk.vtkLight()
        light.SetColor(1.0, 1.0, 1.0)
        light.SetIntensity(2.5)
        light.SetPosition(0.5,0.25,1)
        light.SetLightTypeToCameraLight()
        # light.SetConeAngle(180)
        renderer.AddLight(light)
        
        light = vtk.vtkLight()
        light.SetColor(1.0, 1.0, 1.0)
        light.SetIntensity(1)
        light.SetPosition(-0.5,-0.25,1)
        light.SetLightTypeToCameraLight()
        # light.SetConeAngle(180)
        renderer.AddLight(light)
        
    
    def add_lattice(self, lv):
        self.add_unit_cell()
    
        
    def add_unit_cell(self, lv, renderer=None, surface=False):
        """
        Adds lattice vectors to the image. 
        
        """
        if renderer == None:
            renderer = self.renderer
        
        ### Obtain all corners of the unit cell
        frac_a,frac_b,frac_c = np.meshgrid(
            [0,1],[0,1],[0,1],
            indexing="ij",
            )
        all_frac = np.c_[frac_a.ravel(), frac_b.ravel(), frac_c.ravel()]
        all_points = np.dot(all_frac, lv)
        
        ### Hard-coded edges of unit cell wrt indices of the points
        edges = [[0,4], #a
                 [0,2], #b
                 [0,1], #c
                 
                 [1,3],
                 [1,5],
                 
                 [2,3],
                 [2,6],
                 
                 [4,5],
                 [4,6],
                 
                 [3,7],
                 [5,7],
                 [6,7]
                 ]
        
        for idx,entry in enumerate(edges):
            if not surface:
                if idx < 3:
                    color = self.color_abc[idx]
                else:
                    color = self.default_lattice_vector_color
            else:
                if idx == 2:
                    color = self.color_abc[2]
                elif idx == 3:
                    color = self.color_abc[1]
                elif idx == 4:
                    color = self.color_abc[0]
                else:
                    color = self.default_lattice_vector_color
            
            point1 = all_points[entry[0]]
            point2 = all_points[entry[1]]
            
            # self.renderer.AddActor(actor)
            height = np.linalg.norm(point2 - point1)
            diff = point2 - point1
            dist = np.linalg.norm(diff)
            unit_diff = diff / dist
            rot = R.align_vectors([unit_diff], [[0,1,0]])[0]
            euler_angles = rot.as_euler("xyz", degrees=True)
            translation = (point1 + point2) / 2
            
            transFilter = self.aligned_cylinder(
                                  self.unit_cell, 
                                  height, 
                                  euler_angles, 
                                  translation, 
                                  self.unit_cell_resolution)
            
            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transFilter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.GetProperty().SetInterpolationToPBR()
            actor.GetProperty().SetMetallic(self.unit_cell_metallic)
            actor.GetProperty().SetRoughness(self.unit_cell_roughness)
            actor.GetProperty().SetColor(color)
            
            actor.SetMapper(mapper)
            renderer.AddActor(actor)
            
            
    def add_unit_cell_labels(self, lv, renderer=None, surface=False):
        """
        Have to create a second renderer in order to have unit cell labels
        always rendered on top of everything else. 
        
        http://vtk.1045678.n5.nabble.com/Text-always-on-top-td5721132.html\
        
        Another good idea to save for later 
        https://github.com/holoviz/panel/issues/1413
        
        """
        self.tt = []
        if renderer == None:
            renderer = self.ucv_label_renderer
        
        for idx,loc in enumerate(lv):
            if surface == True:
                if idx == 0:
                    loc = np.dot(np.array([1,0,1])[None,:], lv)[0]
                elif idx == 1:
                    loc = np.dot(np.array([0,1,1])[None,:], lv)[0]
                
            textSource = vtk.vtkBillboardTextActor3D()
            textSource.SetInput(self.lattice_vector_labels[idx])
            textSource.SetPosition(loc)
            textSource.GetTextProperty().SetFontSize(self.lattice_vector_label_size)
            textSource.GetTextProperty().SetJustificationToCentered()
            textSource.GetTextProperty().SetColor(
                self.lattice_vector_label_color[0],
                self.lattice_vector_label_color[1],
                self.lattice_vector_label_color[2])
            textSource.GetTextProperty().BoldOn()
            
            ### Setup outline of text
            textSource.GetTextProperty().ShadowOn()
            textSource.GetTextProperty().SetShadowOffset(2, -2)
            
            renderer.AddActor(textSource)
            
            self.tt.append(textSource)
        
    
    def get_unit_cell_sites(self, struct):
        lv = struct.get_lattice_vectors()
        if len(lv) < 3:
            # raise Exception("Proper unit cell not detected")
            return []
        
        lv = np.vstack(struct.get_lattice_vectors())
        ### Obtain all corners of the unit cell
        frac_a,frac_b,frac_c = np.meshgrid(
            [0,1],[0,1],[0,1],
            indexing="ij",
            )
        all_frac = np.c_[frac_a.ravel(), frac_b.ravel(), frac_c.ravel()]
        
        ### Convert to cartesian
        all_points = np.dot(all_frac, lv)
        
        return all_points
    
    def get_edge_idx(self):
        ### Hard-coded edges of unit cell wrt indices of the points
        edges = [[0,4], #a
                 [0,2], #b
                 [0,1], #c
                 
                 [1,3],
                 [1,5],
                 
                 [2,3],
                 [2,6],
                 
                 [4,5],
                 [4,6],
                 
                 [3,7],
                 [5,7],
                 [6,7]
                 ]
        return edges


class OptimizedRender(Render):
    """
    The memory requirements of VTK using the method I have programmed above
    is too large. I am trying some tricks here in order to optimize the 
    performance so that large systems can be generated. 
    
    """
    def add_geometry(self, struct, plot_bonds=None, periodic_bonds=None, colors=[]):
        """
        Adds atoms of given structure to the image.
        
        """
        if plot_bonds == None:
            plot_bonds = self.plot_bonds
        if periodic_bonds == None:
            periodic_bonds = self.periodic_bonds

        geo = struct.get_geo_array()
        ele = struct.elements
        
        unique_ele = np.unique(ele)
        
        for temp_ele in unique_ele:
            ele_idx = np.where(ele == temp_ele)[0]
            temp_geo = geo[ele_idx]
            self.add_element(temp_geo, temp_ele)
        
        if plot_bonds:
            if periodic_bonds:
                bonds = struct.get_bonds(**self.bonds_kw)
            else:
                temp_non_periodic = Structure.from_geo(struct.get_geo_array(),
                                                       struct.elements)
                bonds = temp_non_periodic.get_bonds(**self.bonds_kw)
                
            added = {}
            for idx1 in range(len(bonds)):
                bonded = bonds[idx1]
                for idx2 in bonded:
                    pos1 = geo[idx1]
                    pos2 = geo[idx2]
                    ele1 = ele[idx1]
                    ele2 = ele[idx2]
                    
                    temp_added = tuple(np.sort([idx1,idx2]))
                    if temp_added not in added:
                        if len(colors) > 0:
                            temp_colors = [colors[idx2], colors[idx1]]
                        else:
                            temp_colors = []
                        self.add_bond(pos1, pos2, ele1, ele2, color=temp_colors)
                        added[temp_added] = True
                    else:
                        continue
                    
    
    def add_element(self, pos, element, vdw=None, renderer=None):
        """
        Add all atoms of a single element to the scene. This allows the use
        of vtkGlyph3D which just copies the same actor to multiple locations
        thus reducing memory use.
        
        """
        if renderer == None:
            renderer = self.renderer
        if vdw == None:
            vdw = self.vdw
        
        if type(element) == str:
            color = self.ele_colors[atomic_numbers[element]]
            radius = self.vdw_radii[atomic_numbers[element]]
        elif type(element) in [np.str, np.str_, np.string_]:
            color = self.ele_colors[atomic_numbers[str(element)]]
            radius = self.vdw_radii[atomic_numbers[element]]
        elif type(element) == int or type(element) == float:
            color = self.ele_colors[int(element)]
            radius = self.vdw_radii[int(element)]
        else:
            raise Exception("Type of {} cannot be used parsed as an element."
                .format(type(element)))
            
        radius = radius * vdw
        
        points = vtk.vtkPoints()
        for entry in pos:
            points.InsertNextPoint(entry[0],entry[1],entry[2])
        
        colors = vtk.vtkFloatArray()
        colors.SetName('Colors')
        for entry in pos:
            colors.InsertNextValue(1)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # pointsData = polydata.GetPointData()
        # _ = pointsData.SetScalars(colors)
        
         # Create a sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(radius)
        
        # Make the surface smooth.
        sphereSource.SetPhiResolution(self.atom_resolution)
        sphereSource.SetThetaResolution(self.atom_resolution)
        
        glyph3Dmapper = vtk.vtkGlyph3DMapper()
        glyph3Dmapper.SetSourceConnection(sphereSource.GetOutputPort())
        glyph3Dmapper.SetInputData(polydata)
        
        # glyph3Dmapper.SetColorModeToMapScalars()
        # glyph3Dmapper.SetColorModeToColorByVector()
        
        glyph3Dmapper.Update()
        
        # self.glyph = glyph3Dmapper
        
        # Visualize
        actor = vtk.vtkActor()
        actor.SetMapper(glyph3Dmapper)
        
        actor.GetProperty().SetInterpolationToPBR()
        actor.GetProperty().SetColor(color)
        # actor.GetProperty().SetOpacity(0.5)
        actor.GetProperty().SetMetallic(self.atom_metallic)
        actor.GetProperty().SetRoughness(self.atom_roughness)
        
        
        
        renderer.AddActor(actor)
    
    
    def add_interaction(self, pos1, pos2, 
                        radius=0, length=0, gap=0, color=(), resolution=0,
                        renderer=None):
        """
        Arguments
        ---------
        length: float
            Units of length for the dotted cylinder line
        gap: float
            Ratio of the length value. Should be given as a value less than 1.0. 
        
        """
        # interactions=0.075
        # interaction_length=0.2, 
        # interaction_gap=0.4, 
        # interaction_color=(0.09019607843137255, 0.7450980392156863, 0.8117647058823529), 
        # interaction_resolution=1000,
        
        if radius == 0:
            radius = self.interactions
        if length == 0:
            length = self.interaction_length
        if gap == 0:
            gap = self.interaction_gap
        if len(color) == 0:
            color = self.interaction_color
        if resolution == 0:
            resolution = self.interaction_resolution
        if renderer == None:
            renderer = self.renderer
        
        gap_length = length*gap
        total_length = length + gap_length
        
        #### Calculate angle between pos1 and pos2
        diff = np.array(pos1) - np.array(pos2)
        dist = np.linalg.norm(diff)
        unit_diff = diff / dist
        rot = R.align_vectors([unit_diff], [[0,1,0]])[0]
        euler_angles = rot.as_euler("xyz", degrees=True)
        
        #### Calculate all translations
        num_cylinders = int(dist / total_length)
        ## Offset from centers of atoms
        offset = dist - num_cylinders*total_length 
        ## Remaining length that is exactly divisible by cylinder total length
        remaining = dist - offset
        
        ## Get all starting position by adding half of offset to pos2
        trans_vec = []
        for i in range(num_cylinders):
            temp_trans = pos2 + unit_diff*(offset / 2) + unit_diff*total_length*i
            trans_vec.append(temp_trans)
        
        cylinderSource = self.aligned_cylinder(radius,
                                               length, 
                                               euler_angles, 
                                               np.array([0,0,0]),
                                               resolution)
        
        points = vtk.vtkPoints()
        for entry in trans_vec:
            points.InsertNextPoint(entry[0],entry[1],entry[2])
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        glyph3Dmapper = vtk.vtkGlyph3DMapper()
        glyph3Dmapper.SetSourceConnection(cylinderSource.GetOutputPort())
        glyph3Dmapper.SetInputData(polydata)
        glyph3Dmapper.Update()
        
        # Visualize
        actor = vtk.vtkActor()
        actor.SetMapper(glyph3Dmapper)
        actor.GetProperty().SetInterpolationToPBR()
        actor.GetProperty().SetMetallic(self.interaction_metallic)
        actor.GetProperty().SetRoughness(self.interaction_roughness)
        actor.GetProperty().SetColor(color)
        renderer.AddActor(actor)



    
if __name__ == "__main__":
    pass
