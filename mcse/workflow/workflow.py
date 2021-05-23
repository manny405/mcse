# -*- coding: utf-8 -*-

import os,json

from mcse import StructDict
from mcse.io import read,write,check_dir
from mcse.workflow.settings import get_settings,from_settings_file,\
                                    check_settings_file

from mpi4py import MPI


class Workflow():
    """
    Beginning of the workflow class that will handle the execution of mcse
    Drivers with different parameters such as dependancies. Also handlese 
    the management of files through the workflow. 
    
    Workflow should have two modes of execution:
        1. Operate/Manipulate the MPI ranks directly
        2. Workflow manager executes on the headnode while it spawns the 
            executables on the compute nodes it has been allocated. Therefore,
            in an ideal world need to have srun, mpirun, aprun support. 
            And Lassen. 
            
    20201221 Next Steps:
        1. Finish implementing the settings writing and checking algorithm
        2. Implement health monitoring method
        3. Unify IO models used throughout mcse such as h5py, structure files, 
            and directories of FHI-aims calculations
        5. Unify workflow model with libmpi models 
            
    Arguments
    ---------
    input_folder: str
        Path to the folder of Structures to use as the input for the workflow. 
    workflow_folder: str
        Where workflow should take place. 
    drivers: iterable
        List of any number of drivers can be passed in to define the workflow. 
        Operation will move from one driver to the next. 
    driver_folder_names: iterable
        Iterable of names of folders which each driver will operate in.
    driver_libmpi: iterable
        Iterable containing libmpi objects that should be initialized and used
        with each driver at each step in the workflow. 
    
    """
    def __init__(self, 
                 input_folder="Structures",
                 workflow_folder="Workflow",
                 drivers=[],
                 driver_folder_names=[],
                 driver_libmpi=[],
                 comm=None):
        if comm == None:
            comm = MPI.COMM_WORLD
        
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.input_folder = input_folder
        self.workflow_folder = workflow_folder
        check_dir(self.workflow_folder)
        
        self.drivers = drivers
        self.driver_folder_names = driver_folder_names
        self.driver_folders = []
        
        if len(self.driver_folder_names) == 0:
            for driver_idx,entry in enumerate(self.drivers):
                driver_idx += 1
                temp_name = type(entry).__name__
                default_driver_folder = "{}_{}".format(driver_idx,temp_name)
                folder_path = os.path.join(self.workflow_folder, 
                                           default_driver_folder)
                self.driver_folders.append(folder_path)
                self.driver_folder_names.append(default_driver_folder)
        else:
            if len(driver_folder_names) != len(self.drivers):
                raise Exception("The number of driver_folder_names must be "+
                    "equal to 0 or the number of drivers.")
                
            for entry in self.driver_folder_names:
                folder_path = os.path.join(self.workflow_folder, entry)
                self.driver_folders.append(folder_path)
        
        ### Check settings before creating workflow driver folders
        self.workflow_settings_path = os.path.join(self.workflow_folder, 
                                                   "settings.json")
        check_settings_file(self, self.workflow_settings_path)
        
        ### Create folder for each workflow driver
        for entry in self.driver_folders:
            check_dir(entry)
            
    
    def calc(self):
        """
        Calc also needs to combine the ideas of including libmpi modules 
        inside and not just Drivers. Although, this is easy in principal. Just
        check if the Driver step comes from libmpi or comes from BaseDriver_.
        Behavior might have to be a bit different for each one. 
        
        Steps of the workflow calculation will be as follows:
            1. 
        
        """
        raise Exception("Workflow calculation not implemented yet")
    
    
    @classmethod
    def from_folder(self, workflow_folder):
        """
        Loads in the Workflow class using the information from the 
        workflow_folder that is presented to it. Useful for the purpose of 
        reproducibility. 
        
        """
        
        
    
    
class WorkflowDB(Workflow):
    """
    Same as workflow, but operates in conjunction with a database. 
    
    """

class Worker(Workflow):
    """
    Worker is the same as the workflow except that at the end/beginning, it
    idles and waits for structures to be added to its starting directory. Once
    a new structure is added, it begins the workflow for that structure. 
    
    """
    
    
    
class WorkerSocket(Workflow):
    """
    Worker that listens over a SOCKS socket for a new Structure to begin
    calculating. Once calculated, sends results back. In addition, can listen
    for commands that can ask about the system status. Can also automatically
    update the library?
    
    """
    
    
    
    