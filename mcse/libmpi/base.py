# -*- coding: utf-8 -*-


import json,datetime,os,sys
import numpy as np
import pandas as pd

from mcse import Structure
from mcse.io import write

from mpi4py import MPI


class _NaiveParallel():
    """
    Base class for naively parallel operations performed through the file 
    system. 
    
    
    """
    def __init__(self, struct_dir="", comm=None):
        self.struct_dir = struct_dir
        
        if comm == None:
            comm = MPI.COMM_WORLD
        
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
    
    def get_files(self, path=""):
        """
        Get the idx range for the rank by respecting the size of the 
        communicator and the number of files to be parallelized over. 
        Then return the filename corresponding to these idx and return.
        
        Arguments
        ---------
        path: str
            Path to get files for. If the length is zero, then the default is 
            to use the variable self.struct_path
            
        """
        if len(path) == 0:
            path = self.struct_path
            
        ## Get all files in the target directory
        file_list = np.array(os.listdir(path))
        
        ## Split files for each rank into most even 
        ## division of work possible
        my_files = file_list[self.rank::self.size]
        my_files = [os.path.join(path,x) for x in my_files]
        
        return my_files


    def get_list(self, arg_list):
        """
        Returns the split of a list for the current rank.

        """
        return arg_list[self.rank::self.size]
    

class _JobParallel(_NaiveParallel):
    """
    Parallelizes naively over a list of jobs to execute. At the end, gather all
    results onto rank 0.

    Arguments
    ---------
    job_list: list
        List made up of two components, the function to call and the arguments
        to the function as a dictionary. Two formats are accepted.
           [(callable_function, kwargs)]: Just a callable function and all the
               arguments for that function.
           [(callable_function, Structure, kwargs)]: Callable function with a
               mcse.Structure and the key word arguments for the callable
               function.
    write_mode: str
        Changes the way that the results are written. There are two options:
            "file": Results are written to a file, whose filename is specified
                    when _JobParallel.write is called
            "dict": Results are written as a structure dictionary whose
                    directory name is specified when _JobParallel.write is
                    called. Note that overwrite is assumed to be True in
                    this case.
    name: str
        If you would like to give a name to the JobParallel calculation.
        This may be used when writing results if no name is supplied
        by hand.

    """
    def __init__(self, job_list=[], write_mode="", name="", comm=None,
                 verbose=False):
        if comm == None:
            comm = MPI.COMM_WORLD

        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.verbose = verbose
        self.write_mode = write_mode
        self.name = name

        self.job_list = job_list
        
        if len(job_list) > 0:
            self.my_list = self.get_list(self.job_list)
        else:
            self.my_list = []

        self.all_results = None
        
        self.allowed_write_modes = ["file", "dict"] 
        if len(write_mode) > 0:
            if write_mode not in self.allowed_write_modes:
                raise Exception("JobParallel write_mode was not recognized. "+
                    "Please use one of {} in place of {}."
                    .format(write_mode, self.allowed_write_modes))
                
    
    def send_jobs(self):
        """
        Communicates and equal number of jobs to all ranks from rank 0. This is 
        a more advanced use case, but still works in generally all general 
        cases. 
        
        This has been verified to create the same lists on all ranks as the 
        traditional approach. 
        
        Results from preparing a job list by reading in 17,770 structure using
        6 MPI ranks. The results are a little noisy, but there is significant 
        improvements to be gained. I would expect this would improve even 
        further with more MPI ranks and a slow file system. 
        Traditional: 50.46 s, 49.73 s, 42.59 s
#        send_jobs: 35.04 s, 34.12 s, 30.24 s
        
        """
        if self.rank == 0:
            if len(self.job_list) == 0:
                raise Exception("No jobs to send.")
        
        if self.rank == 0:
            ## Prepare job lists to communicate
            send_job_list = []
            for rank in range(self.size):
                send_job_list.append(
                    self.job_list[rank::self.size]
                    )
            
            self.my_list = send_job_list[0]
            
            for dest,temp_job_list in enumerate(send_job_list[1:]):
                ## Skip rank 0
                dest = dest+1
                self.comm.send(temp_job_list, dest=dest)
        else:
            self.my_list = self.comm.recv(source=0)
            

    def calc(self):
        """
        Main calculation routine

        """
        ## This is here if the user has updated job_list after instantiation
        if len(self.my_list) == 0:
            self.my_list = self.get_list(self.job_list)
            
        result_list = []
        total = len(self.my_list)
        for entry in self.my_list:
            func = entry[0]
            if len(entry) == 3:
                struct = entry[1]
                kwargs = entry[2]
                result = func(struct, **kwargs)
            elif len(entry) == 2:
                kwargs = entry[1]
                result = func(**kwargs)
            else:
                raise Exception("Job list could not be parsed: {}."
                                .format(entry))
                
            result_list.append([entry, result])
            
            if self.verbose == True:
                print("{}: {}={}".format(total, entry, result),
                      flush=True)
            
            total -= 1
        
        self.comm.Barrier()
        ### Decide if slow algorithm is needed based on communication size
        results_list_bytes = sys.getsizeof(result_list)
        all_results_size = self.comm.gather(results_list_bytes, root=0)
        if self.rank == 0:
            all_results_size = sum(all_results_size)
            ### This is a low limit, but memory determination is not an exact 
            ### science in Python, still avoids potentiall poor performance for 
            ### small messages
            if all_results_size > 1e6:
                slow_algorithm = True
            else:
                slow_algorithm = False
            #print("USING SLOW ALGORITH: {},{}".format(slow_algorithm, all_results_size), 
            #      flush=True)
        else:
            slow_algorithm = False
         
        slow_algorithm = self.comm.bcast(slow_algorithm, root=0)
        if not slow_algorithm:
            self.all_results = self.comm.gather(result_list, root=0)
            if self.rank == 0:
                self.all_results = sum(self.all_results, [])
            return self.all_results
        else:
            if self.rank == 0:
                self.all_results = result_list
                ### Get results individually from all other ranks
                for rank in range(self.size-1):
                    self.all_results += self.comm.recv(
                        source = rank+1
                        )
            else:
                self.all_results = []
                self.comm.send(result_list, dest=0)

            return self.all_results


    def write(self, filename="", mode="file"):
        """
        Write just from rank 0

        Arguments
        ---------
        filename: str
            Filename when mode is "file"
            Directory name when mode is "dict"
        mode: str
            Described in main docstring. If a mode is provided
            in the instantiation, then this will overwrite
            any setting used here.

        """
        if self.rank != 0:
            return

        if len(self.write_mode) > 0:
            mode = self.write_mode

        if self.all_results == None:
            raise Exception("Called JobParallel write without running "+
                "calculations first.")

        if len(filename) == 0:
            temp_date = datetime.datetime.now()
            temp_date = temp_date.strftime(
                    "%Y-%m-%d_%H-%M-%S"
                    )

            if len(self.name) != 0:
                basename = self.name
            else:
                basename = "JobParallel"

            if mode ==  "file":
                filename = "{}_Results_{}.json".format(
                        basename,
                        temp_date
                        )
            else:
                filename = "{}_Results_{}".format(
                        basename,
                        temp_date)


        if mode == "file":
            ## This will make all_results serializable through recursive
            ## calls
            self.all_results = self.json_serialize(self.all_results)
            with open(filename, "w") as f:
                f.write(json.dumps(self.all_results))
        elif mode == "dict":
            write_dict = {}
            for entry in self.all_results:
                settings = entry[0]
                if len(settings) == 3:
                    struct = settings[1]
                elif len(settings) == 2:
                    kwargs = settings[-1]
                    if "struct" in kwargs:
                        struct = kwargs["struct"]
                    else:
                        raise Exception("Structure could not be found in "+
                          "settings. Please contact Manny for help.")

                result = entry[-1]
                result = self.json_serialize(result)

                if len(self.name) > 0:
                    if self.name in struct.properties:
                        prop_name = "{}_JobParallel_Result".format(self.name)
                    else:
                        prop_name = self.name
                    struct.properties[prop_name] = result
                else:
                    ## This should really not be used, but it will be there
                    ## for mistakes. Ideally, the func using in the job_list
                    ## will already modify the structure properties
                    struct.properties["Temp_JobParallel_Result"] = result

                write_dict[struct.struct_id] = struct

            write(filename, write_dict, overwrite=True)

        else:
            raise Exception("Write mode was not recognized: {}".format(mode)+
                    " Please use one of {}".format(self.allowed_write_modes))


    def json_serialize(self, obj):

        ## Numpy array type
        if type(obj) == np.array or type(obj) == np.ndarray:
            obj = obj.tolist()
            return obj

        ## DataFrame type
        if type(obj) == pd.DataFrame:
            obj = obj.to_dict()
            return obj

        if type(obj) == str:
            return obj

        ## If list, need to call this function again
        if type(obj) == list:
            for idx,entry in enumerate(obj):
                obj[idx] = self.json_serialize(entry)
            return obj

        ## Turn Structure into just struct_id
        if type(obj) == Structure:
            obj = obj.struct_id
            return obj

        if type(obj) == int:
            return obj

        if type(obj) == float:
            return obj

        ## Otherwise have to return as a string
        return str(obj)
            

class _Queue():
    """
    Queue type parallelization. Queue is held on rank 0. Jobs are distributed 
    to other ranks. As a rank finishes a job, it asks rank 0 for another job 
    from the top of the queue. Thus, there is a trade-off with a naively 
    parallel approach where there is more communication but more efficient 
    use of the available resources. 
    
    """
    def __init__(self, comm=None):
    
        if comm == None:
            comm = MPI.COMM_WORLD
        
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        
        if self.size <= 1:
            raise Exception("Cannot use Queue with only 1 MPI process.")
    
    
    def simple_queue(self, queue):
        """
        The simplest queue possible. The input is just a list of structures 
        that are sent one by one to processes. 
        
        """
        self.queue = queue
        ### Rank 0 will sit here until the queue is finished.
        if self.rank == 0:
            while True:
                item = queue.pop()
                status = MPI.Status()
                source = self.comm.recv(source=MPI.ANY_SOURCE,
                                        status=status)
                
                ## Using queue length plus 1 here because I never want to send
                ## a queue length of zero until the ending message.
                message = {"item": item, "queue_length": len(queue)+1}
                self.comm.send(message, dest=status.Get_source())
                
                if len(queue) == 0:
                    break
            
            ### Send end message to everyone. Using size-1 because the 
            ### master ranks hould not be included. Messages sent sequentially
            ### to all ranks. There's certainly the possibility to use 
            ### non-blocking communciation here for improved performance. 
            for idx in range(self.size-1):
                self.comm.send({"item": None, "queue_length": 0}, 
                               dest=idx+1)
        
    
    def pop(self):
        """
        Pop an item from the top of the queue. This should be called everytime
        that a rank needs a new item. 
        
        """
        if self.rank == 0:
            #### If rank zero has called this, assume it is because the 
            #### queue has already ended
            return {"item": None, "queue_length": 0}
        
        item = (None, 0)
        self.comm.send((True), dest=0)
        message = self.comm.recv(source=0)
        
        return message
        




