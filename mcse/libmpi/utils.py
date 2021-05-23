# -*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI


def split_by_cost(cost_list, 
                  comm=None, 
                  return_work=False,
                  return_all=False):
    """
    
    Split up a list of jobs by their cost to be even across all the ranks
    in the communicator. This is important when trying to parallelize naively
    over a list of jobs, but each job can require a widely different number
    of computations (the cost). If this is done naively by index, this can 
    create load balancing issues. If you can compute an approximate cost of 
    each job in the list beforehand, this function can be used to achieve
    better load balancing. 
    
    This function can be called by all ranks because the way the list 
    is split is deterministic. 
    This function can also be called only by rank 0 if the return_all option
    is set to True. 
    
    Arguments
    ---------
    cost_list: iterable
        List of integer or float costs for each job.
    job_list: iterable
        Job list corresponding to the cost list. 
    comm: MPI.COMM
        MPI communicator
    return_work: bool
        If the work assigned to the rank should be return as well
    return_all: bool
        If the entire work_idx and work_sum lists should be returned. This 
        should be used if only rank 0 will call this routine. 
    
    """
    if comm == None:
        comm = MPI.COMM_WORLD
    
    size = comm.Get_size()
    rank = comm.Get_rank()

    ### Total cost of job_list
    total = np.sum(cost_list) 
    ### Ideal work for each rank
    max_work = (total / size)*1.01
    
    ### Preparing indices that each rank will use
    work_idx = [[] for x in range(size)]
    work_sum = [0 for x in range(size)]
    current_worker = 0
    withheld_idx_list = []
    withheld_value_list = []
    for idx,value in enumerate(cost_list):
        ## Decide whether to withhold value
        if work_sum[current_worker] + value > max_work*1.05:
            withheld_idx_list.append(idx)
            withheld_value_list.append(value)
            continue
        
        work_idx[current_worker].append(idx)
        work_sum[current_worker] += value
        if work_sum[current_worker] > max_work:
            current_worker += 1
    
    withheld_idx_list = np.array(withheld_idx_list)
    withheld_value_list = np.array(withheld_value_list)
    withheld_sort_idx = np.argsort(withheld_idx_list)
    withheld_idx_list = withheld_idx_list[withheld_sort_idx]
    withheld_value_list =  withheld_value_list[withheld_sort_idx]
    for idx,withheld_idx in enumerate(withheld_idx_list):
        min_idx = np.argmin(work_sum)
        work_sum[min_idx] += withheld_value_list[idx]
        work_idx[min_idx].append(withheld_idx)
        
    my_list = work_idx[rank]
    
    if not return_all:
        if not return_work:
            return my_list
        else:
            return my_list,work_sum[rank]
    else:
        if not return_work:
            return work_idx
        else:
            return work_idx,work_sum


def pair_idx(rows, comm=None):
    """
    
    Parallelized method for getting pairwise indices such that all ranks 
    have an equal number and all indices are used. Beginning of algorithm here
    but still needs to be finished. 
    
    """
    raise Exception("Not implemented")
    
    if comm == None:
        comm = MPI.COMM_WORLD
    
    total = comb(rows,2,exact=True)
    size = comm.Get_size()
    
    size = 1000
    
    print(total / size)
    
    target = total / size
    
    current_row = 0
    calc_list = []
    row_list = [[] for x in range(size)]
    for rank in range(size):
        row_list[rank].append(current_row)
        
        current_calcs = 0
        
        for value in range(current_row, rows):
            current_calcs += value
            if current_calcs > target:
                if rank == size-1:
                    pass
                else:
                    break
        
        calc_list.append(current_calcs)
        row_list[rank].append(value)
        current_row = value
    
    return row_list,calc_list


def tournament_communication(comm, 
                             comm_fn=lambda x,y: None,
                             comm_kw={}):
    """
    This is useful for the development of parallelized O(N) duplicate check
    functions. The problem with such functions is that the operation of 
    checking if a set of parameters/descriptor has been observed previously 
    requires there be a master set of observed values to compare against and
    add to. This means that it is not naively parallel. In order to achieve
    decent scaling for O(N) duplicate checks, this method has been implemented.
    This method will combine the results from ranks in a tournament braket 
    style such that in the end, the master list will be on rank 0. This 
    achieves better scaling because in the beginning, all ranks compare their
    lists to another rank adding unique values to one of the ranks. This rank
    moves forward to another comparions with another rank that has completed
    its comparisons as well. This continues until all results are on the master
    rank. 
    
    comm_fn API: comm_fn(comm, (rank1, rank2), **kwargs)
    
    """
    ### Step 1 is to build the tournament braket
    size = comm.Get_size()
    rank = comm.Get_rank()
    rank_list = np.arange(0,size)
    
    tournament = []
    temp_tournament = []
    for idx,value in enumerate(rank_list[::2]):
        value2 = value+1
        temp_tournament.append((value,value2))
    tournament.append(temp_tournament)
    
    if size <= 1:
        tournament = [(0,)]
    
    prev_tournament = tournament[0]
    while len(prev_tournament) != 1:
        temp_tournament = []
        for idx,entry in enumerate(prev_tournament[::2]):
            next_idx = idx*2+1
            keep_rank1 = min(entry)
            if (next_idx+1) > len(prev_tournament):
                temp_tournament.append((keep_rank1,))
            else:
                keep_rank2 = min(prev_tournament[next_idx])
                temp_tournament.append((keep_rank1, keep_rank2))
        tournament.append(temp_tournament)
        prev_tournament = tournament[-1]
    if len(tournament) > 1:
        tournament.append([(0,)])
    
    if tournament == [(0,)]:
        return 
    
    idx = 0
    for braket in tournament:
        if rank == 0:
            print("Round {} of {}".format(idx, len(tournament)), flush=True)
            idx += 1

        # ### Rank loop is here to emulate parallel execution
        # for rank in rank_list:
        found = False
        for entry in braket:
            if rank in entry:
                found = True
                break

        if found:
            comm_fn(comm, entry, **comm_kw)
        # if found:
        #     print("{}: {}".format(rank, entry))

    return tournament



if __name__ == "__main__":
    pass

    # work_list = np.arange(0,10000)
    # my_list,my_work = split_by_cost(work_list, return_work=True)
    
    # print("{}".format(my_work))
    
    