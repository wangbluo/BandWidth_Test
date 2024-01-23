import os
import sys
import time
import argparse
from abc import abstractmethod

import torch
import torch.distributed as dist
from typing import List
from prettytable import PrettyTable
    
def setup():
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    torch.cuda.set_device(local_rank)
    device_id = torch.cuda.current_device()
    print("Process running on GPU:", device_id)

"""    
    BandWidth Computation: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

    B: Bus bandWidth of the device
    S: Elements transformed when execute one collective op
    t: Time cost when execute one collective op 
    n: Ranks number

    AllReduce:
        B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)
    AllGather:
        B = S/t * (n-1)/n = algbw * (n-1)/n
    Broadcast:
        B = S/t = algbw 
    AllToAll:
        B = S/t * (n-1)/n = algbw * (n-1)/n
"""

class CollectiveOp:

    def __init__(self, world_size: int):
        self.world_size = world_size

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> None:
        pass

    @abstractmethod
    def ranks_factor(self):
        pass

class AllReduce(CollectiveOp):
    def __call__(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        dist.all_reduce(tensor)
    
    def ranks_factor(self):
        return 2 * (self.world_size - 1) / self.world_size


class AllGather(CollectiveOp):
    def __call__(self, tensor: torch.Tensor) -> None:
        sub_tensor_list = list(torch.chunk(tensor, self.world_size, dim=0))
        dist.all_gather(sub_tensor_list, sub_tensor_list[0]) 

    def ranks_factor(self):
        return (self.world_size - 1) / self.world_size 

class Broadcast(CollectiveOp):
    def __call__(self, tensor: torch.Tensor,src) -> None:
        tensor = tensor.contiguous()
        dist.broadcast(tensor,src)

    def ranks_factor(self):
        return 1

class AllToAll(CollectiveOp):
    def __call__(self, tensor: torch.Tensor) -> None:
        input_tensor_list = list(torch.chunk(tensor, self.world_size, dim=0))
        output_tensor_list = input_tensor_list[:]
        dist.all_to_all(output_tensor_list, input_tensor_list)

    def ranks_factor(self):
        return (self.world_size - 1) / self.world_size 

def avg_time(op, tensor, iters):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        if isinstance(op, Broadcast):
            if dist.get_rank == 0:
                src = dist.get_rank()
                op(tensor, src)
        else:
            op(tensor)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / iters

def test_bandwidth(op, tensor_list:List[int], iters: int = 10, warmup_iters: int = 5, dtype = torch.float, collective_op: str = 'allreduce'):
    #warm up
    mintensor = torch.rand(tensor_list[0], dtype=dtype, device='cuda')
    maxtensor = torch.rand(tensor_list[-1], dtype=dtype, device='cuda')
    avg_time(op, mintensor, warmup_iters)
    avg_time(op, maxtensor, warmup_iters)

    pretty_table = PrettyTable(['bytes(B)', 'tensorElements', 'dtype', 'collective op', 'time(ms)',
                        'algbw(GB/s)', 'busbw(GB/s)'], float_format='.3')
    busbw_total = 0
    for tensor_elements in tensor_list:
        tensor = torch.rand(tensor_elements, dtype=dtype, device='cuda')
        time = avg_time(op, tensor, iters)
        if isinstance(op, Broadcast):
            time_per_node = time
        else:
            #total time
            time = torch.tensor([time],device = 'cuda')
            dist.all_reduce(time)
            time_per_node = time/dist.get_world_size()
        if hasattr(time_per_node, 'item'):
            time_per_node = time_per_node.item()
        #bandwidth
        algbw = (tensor_elements * (torch.finfo(dtype).bits // 8))/ time_per_node
        busbw = algbw * op.ranks_factor()
        busbw_total += busbw
        pretty_table.add_row([(tensor_elements * (torch.finfo(dtype).bits // 8)), tensor_elements,
                              dtype, collective_op, time_per_node*1000, algbw / 1024**3, busbw / 1024**3])
    if dist.get_rank() == 0:
        avg_busbw = busbw_total / len(tensor_list)
        print(pretty_table)
        print(f'Average busbw: {avg_busbw/1024**3:.3f} GB/s')    

def get_op(ops: str, world_size: int = 1)-> CollectiveOp:
    if ops == 'allreduce':
        return AllReduce(world_size)
    elif ops == 'allgather':
        return AllGather(world_size)
    elif ops == 'broadcast':
        return Broadcast(world_size)
    elif ops == 'alltoall':
        return AllToAll(world_size)

def str_to_int(bytes: str) -> int:
    if bytes[-1] == 'B':
        bytes = bytes[:-1]
    elif bytes[-1] == 'K':
        return int(bytes[:-1]) * 1024
    elif bytes[-1] == 'M':
        return int(bytes[:-1]) * 1024**2
    elif bytes[-1] == 'G':
        return int(bytes[:-1]) * 1024**3
    return int(bytes)

def get_tensor_list(minbytes:str, maxbytes:str, step_bytes:str, step_factor:str, dtype = torch.float) -> List[int]:
    minbytes = str_to_int(minbytes) 
    maxbytes = str_to_int(maxbytes) 
    step_bytes = str_to_int(step_bytes)  
    bytes_list = []
    if step_factor == 1:
        bytes_list = list(range(minbytes, maxbytes + 1, step_bytes))
    else:
        while minbytes <= maxbytes:
            bytes_list.append(minbytes)
            minbytes *= step_factor

    # The list of elements number for testing tensors, arranged in ascending order. 
    # 32M bytes -> 32*1024**2 / (torch.finfo(float).bits // 8)
    bytes_list = sorted(bytes_list) 
    tensor_list = [bytes // (torch.finfo(dtype).bits // 8) for bytes in bytes_list]
    
    return tensor_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--collective_op', type=str, default='allreduce',
                        choices=['allreduce', 'allgather', 'broadcast', 'alltoall'])
    parser.add_argument('-min', '--minbytes', type=str, default='1M')
    parser.add_argument('-max', '--maxbytes', type=str, default='64M')
    parser.add_argument('-b', '--stepbytes', type=str, default='1M')
    parser.add_argument('-f', '--stepfactor', type=int, default=2)
    parser.add_argument('-w', '--warmup_iters', type=int, default=5) 
    parser.add_argument('-i', '--iters', type=int, default=20)
    parser.add_argument('-d', '--dtype', type=str,default='float', 
                        choices=['fp16', 'bf16', 'float'])
    args = parser.parse_args()

    #setup
    if torch.cuda.is_available():
        print("CUDA is available. GPU can be used.")
    else:
        print("CUDA is not available. Using CPU.")

    setup()

    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'float':
        dtype = torch.float
        
    world_size = int(os.getenv("WORLD_SIZE", 1))
    collective_op = get_op(args.collective_op, world_size) 
    tensor_list = get_tensor_list(args.minbytes, args.maxbytes, args.stepbytes, args.stepfactor, dtype)

    test_bandwidth(collective_op, tensor_list, args.iters, args.warmup_iters, dtype, args.collective_op)
