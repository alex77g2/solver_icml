# NNBenchmark.py (2023)
# container class only

import torch
# from math import inf as math_inf
from torch.utils.data import DataLoader, Dataset
from Cos2MinTorchFunctionOptimizer import SelfConstOptimTorch
from UniversalTorchModelTrainer import train
from os import path, makedirs
from datetime import datetime as datetime2
from fastparquet import write as fastp_write
import pandas as pd # optional: disabled below
from mmh3 import hash128
from time import time as time_time # shrink function table
from timm.data.loader import MultiEpochsDataLoader

def GetTorchInfo(f) -> None:
    if (f is None) or (f.closed) or (f.tell() > 9): return
    f.write("#Torch:%s,%s,%s+%s\n" % (torch.__version__,torch.get_default_dtype(),torch.cuda.is_available(),torch.cuda.is_initialized()))
    return

def run_benchmarks(train_data:Dataset, test_data:Dataset, 
                   model_class, model_hyperparams, optimizers, loss_func, task_name:str = "",
                   runs:int = 1, max_epochs:int = 10, target_loss = float('inf'), batch_sizes = None,
                   different_batch_sets:int = 1, device = torch.device("cpu"), seed0:int = 0) -> None:

    assert(callable(model_class)), "model_class not model()"
    initilizations = [model_class(*model_hyperparams).state_dict() for i in range(1, runs+1)]
    initilizations = [
        (hash128(str(init)).to_bytes(16, 'big').hex(), init) 
        for init in initilizations ]
    # for cid, checkpoints in initilizations:
    #    torch.save(checkpoints, f"{cid}.pt") # why we save them?

    batch_seeds = [torch.randint(1, 1000, (1,)).item() for i in range(0, different_batch_sets)] # only 1 number
    if (seed0 > 1000) and (different_batch_sets <= 1): batch_seeds = [ seed0 ] # optional

    if not path.exists("benchmarks"):
        makedirs("benchmarks")

    counter: int = 0
    log = open("history.txt", "a")
    GetTorchInfo(log)

    for o_name, (opt_class, opt_params) in optimizers.items():

        filename: str = "benchmarks/" + datetime2.now().strftime('%Y-%m-%d_%H-%M-%S')+"_"+o_name+"_"+task_name+"_seed"+str(seed0)+".parquet"

        for opt_param in opt_params:
            if (len(opt_param) > 2) and (opt_param[2] is None):
                opt_param = list(opt_param)
                opt_param[2] = loss_func
                opt_param = tuple(opt_param)
            
            for sid, state_dict in initilizations:

                for seed in batch_seeds:

                    for batch_size in batch_sizes:
                
                        model = model_class(*model_hyperparams).to(device)
                        model.load_state_dict(state_dict)
                        pdim: int = sum(p.numel() for p in model.parameters())
                        
                        if path.exists("/etc/"): # Windows not supported (01.11.2023)
                            s: float = time_time()
                            torch.compile(model) 
                            print("Compile time %.3f sec, n=%d" % (time_time() - s, pdim))
                        else:
                            print("Compile <Windows=Off>, n=%d" % (pdim))
                        
                        log.write("#RUN,%d,%s,%d,%d,%d,%s,\n" % (counter,o_name,batch_size,pdim,seed0,datetime2.now().strftime('%Y-%m-%d_%H-%M-%S')))
                        log.flush()
                        print(o_name, opt_param, batch_size, seed, sid)
                    
                        if opt_class is SelfConstOptimTorch: # hack as long as optimizer does not fit generall torch style
                            optimizer = opt_class(model.parameters(), model, *opt_param)
                        else:
                            optimizer = opt_class(model.parameters(), *opt_param)

                        torch.manual_seed(seed) # set seed before each run, to make sure same batches are used

                        data_kwargs = {'batch_size': batch_size}
                        # num_workers = Zahl der Threads die die CPU nutzt um die Daten nachzuladen, empirisch gesetzt, müsste ggf. angepasst werden !!!
                        if next(model.parameters()).is_cuda: # nm=12 suggested by torch, was 25 (6 is faster than 12)
                            cuda_kwargs =   {'num_workers': 4,# optimal value is setting / batch dependend, should be a parameter
                                            'pin_memory': True,
                                            #'prefetch': False, # test
                                            'persistent_workers' : True
                                            #'shuffle': True
                                            }
                            data_kwargs.update(cuda_kwargs)

                        # kommt vom Paket Timm -> DefaultLoader liest von Festplatte ggf., deswegen läuft es so schneller
                        train_dataloader:DataLoader = MultiEpochsDataLoader(train_data, **data_kwargs, shuffle=True)# here shuffle (time+5%)
                        test_dataloader:DataLoader = None if test_data is None else MultiEpochsDataLoader(test_data, batch_size=min(1024, batch_size*2)) # big-batches
                        # train_hbs_data:DataLoader = None # MultiEpochsDataLoader(train_data, batch_size=1<<11, shuffle=False) # big-batches (full batch) or None
                        #print("HASH1:", hash(test_dataloader.dataset), batch_size)

                        # train_dataloader = DataLoader(train_data, **data_kwargs)
                        # test_dataloader = DataLoader(test_data, **data_kwargs) if test_data is not None else None

                        start_time : float = time_time()

                        # Kern-Training
                        losses, batches, epochs, types, steps, f_calls, g_calls = train(train_dataloader, test_dataloader, model, loss_func,
                                optimizer, max_epochs, target_loss, batch_size=batch_size, device=device, log=log)

                        runtime : float = time_time() - start_time
                        print("Training runtime: %.6f seconds (ep=%d)" % (runtime, max_epochs))
                        #exit() #break here for not saving
                        counter += 1

                        if False: # not needed?
                            data = {
                            'run_id' : counter*len(losses),
                            'loss': losses, # caution: only list[floats], not list[tensor]
                            'batch_idx': batches,
                            'epoch': epochs,
                            'type': types,
                            'step': steps,
                            'forward_pass': f_calls,
                            'backward_pass': g_calls,
                            'theta_0' : [str(sid)] * len(losses),
                            'batching_seed' : [seed] * len(losses),
                            'optimizer_params' : [', '.join(map(str, opt_param))] * len(losses),
                            'optimizer' : [o_name] * len(losses),
                            'batch_size' : [batch_size] * len(losses),
                            'my_seed' : seed0 # added 22.09.2023
                            }

                            df = pd.DataFrame(data)
                            #print(type(df)) # <class 'pandas.core.frame.DataFrame'>

                            # Check if the file exists
                            file_exists: bool = path.isfile(filename)
                            # Write to the Parquet file
                            # extrem komprimiertes Format for CSV-data, nicht optimal konfiguriert aktuell, ggf. Nacharbeit ??
                        # wenn fehler kommt, dass zuviele Dateien offen sind müssen die Dateien auch noch geschlossen werden
                            fastp_write(filename, df, append=file_exists, open_with=open)
                    
                            print(f"Saved results to file: {filename}")                    

                        # with open("history.txt", "a") as log:
                        log.write("#END,%d,%d,%.3f,%s,\n" % (counter-1, max_epochs, runtime, filename))
                        
    log.close()
    return

# EoF.
