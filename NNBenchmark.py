# NNBenchmark.py (2024)
# container class only

import torch
from torch.utils.data import Dataset # DataLoader, random_split
# from fastparquet import write as fastp_write
# import pandas as pd # optional: disabled below
from time import time as time_time # shrink function table

def GetTorchInfo(f) -> None:
    "torch env info"
    if (f is None) or (f.closed) or (f.tell() > 9): return
    import platform
    f.write("#Torch:%s,%s,%s,%s+%s\n" %
        (torch.__version__, platform.node(), torch.get_default_dtype(), torch.cuda.is_available(), torch.cuda.is_initialized()))
    return

#def worker_init_fn(worker_id): # unused
#    from os import cpu_count
#    os.sched_setaffinity(0, range(cpu_count()))
#    return

def append_dropout(model, rate:float=0.2) -> None:
    "add dropout to model"
    # https://discuss.pytorch.org/t/where-and-how-to-add-dropout-in-resnet18/12869/3
    if (rate <= 0.0): return
    import torch.nn as nn
    
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        # print(type(module))
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)
            assert(0), "never reached"
    # append_dropout(model)
    return

def prepare(dataset, rank:int, world_size:int, batch_size:int=32, pin_memory:bool=False, num_workers:int=0):
    "unused (DistributedSampler)"
    # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    from torch.utils.data.distributed import DistributedSampler
    # assert(torch.cuda.device_count() > 1), "Multi-GPU"
    # dataset = Your_Dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    return DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    # return dataloader

def GetDataLoaders(batch_size:int, train_data:Dataset, test_data:Dataset, num_classes:int, is_cuda:bool) -> tuple:
    "return 3x MultiEpochsDataLoader"
    from torch.utils.data import DataLoader, random_split
    from timm.data.loader import MultiEpochsDataLoader
    assert(batch_size >= 1), "positive int"
    # print("Dataset.classes:", len(train_data.classes), len(test_data.classes))

    assert(len(train_data) > 1), "no samples"
    if hasattr(train_data, "classes"):
        ldc: int = len(train_data.classes)
        assert(ldc > 0), "no labels"
        if (num_classes > 0):
            assert(num_classes >= 2), "binary at least"
        num_classes = ldc

    if (test_data is not None):
        assert(len(test_data) > 1), "no samples"
        if hasattr(test_data, "classes"):
            assert(len(test_data.classes) > 0), "no labels"

    data_kwargs = {'batch_size': batch_size*4} # multi-batch 4x: split train later
    # num_workers = Number Threads die die CPU nutzt um die Daten nachzuladen, empirisch gesetzt, ggf. angepassen !!
    if is_cuda: # nm=12 suggested by torch, was 25 (6 is faster than 12)
        nw:int = 8 if torch.cuda.is_available() and (torch.cuda.device_count() > 1) else 4
        cuda_kwargs =  {'num_workers': nw,# optimal value is setting / batch dependend, should be a parameter
                        'pin_memory': True,
                        'drop_last' : False, # SkipLastBatch
                        #'prefetch': False, # test
                        'persistent_workers' : True
                        #'shuffle': True
                        }
        data_kwargs.update(cuda_kwargs)
    dl_time : float = time_time()

    # kommt vom Paket Timm -> DefaultLoader liest von Festplatte ggf., deswegen so schneller
    train_dl:DataLoader = MultiEpochsDataLoader(train_data, **data_kwargs, shuffle=True)# here shuffle (time+5%)
    test_dl:DataLoader = None if (test_data is None) else MultiEpochsDataLoader(test_data, batch_size=batch_size*2) # big-batches

    len_train: int = len(train_dl.dataset)
    len_test: int = 0 if (test_dl is None) else len(test_dl.dataset)
    assert(len_train > 1), "number of samples > 1"

    pos: int = 10000 # MNIST=50k, CIFAR=60k
    if (len_train <= pos):
        train_fast = train_dl # FullBatch == TrainData
    else:
        len2: int = pos + (len_train - pos) // 5 # shrink FBL to smaller size
        len2 -= len2 % batch_size # fit last batch
        ds3, _ = random_split(train_data, [len2, len_train - len2])
        train_fast = MultiEpochsDataLoader(ds3, batch_size=batch_size*2)

    dl_time = time_time() - dl_time # much time is spend here !
    if (dl_time > 6.0):
        print("MultiEpochsDataLoader(%d,%d,%d), cls=%d, dt = %.1f s" % (len_train, len_test, len(train_fast.dataset), num_classes, dl_time))

    # train_dl = DataLoader(train_data, **data_kwargs)
    # test_dl = DataLoader(test_data, **data_kwargs) if test_data is not None else None
    return (train_dl, test_dl, train_fast, num_classes)

def run_benchmarks(train_data:Dataset, test_data:Dataset,
                   model_class, model_hyperparams, optimizers, loss_func, task_name:str = "",
                   runs:int = 1, max_epochs:int = 10, target_loss = float('inf'), batch_sizes = None,
                   different_batch_sets:int = 1, device = torch.device("cpu"), seed0:int = 0,
                   wdecay:float = 0.0, drops:float=0.0, num_classes:int = 0) -> None:

    # from mmh3 import hash128
    from os import path, makedirs
    from datetime import datetime as datetime2
    from UniversalTorchModelTrainer import train
    from Cos2MinTorchFunctionOptimizer import ElraOptimizer # ELRA_class.py (tbd 2024)
    assert(callable(model_class)), "model_class not model()"

    initilizations = [model_class(*model_hyperparams).state_dict() for i in range(1, runs+1)]
    initilizations = [
        (hash(str(init)).to_bytes(8, 'big', signed=True).hex(), init)
        # (hash128(str(init)).to_bytes(16, 'big').hex(), init)
        for init in initilizations ]
    # for cid, checkpoints in initilizations:
    #    torch.save(checkpoints, f"{cid}.pt") # why we save them?

    batch_seeds = [torch.randint(1, 1000, (1,)).item() for i in range(0, 1)] # only 1 number
    if (seed0 > 1000) and (1 <= 1): batch_seeds = [ seed0 ] # optional

    # if not path.exists("benchmarks"): makedirs("benchmarks")

    counter: int = 0
    logf = open("history.txt", "a")
    GetTorchInfo(logf)

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
                        if (drops > 0.0):
                            print("append_dropout to model", drops)
                            append_dropout(model, drops)
                        pdim: int = sum(p.numel() for p in model.parameters())

                        if True: # path.exists("/etc/"): # Windows not supported (01.11.2023)
                            s: float = time_time()
                            torch.compile(model, mode='reduce-overhead', backend='cudagraphs')
                            print("Compile time %.3f sec, n=%d" % (time_time() - s, pdim))
                        else:
                            print("Compile <Windows=Off>, n=%d" % (pdim))
                        if (pdim > 12000000):
                            assert(device != torch.device("cpu")), "VERY SLOW on CPU"

                        logf.write("#RUN,%d,%s,%d,%d,%d,%s,%s\n" % (counter,o_name,batch_size,pdim,seed0,datetime2.now().strftime('%Y-%m-%d_%H-%M-%S'),str(torch.get_default_dtype())))
                        logf.flush()
                        print(o_name, opt_param, batch_size, seed, sid)

                        if opt_class is ElraOptimizer: # hack as long as optimizer does not fit generall torch style
                            optimizer = opt_class(model.parameters(), model, *opt_param, wd=wdecay)
                        else:
                            if (o_name != "SGD"):
                                optimizer = opt_class(model.parameters(), *opt_param)
                            else: # SGD, DeVries (2017), lr=0.1, bs=128
                                lr = opt_param[0]
                                print("LR=", lr)
                                optimizer = opt_class(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

                        torch.manual_seed(seed) # set seed before each run, to make sure same batches are used

                        dataloaders = GetDataLoaders(batch_size, train_data, test_data, num_classes, next(model.parameters()).is_cuda)

                        start_time : float = time_time()

                        # Kern-Training
                        # losses, batches, epochs, types, steps, f_calls, g_calls = train()
                        _, _, _, _, _, _, _ = train(dataloaders, model, loss_func,
                                optimizer, max_epochs, target_loss, batch_size=batch_size, device=device, logf=logf)

                        runtime : float = time_time() - start_time
                        print("Training runtime: %.6f sec (%.1fh, ep=%d)" %
                            (runtime, runtime * (1/3600.0), max_epochs))
                        #exit() #break here for not saving
                        counter += 1


                        # with open("history.txt", "a") as log:
                        logf.write("#END,%d,%d,%.3f,%s,\n" % (counter-1, max_epochs, runtime, filename))

    logf.close()
    return

# EoF.
