# NNBenchmark.py (2024)
# container class only

import torch
from torch.utils.data import Dataset, DataLoader  # random_split
from torch import cuda, get_default_dtype
# from fastparquet import write as fastp_write
# import pandas as pd # optional: disabled below
from time import time as time_time # shrink function table
from math import inf
device_cpu: torch.device = torch.device('cpu')

def GetTorchInfo(f) -> None:
    "torch env info"
    if (f is None) or (f.closed) or (f.tell() > 9): return
    import platform
    f.write("#Torch:%s,%s,%s,%s+%s\n" %
        (torch.__version__, platform.node(), get_default_dtype(), cuda.is_available(), cuda.is_initialized()))
    return

def GetDeviceList(model, device):
    "intern: other MultiGpuDevice (SMP/DDP)" # todo-unused
    device2 = None # device_cpu

    if not cuda.is_available(): return None, None
    # assert len(device)

    # return device, deepcopy(model) # debug/test on single-gpu
    return None, None # return (use single gpu only) !!!!! (comment out to SMP/DDP)

    did: int = torch.zeros(0, device=device).get_device()
    dc:  int = cuda.device_count()
    if did < 0: return None, None # cpu-only
    if dc  < 2: return None, None # single gpu

    m = [cuda.mem_get_info(i) for i in range(dc)] # (free_bytes, device_bytes)
    used_mb = [(m[i][1]-m[i][0])>>20 for i in range(dc)]
    dev0mb:int = m[0][1] >> 20 # MB physical VRAM

    if True:
        if (model is None):
            device2 = torch.device("cuda", (did + 1) % dc)
        else:
            if sum(p.numel() for p in model.parameters()) > (1<<20):
                device2 = torch.device("cuda", (did + 1) % dc)

    print("(Multi-GPU detected) %dx%dMB, used:%s, d2=%s" % (dc, dev0mb, str(used_mb), str(device2)))
    # model2 = None if (model is None or device2 is None) else deepcopy(model).to(device2)
    return device2 # model2

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

def CheckDataLoaderHist(dl:DataLoader, num_classes:int=0, silent:bool=False) -> float:
    if dl is None or not len(dl): return -1.0
    import numpy as np
    npl = np.concatenate([y.numpy() for _, y in dl])
    cls: int = np.max(npl) + 1  # classes
    assert cls >= 2, "single label"
    h = np.histogram(npl, bins=cls, range=(0, cls), density=False)

    h2 = h[0][ : cls]
    m1, m2 = np.min(h2), np.max(h2)
    if m1 == m2:
        if not silent:
            print("DataLoader hist: %d==%d, cls=%d, len=%d, ctr = 0%%" \
                % (m1, m2, cls, len(npl)))
        return 0.0
    c: float = (m2 - m1) / (m2 + m1)  # contrast
    if not silent:
        std = np.std(h2)
        print("DataLoader hist: %d<%d, std=%.3f±%.3f, cls=%d, len=%d, ctr = %.2f%%" \
            % (m1, m2, len(npl)/cls, std, cls, len(npl), c*100))
    return c

def GetTransform(data:Dataset):
    "extract transform of Dataset or Subset"
    assert data is not None, 'need source Dataset'
    sub_cnt: int = 0
    trafo = None
    classes: int = 0
    d: Dataset = data
    while hasattr(d, 'dataset'):
        d = d.dataset; sub_cnt+=1
    if hasattr(d, 'transform'):
        trafo = d.transform
    if hasattr(d, 'classes'):
        classes = len(d.classes)
    return sub_cnt, trafo, classes

def TransformStr(data) -> str:
    "print short dataset-transform info"
    if data is None: return '(none)'
    type_str:str = str(type(data)).split('.')[-1].split("'")[0]
    sub_cnt, transform, _ = GetTransform(data)

    if 'Subset' in type_str:
        type_str += str(sub_cnt)
    # print(type(transform)) # torchvision.transforms.transforms.Compose
    ds = data.dataset if hasattr(data, 'dataset') else data  # base
    #if transform is None and hasattr(ds, 'transform'):
    #    transform = ds.transform
    # assert isinstance(transform, torchvision.transforms.transforms.Compose)
    s, ts = '', str(transform)
    if transform is not None:
        assert 'Compose(' in ts, 'torchvision.transforms.transforms.Compose'
        for p in ['RandomCrop','CenterCrop','RandomHorizontalFlip','RandomVerticalFlip','RandomResizedCrop','Normalize']:
            if p in ts: s += p + ','
    lc: int = len(ts.splitlines()) - 2
    ldc:int = len(data) # len(data.indices) if hasattr(data, 'indices') else
    if not s: return str('%d/%dx(empty:%d)' % (ldc, len(ds), lc)) + type_str
    return str('%d/%dx(%s:%d)' % (ldc, len(ds), s, lc)) + type_str

def prepare(dataset, rank:int, world_size:int, batch_size:int=32, pin_memory:bool=False, num_workers:int=0):
    "unused (DistributedSampler)"
    # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    from torch.utils.data.distributed import DistributedSampler
    # assert cuda.device_count() > 1, "Multi-GPU"
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    return DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

def ReplaceTransform(dst:Dataset, trafo) -> None:
    "change transform for Subset + Dataset"
    # assert hasattr(src, 'transform'), "src-dataset"
    # _, trafo, _ = GetTransform(src)
    d: Dataset = dst
    while hasattr(d, 'dataset'):
        d = d.dataset
    assert trafo is not None, "src-dataset"
    assert hasattr(d, 'transform'), "dst-dataset"
    d.transform = trafo  # drop augmentation
    return

def GetDataLoaders(batch_size:int, train_ds:Dataset, test_ds:Dataset, ttrafo, num_classes:int, is_cuda:bool) -> tuple:
    "return 3x MultiEpochsDataLoader"
    from torch.utils.data import DataLoader, random_split, Subset
    from timm.data.loader import MultiEpochsDataLoader
    from torchvision import transforms
    from copy import deepcopy
    assert(batch_size >= 1), "positive integer"
    # print("Dataset.classes:", len(train_ds.classes), len(test_ds.classes))
    empty_tf = transforms.Compose([ transforms.ToTensor(), transforms.CenterCrop(0), ])
    train_len0:int = len(train_ds)
    assert(train_len0 > 1), "no samples"
    _, _, ldc = GetTransform(train_ds)
    assert ldc > 0, 'no classes, binary at least'
    num_classes = ldc

    test_trafo:Compose = ttrafo
    if test_ds is not None:
        assert len(test_ds) > 1, "no samples"
        if hasattr(test_ds, "classes"):
            assert len(test_ds.classes) > 0, "no labels"
        if test_trafo is None:  # empty param
            _, test_trafo, _ = GetTransform(test_ds)  # derive from test
        else:
            if hasattr(test_ds, "dataset"): test_ds = deepcopy(test_ds) # only Subset
            ReplaceTransform(test_ds, test_trafo)  # overwrite (if param)
    if test_trafo is not None:
        assert isinstance(test_trafo, transforms.transforms.Compose)

    data_kwargs = {'batch_size': batch_size*4} # multi-batch 4x: split train later
    # num_workers = Number CPU-threads for file-IO, adjust !!
    if is_cuda: # nm=12 suggested by torch, was 25 (6 is faster than 12)
        nw:int = 8 if cuda.is_available() and (cuda.device_count() > 1) else 4
        cuda_kwargs =  {'num_workers': nw,# optimal value is setting / batch dependend, should be a parameter
                        'pin_memory': True,
                        'drop_last' : False, # SkipLastBatch
                        #'prefetch': False, # test
                        'persistent_workers' : True
                        #'shuffle': True
                        }
        data_kwargs.update(cuda_kwargs)
    dl_time : float = time_time()

    valid_fast:DataLoader|None = None
    val_cut: int = max(1024*1, num_classes*10)  # switch here for val
    if val_cut >= train_len0: val_cut = train_len0 // 2
    val_cut += 0 - (val_cut % 8) # round-8
    if val_cut > 1 and train_len0 >= 0: #50000:
        print('prepare valid:Dataset %d-%d' % (train_len0, val_cut))
        assert test_ds is not None, 'need test_ds.transform'
        val_ds:Dataset = deepcopy(train_ds)
        if test_trafo is not None:
            ReplaceTransform(val_ds, test_trafo)
        # from sklearn.model_selection import train_test_split # scikit-learn
        # train_indices, val_indices = train_test_split(list(range(train_len0)))
        #if val_ds is None:
            # gen = torch.Generator().manual_seed(42)
            #train_ds, ds4_val = random_split(train_ds, [train_len0 - val_cut, val_cut])
            # assert train_ds.dataset.transform == test_ds.transform, "train<>test"
        if True:
            assert val_cut < train_len0, "reduce valid."
            indices: tt.tensor = torch.randperm(train_len0)
            train_indices, val_indices = indices[:-val_cut], indices[-val_cut:]
            assert len(train_indices), "empty list"
            train_ds = Subset(train_ds, train_indices)
            ds4_val = Subset(val_ds, val_indices)
            # assert ds4_val.dataset.transform == test_ds.transform, "valid<>test"

        # if val_ds is not None: print('src:', val_ds.transform)
        print('TF test+valid:', TransformStr(test_ds), ';', TransformStr(ds4_val))
        valid_fast = MultiEpochsDataLoader(ds4_val, batch_size=batch_size*4)
        CheckDataLoaderHist(valid_fast)

    train2_ds:Dataset = deepcopy(train_ds)  # full-batch-fast (exclude valid.)
    ReplaceTransform(train2_ds, test_trafo)

    # train_ds, val_ds = MyDataset(train_transform), MyDataset(val_transform)
    # train_ds, val_ds = Subset(train_ds, train_indices), Subset(val_ds, val_indices)

    # from pip Timm -> DefaultLoader reads Files (faster?)
    train_dl:DataLoader = MultiEpochsDataLoader(train_ds, **data_kwargs, shuffle=True) # shuffle (time+5%)
    test_dl:DataLoader = None if (test_ds is None) else MultiEpochsDataLoader(test_ds, batch_size=batch_size*4) # big-batches

    is_distributed: bool = False
    if is_distributed:  # SMP = Multi-GPU
        from torch.utils.data.distributed import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel
        sampler = DistributedSampler(train_ds) if is_distributed else None
        loader = DataLoader(train_ds, shuffle=(sampler is None), sampler=sampler)
        model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    len_train: int = len(train_ds) # (train_dl.dataset)
    len_test: int = 0 if (test_dl is None) else len(test_dl.dataset)
    assert(len_train > 1), "number of samples > 1"

    pos: int = 10000  # MNIST=50k, CIFAR=60k
    if len_train <= 60000:
        # train_fast = train_dl  # FullBatch == TrainData
        train_fast = MultiEpochsDataLoader(train2_ds, batch_size=batch_size*4)
    else:
        assert pos <= len_train, 'avoid negative ds-split'
        len2: int = pos + (len_train - pos) // 5  # shrink FBL to smaller size
        len2 -= len2 % batch_size  # fit last batch
        ds3, _ = random_split(train2_ds, [len2, len_train - len2])  # !!!!
        train_fast = MultiEpochsDataLoader(ds3, batch_size=batch_size*4)
        # print(len2, pos, len_train, len(ds3), len(train_fast.dataset.indices))

    # CheckDataLoaderHist(train_fast) # debug
    print('TF train 1+2:', TransformStr(train_ds), ';', TransformStr(train_fast.dataset))
    dl_time = time_time() - dl_time  # much time is spend here !
    if dl_time > 1.5:
        train_fast_ds_len: int = len(train_fast.dataset.indices) # !no:numel()!
        print("MultiEpochsDataLoader(%d,%d,%d), cls=%d, dt = %.1f s" %
        (len_train, len_test, train_fast_ds_len, num_classes, dl_time))

    # train_dl = DataLoader(train_ds, **data_kwargs)
    # test_dl = DataLoader(test_ds, **data_kwargs) if test_ds is not None else None
    return (train_dl, test_dl, train_fast, valid_fast, num_classes)

def run_benchmarks(train_ds:Dataset, test_ds:Dataset, ttrafo, 
                   model_class, model_hyperparams, optimizers, loss_func, task_name:str = "",
                   runs:int = 1, max_epochs:int = 10, target_loss = inf, batch_sizes = None,
                   different_batch_sets:int = 1, device = device_cpu, seed0:int = 0,
                   wdecay:float = 0.0, drops:float=0.0, num_classes:int = 0) -> None:

    # from mmh3 import hash128
    from os import path, makedirs
    from datetime import datetime as dt2
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

        filename: str = "benchmarks/" + dt2.now().strftime('%Y-%m-%d_%H-%M-%S')+"_"+o_name+"_"+task_name+"_seed"+str(seed0)+".parquet"

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
                            assert device != device_cpu, "VERY SLOW on CPU"

                        logf.write("#RUN,%d,%s,%d,%d,%d,%s,%s\n" % (counter,o_name,batch_size,pdim,seed0,dt2.now().strftime('%Y-%m-%d_%H-%M-%S'),str(get_default_dtype())))
                        logf.flush()
                        print(o_name, opt_param, batch_size, seed, sid)

                        torch.manual_seed(seed) # set seed before each run, to make sure same batches are used

                        dataloaders = GetDataLoaders(batch_size, train_ds, test_ds, ttrafo, num_classes, device != device_cpu) # next(model.parameters()).is_cuda
                        num_classes = dataloaders[-1]

                        if opt_class is ElraOptimizer: # hack as long as optimizer does not fit generall torch style
                            optimizer = opt_class(model.parameters(), model, batch_size, num_classes, *opt_param, wd=wdecay)
                        else:
                            if (o_name != "SGD"):
                                optimizer = opt_class(model.parameters(), *opt_param)
                            else: # SGD, DeVries (2017), lr=0.1, bs=128
                                lr = opt_param[0]
                                print("LR=", lr)
                                optimizer = opt_class(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

                        start_time : float = time_time()

                        # Kern-Training
                        # losses, batches, epochs, types, steps, f_calls, g_calls = train()
                        # _, _, _, _, _, _, _ = 
                        train(dataloaders, model, loss_func,
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
