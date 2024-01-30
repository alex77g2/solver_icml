# UniversalTorchModelTrainer.py (2023)
# includes own DataLoader for constant Data

import torch as tt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# from torch.cuda.amp import autocast, GradScaler
# import math
from Cos2MinTorchFunctionOptimizer import SelfConstOptimTorch
from time import time as time_time
from time import sleep
from os import path, remove
import pickle, copy, glob
from statistic_helper import GlobalStatist as gstat
from datetime import datetime as datetime2

def PickleWrite(fn: str, lst: list) -> int:
    if (len(fn) < 2) or (len(lst) < 1): return -1
    f = open(fn, 'wb')
    if (f.closed): return -2 # silent fail
    pickle.dump(lst, f)
    f.close()
    return len(lst)
    
def PickleRead(fn: str, llen: int) -> list:
    if (len(fn) < 2) or (not path.isfile(fn)): return []
    f = open(fn, 'rb')
    if (f.closed): return []
    lst = pickle.load(f)
    f.close()
    if (llen > 0):
        if (llen != len(lst)):
            remove(fn) # unexpected content
            lst = []
    return lst
    

class DataLdrFast:
    dlf_Instances: int = 0
    # dlf_listXY = []
    # dlf_Xsize: tt.Tensor = tt.zeros([0], dtype=tt.int32) # X.size(0) = 1..BatchSize
    
    def InitVectors(self) -> None:
        self.dlf_samples = 0
        self.dlf_RandomPos: np.ndarray = np.zeros(0, dtype=np.uint16) # numpy.random.shuffle()
        self.dlf_Images: tt.Tensor = tt.zeros([0,0,0,0], dtype=self.dlf_ImgType)
        self.dlf_Labels: tt.Tensor = tt.zeros([0], dtype=tt.uint8)
        self.dlf_LastTmp: tt.Tensor = tt.zeros([0], dtype=tt.get_default_dtype()) # last batch (if smaller + training)
        self.dlf_LastLab: tt.Tensor = tt.zeros([0], dtype=tt.uint8)
        return
    
    def __init__(self):
        DataLdrFast.dlf_Instances += 1
        self.dlf_samples: int = 0 # Sample-Count (Images) in Dataset
        self.dlf_mb_size: int = 0 # MiniBatchSize (variable)
        self.dlf_sample_elem: int = 0 # 32x32x3 or 28x28x1
        self.dlf_label_max = -1
        self.dlf_final_tail = 0
        self.dlf_ShuffleCnt = 0
        self.dlf_EpochCount = 0
        self.dlf_pos_ext: int = 0
        self.dlf_pos_fast: int = 0 # no shuffle
        self.dlf_final_fulls: int = 0 # (dlf_samples/dlf_mb_size)
        self.dlf_final_tail:  int = 0 # (dlf_samples%dlf_mb_size)
        self.dlf_init_mbs: int = 0
        self.dlf_shuffle: bool = False
        self.dlf_is_train: bool = False
        self.dlf_only_lab: bool = False
        self.dlf_ImgType = tt.bfloat16 # .float32 or .bfloat16
        self.dlf_device: tt.device = tt.device('cpu')
        # new: 1 big Tensor
        self.InitVectors()
        
    def __delete__(self, instance):
        # print("delete DataLdrFast,", self.dlf_samples)
        self.InitVectors()
        DataLdrFast.dlf_Instances -= 1
    
    @staticmethod
    def CacheFilename(data:DataLoader, is_train:bool=False) -> str:
        if (data is None) or (not isinstance(data, DataLoader)) or (len(data) < 1):
            return ""
        return "ds_"+str(len(data.dataset))+"_"+str(len(data))+".tmp" # .pt
    
    @staticmethod
    def CalcDatasetBytes(data:DataLoader) -> tuple[int, int]:
        "calc number of bytes in DataSet+Labels"
        if (data is None) or (not isinstance(data, DataLoader)) or (len(data) < 1):
            return 0, 0
        size_img: int = 0
        size_lab: int = 0
        for X, y in data:
            size_img += tt.numel(X) * X.element_size()
            size_lab += tt.numel(y) * y.element_size()
        # FastenDataloader(179+468)(29+78)[MB+KB],(235+10).
        return size_img, size_lab
    
    @staticmethod
    def EstimBatchSize(data:DataLoader) -> int:
        if (data is None) or (not isinstance(data, DataLoader)):
            return -1
        if (len(data.dataset) < 1) or (len(data) < 1):
            return 0
        est_bs: int = int(1 + len(data.dataset) // len(data)) # always even
        # print("EBS:", len(data.dataset), len(data), est_bs) # fails for low batch count
        return 1 if (est_bs < 2) else (est_bs) & (~1) # usually even
        
    def GetLabelStr(self, cnt: int) -> str:
        n: int = min(cnt, self.dlf_samples)
        if (n < 1): return "-"
        s: str = ""
        for i in range(n):
            s += str( int(self.dlf_Labels[i]) )
        return s
        
    @staticmethod
    def CheckPair(ds1, ds2) -> bool:
        if (ds1.dlf_samples < 1) or (ds2.dlf_samples < 1): return True # one is empty
        classes1:int = ds1.dlf_label_max + 1
        # print("CP:", ds1.dlf_label_max, ds2.dlf_label_max, ds1.dlf_samples, ds2.dlf_samples)
        assert(ds1.dlf_label_max > 0), "No/empty TrainLabels!"
        assert(ds2.dlf_label_max > 0), "No/empty TestLabels!"
        assert(ds1.dlf_label_max == ds2.dlf_label_max), "ClassesMaxConflict"
        assert(ds1.dlf_samples >= ds2.dlf_samples), "Train-Test-Swap"
        
        if (ds1.dlf_sample_elem != ds2.dlf_sample_elem):
            print("PairConflict(cls=%d, smp:%d<>%d, se:%d<>%d)!" % (classes1,
            ds1.dlf_samples, ds2.dlf_samples, ds1.dlf_sample_elem, ds2.dlf_sample_elem))
            if (classes1 == 10): # 10 classes
                for f in glob.glob("ds_" + str(ds2.dlf_samples) + "_*.tmp.pt"):
                    remove(f)
                    print(" <%s>" % f)
                    # remove("ds_10000_10.tmp.pt") # MNIST and CIFAR cache-file collision
                print("Please run again: tmp-file has been removed!")
                exit(0)
            assert(0), "PixelFormatConflict (auto solve by tmp-file-removal)"
        
        if (classes1 != 10): return True
        ls2: str = ds2.GetLabelStr(8) # Test-Labels
        if (ds1.dlf_samples == 60000) and (ds2.dlf_samples == 10000):
            assert(ls2 == "72104149" or ls2 == "92116146"), "MNIST_test" # +MNIST-Fa.
        if (ds1.dlf_samples == 50000) and (ds2.dlf_samples == 10000):
            assert(ls2 == "38806616"), "CIFAR10_test (350MB)"
        # remove("ds_10000_10.tmp.pt") # only test has mixup by size
        return True
        
    def ClearData(self, msg: str = "") -> None:
        if (len(msg) > 0):
            print("ClearDataset(n=%d,classes=%d):%s" % (self.dlf_samples, self.dlf_label_max+1, msg))
        self.dlf_samples = 0
        self.dlf_mb_size = 0
        self.dlf_label_max = -1
        self.dlf_EpochCount = 0
        # self.dlf_listXY = []
        self.dlf_RandomPos = np.zeros(0, dtype=np.uint16)
        self.dlf_Labels = tt.zeros([0], dtype=tt.uint8)
        self.dlf_LastTmp = tt.zeros([0], dtype=tt.get_default_dtype()) # .float32
        self.dlf_LastLab = tt.zeros([0], dtype=tt.uint8)
    
    def GetDsHash(self, maxidx: int = 0) -> str:
        if (self.dlf_samples < 1): return "<empty_DS>"
        if (maxidx <= 0) or (maxidx > self.dlf_samples): maxidx = self.dlf_samples
        lab_sum: int = int( tt.sum(self.dlf_Labels[0: maxidx]).item() )
        flat: tt.Tensor = tt.flatten(self.dlf_Images[0: maxidx])
        pix_xor: int = int( tt.sum(flat.view(tt.int32)).item() ) # .bitwise_xor()
        pix_sum: float = float( tt.sum(flat).item() )
        return str("[%d/%d]x[%d+%d](%d+%.6g+0x%x)" % (maxidx, self.dlf_samples,
        self.dlf_Labels.element_size(), flat.element_size(), lab_sum, pix_sum, pix_xor))
    
    def SetMiniBatchSize(self, mbs: int, shuffle:bool=False, seed:int=0) -> bool:
        if (mbs < 0) or (self.dlf_samples < 1): return False # silent
        mbs = min(mbs, self.dlf_samples) # SingleLargeBatch
        self.dlf_mb_size = mbs if (mbs > 0) else self.dlf_init_mbs # from Timm.DataLoader
        mbs = self.dlf_mb_size
        assert(self.dlf_mb_size > 0), "MiniBatchSize<1"
        if (self.dlf_only_lab):
            assert(not shuffle), "do not shuffle test data"
            return False # silent
        self.dlf_final_fulls = (self.dlf_samples // mbs)
        self.dlf_final_tail  = (self.dlf_samples % mbs)
        self.dlf_shuffle = shuffle
        img_type: tt.dtype = tt.get_default_dtype() # tt.float32
        if (self.dlf_final_tail > 0):
            xshape: list[int] = list(self.dlf_Images.shape)
            xshape[0] = mbs # self.dlf_final_tail
            self.dlf_LastTmp = tt.zeros(xshape, dtype=img_type, device=self.dlf_device)
            self.dlf_LastLab = tt.zeros([mbs], dtype=tt.uint8, device=self.dlf_device)
            lfp: int = int(self.dlf_final_fulls * mbs)
            t = self.dlf_Images[lfp: self.dlf_samples]
            # print(t.shape, lfp, self.dlf_final_fulls, self.dlf_samples)
            self.dlf_LastTmp[0: self.dlf_final_tail] = t.to(dtype=img_type) # crash
            t = self.dlf_Labels[lfp: self.dlf_samples]
            self.dlf_LastLab[0: self.dlf_final_tail] = t.to(dtype=tt.uint8)
        else:
            self.dlf_LastTmp = tt.zeros([0], dtype=img_type)
        if (shuffle):
            mbcnt: int = (self.dlf_samples + 0) // mbs # no_round_up(mbs - 1)
            if (seed != 0): np.random.seed(seed)
            assert(mbcnt < 65536), "limit 16 bit"
            self.dlf_RandomPos = np.arange(mbcnt, dtype=np.uint16)
            # for i in range(mbcnt): self.dlf_RandomPos[i] = i
            np.random.shuffle(self.dlf_RandomPos)
        else:
            self.dlf_RandomPos = np.zeros(0, dtype=np.uint16)
        return True
    
    def __iter__(self): # TODO
        self.dlf_pos_ext = 0
        self.dlf_pos_fast = 0 # rewind
        return self
        
    def __next__(self): # TODO
        if self.dlf_pos_ext <= 5:
            idx: int = self.dlf_pos_ext
            self.dlf_pos_ext += 1
            return idx, 100
        else:
            raise StopIteration
    
    def GetShuffleBatch(self, rewind: bool = False) -> tuple[int, tt.Tensor, tt.Tensor]: # untested
        shuffle: bool = self.dlf_shuffle
        if (rewind) or (self.dlf_pos_ext >= self.dlf_final_fulls):
            # print("GetShuffleBatch.rewind(n=%d)." % self.dlf_samples)
            self.dlf_pos_ext = -1 # rewind
            if (shuffle): # and (len(self.dlf_RandomPos) > 1):
                assert(len(self.dlf_RandomPos) > 1), "call SetMiniBatchSize() before!"
                np.random.shuffle(self.dlf_RandomPos)
            return (-1, None, None)
        elem: int = self.dlf_mb_size # last setting
        # assert(self.dlf_pos_ext < len(self.dlf_RandomPos))
        if (self.dlf_pos_ext == -1):
            self.dlf_pos_ext = 0
            if (self.dlf_final_tail > 0): # (N%mbs)>0
                tail: int = self.dlf_final_tail
                if (shuffle):
                    rid: int = len(self.dlf_RandomPos) // 4 # 25% of RandomOrder
                    ofs: int = elem * self.dlf_RandomPos[rid]
                else: # fill-ups still random (even w/o shuffle)
                    ofs: int = elem * np.random.randint(self.dlf_final_fulls)
                # print(self.dlf_LastLab.shape);exit()
                self.dlf_LastLab[tail: elem] = self.dlf_Labels[ofs+tail: ofs+elem].clone()
                self.dlf_LastTmp[tail: elem] = self.dlf_Images[ofs+tail: ofs+elem].clone()
                return (elem, self.dlf_LastTmp, self.dlf_LastLab) # Tail First (if exist) ! 
        
        start_pos:int = (self.dlf_pos_ext if (not shuffle) else self.dlf_RandomPos[self.dlf_pos_ext]) * elem
        next_pos: int = min(start_pos + elem, self.dlf_samples)
        # assert(elem == next_pos - start_pos)
        self.dlf_pos_ext += 1
        imgs: tt.Tensor = self.dlf_Images[start_pos: next_pos] # .to(device).to(float32)
        return (elem, imgs, self.dlf_Labels[start_pos: next_pos])
    
    def DirectBatch(self, elem: int) -> tuple[int, tt.Tensor, tt.Tensor]:
        "Iterator for 1:1 fast access (no shuffle etc, e.g. test or full batch)"
        if (elem < 0) or (self.dlf_pos_fast >= self.dlf_samples):
            # print("DirectBatch.rewind(n=%d)." % self.dlf_samples)
            self.dlf_pos_fast = 0 # rewind
            return (-1, None, None)
        if (elem == 0): elem = self.dlf_init_mbs # keep from loaded source
        if (elem % 16) and (self.dlf_pos_fast == 0):
            print("DirectBatch: elem%16=%d > 0 !", elem % 16) # alignment
        start_pos:int = self.dlf_pos_fast
        next_pos: int = min(self.dlf_pos_fast + elem, self.dlf_samples)
        elem = next_pos - self.dlf_pos_fast # X.size(0)
        self.dlf_pos_fast = next_pos
        imgs: tt.Tensor = self.dlf_Images[start_pos: next_pos] # .to(device).to(float32)
        return (elem, imgs, self.dlf_Labels[start_pos: next_pos])
    
    def InfoString(self) -> str:
        est_src: str = "<unknown_DS>"
        cls: int = int(self.dlf_label_max) + 1
        if (self.dlf_samples < 1): return "<empty_dataset>"
        lab_str: str = "" if (cls != 10) else self.GetLabelStr(8)
        
        if (cls == 10): # MNIST
            if (self.dlf_sample_elem == (28*28*1)):
                if (self.dlf_samples == 60000):
                    if (lab_str == "57131463"): est_src = "MNIST_train"
                    if (lab_str == "33158971"): est_src = "MNIST-Fas._train"
                if (self.dlf_samples == 10000):
                    if (lab_str == "72104149"): est_src = "MNIST_test"
                    if (lab_str == "92116146"): est_src = "MNIST-Fas._test"
        if (self.dlf_sample_elem == (32*32*3)):
            if (cls == 10):
                if (self.dlf_samples == 50000): est_src = "CIFAR-10_train"
                if (self.dlf_samples == 10000): est_src = "CIFAR-10_test"
            if (cls == 100):
                if (self.dlf_samples == 50000): est_src = "CIFAR-100_train"
                if (self.dlf_samples == 10000): est_src = "CIFAR-100_test"
        if (cls == 196): # CARS
                if (self.dlf_samples == 8144): est_src = "cars196_train"
                if (self.dlf_samples == 8041): est_src = "cars196_test"

        if (cls <= 1) or (self.dlf_sample_elem < 2) or (self.dlf_samples < (cls*10)):
            est_src = "bad_dataset" # + test_lab
        return str("N=%d,E=%d,classes=%d,src=%s" % (self.dlf_samples, self.dlf_sample_elem, cls, est_src))
    
    def CompatibleDS(self, data) -> bool:
        if (not isinstance(data, type(self))) or (len(self.dlf_samples) < 1): return False
        if (data.dlf_samples < 1) or (self.dlf_samples < 1): return False
        return (data.dlf_sample_elem == self.dlf_sample_elem) and (data.dlf_label_max == self.dlf_label_max)
    
    def GetDatasetElemCount(self) -> tuple[int, int]:
        return self.dlf_samples, self.dlf_sample_elem # 60K, 28x28
    
    def _props2list(self) -> list[int]:
        dsn: list[int] = [self.dlf_samples, self.dlf_init_mbs, self.dlf_label_max, int(self.dlf_shuffle), int(self.dlf_is_train)]
        return dsn # tt.tensor(dsn, dtype=tt.int32)
        
    def _list2props(self, dsn: list[int]) -> None:
        assert(len(dsn) == 5), "len(list) <> 5"
        shuf, train = False, False
        self.dlf_samples, self.dlf_init_mbs, self.dlf_label_max, shuf, train = tuple(dsn)
        self.dlf_shuffle, self.dlf_is_train  = bool(shuf), bool(train)
        # self.dlf_sample_elem = 0 # numel()
    
    def _dataset2device(self, device) -> None:
        if (self.dlf_samples < 1):
            return
        self.dlf_device = device
        img_type: tt.dtype = tt.get_default_dtype() # tt.float32
        if (device == tt.device('cpu')):
            self.dlf_Images = self.dlf_Images.to(dtype=img_type)
            self.dlf_Labels = self.dlf_Labels.to(dtype=tt.uint8)
            return
            
        self.dlf_Images = self.dlf_Images.to(device) # .to(tt.bfloat16)
        self.dlf_Labels = self.dlf_Labels.to(dtype=tt.uint8).to(device) # strange: tt.int8 (signed)
        self.dlf_Images = self.dlf_Images.to(dtype=img_type) # here or at usage (memory vs runtime)
        # print(self.dlf_samples, self.dlf_Images.device, self.dlf_Labels.device); exit()
        # print(self.dlf_samples, self.dlf_Images.dtype, self.dlf_Labels.dtype); exit()
    
    def ImportFromFile(self, fn: str, device=tt.device('cpu')) -> int:
        if (len(fn) < 2) or (len(fn) > 999) or (not path.isfile(fn)):
            return -1
        dsnum, self.dlf_Images, self.dlf_Labels = tuple(tt.load(fn))
        assert(len(self.dlf_Images) == len(self.dlf_Labels)), "ImageCount <> LabelCount"
        self._list2props( dsnum.tolist() )
        assert(self.dlf_samples == len(self.dlf_Labels)), "SampleCountVar <> LabelCount"
        pix_sz: int = self.dlf_Images.element_size()
        self.dlf_sample_elem = tt.numel(self.dlf_Images[0])
        i: int = (self.dlf_sample_elem * self.dlf_samples * pix_sz) >> 20 # .bfloat16
        print("DatasetFromCache(n=%d, cls=%d, lab[%s], es=%d+%d, sz=%dMB)." % (self.dlf_samples, self.dlf_label_max+1, self.GetLabelStr(8), pix_sz, self.dlf_Labels.element_size(), i)) # optional
        self._dataset2device(device)
        return 0 # used RAM [MB]
    
    def Import_DataLoader(self, data:DataLoader, is_train:bool=False, maxMB:int=600, device=tt.device('cpu')) -> int:
        if (maxMB <= 0) or (not isinstance(data, DataLoader)) or (len(data) < 1):
            return -1
        maxMB = 600 # test
        dsfn: str = DataLdrFast.CacheFilename(data) + ".pt"
        dlf_Xsize = tt.zeros([len(data)], dtype=tt.int32) # was self.
        dlf_listXY = [] # was self.
        self.dlf_samples = len(data.dataset) # 60K
        self.dlf_init_mbs = DataLdrFast.EstimBatchSize(data)
        self.dlf_is_train = is_train
        i: int = 0
        max_byte:  int = maxMB << 20
        size_byte: int = 0
        max_lab: int = -1 # max(labels)
        np.random.seed(123) # todo
        
        if (len(dsfn) > 1) and (path.isfile(dsfn)):
            self.ImportFromFile(dsfn, device=device)
            if (not is_train):
                for _, y in data:
                    max_lab = max(max_lab, y.max().item())
                assert(max_lab == self.dlf_label_max), "Bad DataLoaderCacheFile! (delete ds_*.tmp.pt)"
            dsfn = "" # no save after load
            return 0 # loaded from file cache
        
        for X, y in data:
            size_byte += tt.numel(X) * X.element_size() # + tt.numel(y) * y.element_size()
            # print("X0:", tt.numel(X), X.element_size(), X.size(0), X.shape); exit(0) # X0: 2408448 4(float32) 16(bs) [16, 3, 224, 224]
            dlf_Xsize[i] = X.size(0); i += 1
            max_lab = max(max_lab, y.max().item())
            if (size_byte <= max_byte):
                y2 = y.clone().to(dtype=tt.uint8) # new (from tt.int64)
                X2 = X.clone().to(dtype=self.dlf_ImgType) # optional/halfing
                dlf_listXY.append( (X2.to(device), y2.to(device)) )
            else:
                dlf_listXY.append( (None, y.to(dtype=tt.uint8, device=device)) )

        if (size_byte > max_byte):
            print("Warn: %d MB > %d MB, c=%d, skip=%d!" % (size_byte>>20, max_byte>>20, sum(dlf_Xsize), i))
            if 1: # (is_train):
                self.ClearData()
                return -5
            else:
                self.dlf_only_lab = True # only test-data (no shuffle)
            
        if (len(dlf_listXY) != len(dlf_Xsize)): # Error:IDL(50<>391,i=391)!
            print("Error:IDL(%d<>%d,i=%d)!" % (len(dlf_listXY), len(dlf_Xsize), i)); exit()

        self.dlf_sample_elem = tt.numel(X[0])
        self.dlf_label_max = int(max_lab)
        if (max_lab <= 0):
            print("Warn: No Labels in Dataset (n=%d) !" % (self.dlf_samples))
        assert(max_lab > 0), "No/empty Labels!"
        self.dlf_init_mbs = int(dlf_Xsize[0]) # overwrite estimate
        xshape: list[int] = list(X.shape)
        xshape[0] = 0 if (self.dlf_only_lab) else self.dlf_samples

        if (self.dlf_label_max > 255): # todo: use int16 then
            print("Error:IDL(n=%d,classes=%d>255)!" % (self.dlf_samples, self.dlf_label_max))
            self.ClearData()
            return -2

        self.dlf_Images = tt.zeros(xshape, dtype=self.dlf_ImgType) # .bfloat16, [1024,1,8,8]
        self.dlf_Labels = tt.zeros([self.dlf_samples], dtype=tt.uint8) # .to(device)

        pos: int = 0
        for X, y in dlf_listXY:
            nxt: int = min(self.dlf_samples, pos + len(y))
            if (not self.dlf_only_lab):
                self.dlf_Images[pos: nxt] = X.to(dtype=self.dlf_ImgType) # .bfloat16
            self.dlf_Labels[pos: nxt] = y # .to(device) # todo: only one final big copy
            pos = nxt
        del dlf_listXY

        if (self.dlf_only_lab):
            self.dlf_mb_size = self.dlf_init_mbs
            self._dataset2device(device)
            print("FromData.OnlyLabel(n=%d, bs=%d, cls=%d, bc=%d)." % (self.dlf_samples, self.dlf_mb_size, self.dlf_label_max+1, i))
            return 0

        if (len(dsfn) > 3) and (self.dlf_samples > 0):
            dsnum: tt.Tensor = tt.tensor(self._props2list(), dtype=tt.int32)
            print("WriteDsCache(n=%d): %s" % (self.dlf_samples, dsfn))
            tt.save( [dsnum, self.dlf_Images, self.dlf_Labels], dsfn )
        self._dataset2device(device)

        print("FromData(n=%d, se=%d, cls=%d, bc=%d)." % (self.dlf_samples, self.dlf_sample_elem, self.dlf_label_max+1, i))
        size_byte >>= 20 # now MB
        if (size_byte > max_byte):
            print("Skip Import: %d > %d MB, samples = %d K." % (size_byte, maxMB, self.dlf_samples>>10))
            self.ClearData()
            return -3

        self.SetMiniBatchSize(self.dlf_init_mbs)
        return size_byte # used RAM [MB]

    def FullShuffle(self, seed:int) -> None:
        if (self.dlf_samples < 2) or (self.dlf_sample_elem < 1): return
        self.dlf_ShuffleCnt += 1
        if (0 == self.dlf_ShuffleCnt % 2): return # skip epochs
        return # debug (early overfitting)
        print("FullShuffle(%d, sum=%d)" % (self.dlf_ShuffleCnt, tt.sum(self.dlf_Labels)))
        if (seed != 0): np.random.seed(seed)
        glue: int = 8 # glue some together
        blk: int = self.dlf_samples // glue
        r: np.ndarray = np.arange(blk, dtype=np.uint32)
        np.random.shuffle(r)
        # ne: int = self.dlf_sample_elem # elem/sample
        
        src: tt.Tensor = self.dlf_Labels.to(tt.device('cpu')).clone() # e.g. 1x 60 KB
        lab: tt.Tensor = tt.zeros_like(src) # shuffle labels on CPU
        for i in range(blk):
            ofs1, ofs2 = i*glue, r[i] * glue
            lab[ofs1: ofs1+glue] = src[ofs2: ofs2+glue]
        self.dlf_Labels = lab.to(self.dlf_device)
        del lab
            
        # Shuffle Images within GPU (do not transfer much memory)
        src = self.dlf_Images.clone() # MegaBytes (MNIST=200MB, CIFAR=600MB)
        for i in range(blk):
            ofs1, ofs2 = i*glue, r[i] * glue
            self.dlf_Images[ofs1: ofs1+glue] = src[ofs2: ofs2+glue]
        del src
        
        if (self.dlf_final_tail > 0):
            todo = 1 # TODO: tail :ok, if SetMiniBatchSize() before GetShuffleBatch()
        
        return

################################################

def FreezeBatchNorm(model):
    # model = ResNet50()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return

def SetParam(model, par: tt.Tensor) -> int:
    "x-tensor into model"
    assert(len(par) > 0), "empty Tensor"
    model.train() # needed if net uses BatchNorm
    # normalisation layers use per-batch statistics + (activates Dropout layers)
    s, e = 0, 0
    for p in model.parameters():
        e += tt.numel(p)
        p.data = tt.reshape(par[s:e], p.size()).to(p.device, p.data.dtype)
        s = e
    return e # n=dim

def ParamLoad(model, fn:str="", nofilewarn:bool=True) -> bool:
    "load x vector from disk" # TODO
    n: int = sum(p.numel() for p in model.parameters())
    assert(n >= 1), "empty model"
    
    if (len(fn) < 2):
        fn = "startx_*.pt"
    fn = fn.replace('*', str(int(n)))
    if (not path.isfile(fn)):
        if (nofilewarn):
            print("ParamLoad(n=%d:%s)=NoFile." % (n, fn))
        return False
    # tt.load(model, 'path')
    par: tt.Tensor = tt.load(fn) # model.load_state_dict()
    assert(len(par) == n), "wrong dimension"
    
    SetParam(model, par)
    pn: float = tt.linalg.vector_norm(par).item()
    print("ParamLoad(n=%d,av=%.3e,n2=%.3e)=OK" % (n, tt.mean(par), pn))
    return True

def GetParam(model) -> tt.Tensor:
    "model to x-tensor"
    params = []
    for p in model.parameters():
        params.append( p.data.view(-1) )
    return tt.cat(params)

def ParamSave(model, fn:str="") -> bool:
    "store x vector on disk" # tested
    # tt.save(model, 'path')
    n: int = sum(p.numel() for p in model.parameters())
    # for p in model.parameters(): n += tt.numel(p)
    assert(n >= 1), "empty model"
    
    if (len(fn) < 2):
        fn = "lastx_*.pt"
    fn = fn.replace('*', str(int(n)))
    # if (path.isfile(fn)): print("ParamSave.overwrite(%s)" % fn)
    
    par: tt.Tensor = GetParam(model)
    tt.save(par, fn)
    
    pn: float = tt.linalg.vector_norm(par).item()
    print("ParamSave(%s,av=%.3e,n2=%.3e)=OK" % (fn, tt.mean(par), pn))
    return path.isfile(fn)

last_x: tt.Tensor = None # debug only
diff_sum: tt.Tensor = None # debug only
elra_solver: bool = True

def CheckElraSolver(optim) -> bool:
    "verify ELRA class"
    global elra_solver
    assert(optim is not None), "input <> None"
    try:
        optim.GetParamAvg
    except AttributeError :
        elra_solver = False
    else:
        elra_solver = True
    print("ELRA:", elra_solver, ", name=", type (optim).__name__)
    return elra_solver

def Interpolx(model, cost_function, dl:DataLoader) -> None:
    "debug: valley cut loss"
    global last_x
    x0 = GetParam(model)
    if (last_x is None) or (dl is None):
        last_x = x0
        return
    print("Interpolx:")
    last, fvl = 0.0, []
    for i in range(101):
        f1: float = i * 0.01
        par: tt.Tensor = last_x * f1 + x0 * (1.0 - f1)
        SetParam(model, par)
        loss: float = full_batch_loss(model, dl, cost_function, x0.device)
        d = 0.0 if (i==0) else loss-last
        fvl.append(loss)
        last = loss
        print("%.2f %.6f %.4e," % (f1, loss, d), end="")
        
    SetParam(model, x0) # restore
    last_x = x0 # double usage (debug only)
    print(".done(%.6f<%.6f<%.6f)." % (min(fvl),sum(fvl)/len(fvl),max(fvl)), flush=True)

def WriteDsHist(reset:bool, log) -> None:
    "debug: histogram of epoch movement exponents"
    global diff_sum
    if (log is None) or (diff_sum is None):
        return
    if (not log.closed):
        _ , e2 = tt.frexp(diff_sum) # get (int) float^2-exponents
        #    print("HIST(",np.min(e2),"<",np.max(e2),"),",np.mean(e2))
        #    print("P(1,50,99):",np.percentile(e2, 1),"<",np.percentile(e2, 50),"<",np.percentile(e2, 99))
        #    #np.median(e2, axis=None, overwrite_input=False)
        e2min, e2max = int(tt.min(e2)), int(tt.max(e2))
        bins: int = 1 + int(e2max - e2min)
        log.write("#hist=%d,s=%.3g,%.3g<%.3g,%d<%d," %
            (len(diff_sum),tt.sum(diff_sum), tt.min(diff_sum),tt.max(diff_sum), e2min, e2max))
        hist,_ = np.histogram(e2.numpy(), bins=bins, range=(e2min, e2max), density=False)
        # print(e2min, e2max, bins, hist);exit()
        log.write("%s\n" % str(hist))
        log.flush()
    if (reset):
        diff_sum = None
    return

def WriteDominant(x: tt.Tensor, t:int = -1, log=None) -> int:
    "debug: plot strong components"
    global last_x, diff_sum
    if (x is None) or (len(x) < 1): return 0
    cpu = tt.device('cpu')
    # xmin, xmax = tt.min(x).item(), tt.max(x).item()
    # x = x.to(dtype=float)
    xsum, xnrm = tt.sum(x).item(), tt.norm(x).item()
    
    if (log is None):
        lf,fc = open("path_dom.dat", "a"), True
    else:
        lf,fc = log, False
    if (lf is None) or (lf.closed):
        print("WriteDominant(t=%d,n=%d):Error=fopen!" % (t,len(x)))
        return -1
        
    if (last_x is None):
        print("Dom(0/%d, xs=%.3g,xn=%.3g, init)." % (len(x), xsum,xnrm))
        lf.write("##NEW,dim=%d,s=%.6g,n=%.6g\n" % (len(x), xsum, xnrm))
        if (fc): lf.close()
        last_x = x.clone()
        return 0
        
    assert(len(x) == len(last_x)), "length(x_Tensors) differs"
    d, last_x = (x - last_x), x
    gmin, gmax = tt.min(d).item(), tt.max(d).item()
    gsum, gnrm = tt.sum(d).item(), tt.norm(d).item()

    th: float = max(abs(gmin), abs(gmax)) * 0.3
    db = (tt.abs(d) > th) # [bool]
    nc: int = tt.sum(db)
    d, db = d.to(cpu), db.cpu()
    
    if (diff_sum is None):
        diff_sum  = d*d
    else:
        diff_sum += d*d
    # print("Dom(%d/%d, xs=%.3g,xn=%.3g, ds=%.3g,dn=%.3g)." % (nc,len(x), xsum,xnrm, gsum,gnrm))
    lf.write("#t=%d,%d/%d, xs=%.3g,xn=%.3g, ds=%.3g,dn=%.3g, th=%.3g\n" %
        (t,nc,len(x), xsum,xnrm, gsum,gnrm, th))
    lf.write("-1,%.6g,%.6g\n" % (xsum, xnrm))
    
    if (gnrm > 0.0):
        x1 = x.to(cpu)
        for i in range(len(x)):
            if (db[i]):
                lf.write("%d,%.6g,%.3g\n" % (i, x1[i], d[i]))

    if (fc): lf.close()
    return 1

# Device-Cache (new class) ..
utmt_DS: DataLdrFast = DataLdrFast()
utmt_TS: DataLdrFast = DataLdrFast()

def StatListStr(lst: list[float]) -> str:
    n : int = len(lst)
    if (n < 2):
        return ("(len=%d<2!)" % n)

    prv : float = lst[0]
    sum_ad : float = 0.0
    for i in range(1,n):
        val : float = lst[i]
        sum_ad += abs(val - prv)
        prv = val
    val = -9.9 if (0.0==sum_ad) else (lst[n-1]-lst[0]) /sum_ad
    # print(len(lst),min(lst),sum(lst)/n,max(lst),val); exit(0)
    return ("(%d:%.3g<%.3g<%.3g:%.3f)" % (len(lst),min(lst),sum(lst)/n,max(lst),val))

def TestEval(model, test_loader:DataLoader, cost_function, device) -> (float,float):
    "calc Accuracy + TestLoss over full test-dataset"
    # model.eval() --
    if (test_loader is None) or (len(test_loader.dataset) < 1):
        return -1.0, -1.0

    global utmt_TS
    utmt_TS_samples: int = utmt_TS.dlf_samples
    only_lab: bool = utmt_TS.dlf_only_lab
    test_loss = tt.zeros(1, dtype=tt.float32).to(device)
    corr = tt.zeros(1, dtype=tt.int64).to(device)
    i: int = 0
    dt: float = time_time()

    if (len(test_loader.dataset) != utmt_TS_samples) or only_lab: # slower
        img_type: tt.dtype = tt.get_default_dtype() # tt.float32
        start_pos, mbs = 0, utmt_TS.dlf_init_mbs
        with tt.no_grad():
            for data, target in test_loader:
                if (only_lab):
                    data = data.to(device, img_type, non_blocking=True)
                    np: int = start_pos + mbs
                    target = utmt_TS.dlf_Labels[start_pos: np]
                    start_pos = np
                else:
                    data, target = data.to(device, img_type, non_blocking=True), target.to(device, non_blocking=True)
                # for k in range(0, len(trgt), 8): data, target = data0[k:k+8], trgt[k:k+8] # ResNet50
                # with autocast(device_type='cuda', dtype=tt.float16): # TODO autocasting.
                output: tt.Tensor = model(data) # .to(dtype=tt.float16)
                # test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr += pred.eq(target.view_as(pred)).sum() #.item() # ,int(utmt_TS_Xsize[i])
                test_loss += cost_function(output, target) * len(target)
                i += 1
    else: # tensors already in device-memory/GPU
        utmt_TS.DirectBatch(-1) # rewind
        with tt.no_grad():
            # for data, target in utmt_TS.dlf_listXY: # todo: iterator
            while 1:
                mbs, data, target = utmt_TS.DirectBatch(1024) # new
                if (mbs < 1): break
                output: tt.Tensor = model(data) # <class 'tt.Tensor'>
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr += pred.eq(target.view_as(pred)).sum() #.item()
                test_loss += cost_function(output, target) * mbs # both TestRuns together
                i += 1

    utmt_TS.dlf_EpochCount += 1 # Test
    corr = corr.item()
    print("TL=%.3fs," % (time_time() - dt), end=" ") # debug
    assert(corr > 0), "test accu > 0.0"
    return corr / len(test_loader.dataset), test_loss.item() / len(test_loader.dataset)

def FastenDataloader(train_data:DataLoader, test_data:DataLoader, maxMB:int=800, device=tt.device('cpu')) -> None:
    "Intial Load Dataset into CUDA/GPU for FullBatchOperation"
    global utmt_TS, utmt_DS
    dt: float = time_time()
    
    if (maxMB <= 0) or (train_data is None): # or (str(device).find("cuda") < 0):
        print("FastenDataloader(skip, dev=%s)." % (str(device)))
        return # dont skip here, even 2x faster on CPU !
    
    if 1: # bypass (cache) timm-DL to be faster
        print("FastenDataloader(%d+%d) .." % (len(train_data.dataset), len(test_data.dataset)))
        utmt_TS.Import_DataLoader(test_data, is_train=False, maxMB=950, device=device)
        if (utmt_TS.dlf_samples > 0): # both or none
            utmt_DS.Import_DataLoader(train_data, is_train=True, maxMB=1400, device=device) # DISABLE DS HERE
        dt = time_time() - dt
        print("FastenDataloader(%d+%d), dt=%.3fs.\n" % (utmt_DS.dlf_samples, utmt_TS.dlf_samples, dt))
        DataLdrFast.CheckPair(utmt_DS, utmt_TS) # FilePairConflicts
        # print(utmt_DS.GetDsHash(), utmt_TS.GetDsHash()) #;exit() # debug
        return
    
    print("FastenDataloader(%d+%d)=SKIP.\n" % (len(train_data.dataset), len(test_data.dataset)))
    return

def full_batch_loss(model, data_loader:DataLoader, cost_function, device) -> float:
    "calc. full batch loss (over all train data), only for printing (no effect on solver)"
    if (data_loader is None) or (len(data_loader.dataset) < 1):
        return -1.0
    global utmt_DS
    i: int = 0
    t0: float = time_time()
    # dc: int = tt.cuda.device_count()
    # accumulate loss over batches
    total_loss: float = 0.0
    count: int = len(data_loader.dataset)
    # loss = tt.zeros(1, dtype=tt.float32).to(device)
    corr = tt.zeros(1, dtype=tt.int64).to(device)

    if (len(data_loader.dataset) == utmt_DS.dlf_samples): # fast - train
        ls = tt.zeros(1, dtype=tt.float32).to(device)
        utmt_DS.DirectBatch(-1) # rewind
        with tt.no_grad():
            # for X, y in utmt_DS.dlf_listXY:
            while 1: # full_batch_loss
                mbs, X, y = utmt_DS.DirectBatch(1024) # new
                if (mbs < 1): break
                output: tt.Tensor = model(X)
                ls += cost_function(output, y) * mbs # int(utmt_DS.dlf_Xsize[i])
                pred: tt.Tensor = output.max(1, keepdim=True)[1]
                corr += pred.eq(y.view_as(pred)).sum()
                i += 1
            total_loss = ls.item()

    if (0 == i): # tensors not in device-memory/GPU
        img_type: tt.dtype = tt.get_default_dtype() # tt.float32
        flst = []
        count = 0
        t: float = time_time()
        with tt.no_grad():
            for X, y in data_loader: # X.device=cpu
                # X_size: int = X.size(0) # bs=256
                if (i <= 400) or (0 == (i & 3)): # mod 4, reduce time on large datasets (we spend 1/3 on full batch loss)
                    # batch_loss = cost_function(model(X.to(device, img_type, non_blocking=True)), y.to(device, non_blocking=True)).item()
                    # loss = loss_fn(outputs1, labels1).to('cuda:0') + loss_fn(outputs0, labels0)
                    y = y.to(device, non_blocking=True)
                    output: tt.Tensor = model(X.to(device, img_type, non_blocking=True))
                    batch_loss = cost_function(output, y).item()
                    flst.append(batch_loss)
                    pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                    corr += pred.eq(y.view_as(pred)).sum()
                    total_loss += batch_loss * X.size(0)
                    count += X.size(0)
                i += 1
                if ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    t = time_time()
        dev, avg = tt.std_mean(tt.tensor(flst, dtype=tt.float32))
        print("FBL(%.4g+%.3g)" % (avg.item(), dev.item()), end=" ")
            
    utmt_DS.dlf_EpochCount += 1 # Train
    t0 = time_time() - t0
    if (t0 > 50.0):
        print("sec=%.0f/%.1f%%, " % (t0, 100.0*count/len(data_loader.dataset)), end=" ")
    # print(len(data_loader.dataset), len(data_loader)); exit() # [256..,last=96], 60000 235
    return corr.item() / count, float(total_loss) / count

booster_on: bool = False
# for combined_X() - to be removed here
combiX_sum, combiX_wsum = None, None
combi_fx_min, combi_fx_sum, combi_wsum = 0.0, 0.0, 0.0
combi_cnt: int = 0
combiX_last: tt.Tensor = None
combiX_ldir: tt.Tensor = None

def combined_X(model, fval:float) -> None:
    "test only: combined x (param) for late improvement" # to be replaced by GetParamAvg()
    return # debug only
    global combiX_sum, combiX_wsum, combi_fx_min, combi_fx_sum, combi_fx_min, combi_wsum, combi_cnt
    if (model is None): # reset
        combiX_sum, combiX_wsum, combi_cnt = None, None, 0
        return
    
    x: tt.Tensor = GetParam(model)
    if (len(x)>>20 > 20) and ((tt.cuda.get_device_properties(0).total_memory >> 30) < 11): # dim>20mio AND vram<15GB
        return # skip both for ResNet50(24mio)
        
    combi_cnt += 1
    if (combiX_sum is None): # first
        combiX_sum = x.clone()
        combi_fx_min, combi_fx_sum = fval, fval
        if (len(x)>>20 < 15) and (fval > 1e-9): # optional
            combi_wsum = 1.0 / fval
            combiX_wsum = x * combi_wsum
        else:
            combi_wsum = 0.0
        return
    combi_fx_min = min(combi_fx_min, fval)
    combi_fx_sum += fval
    combiX_sum += x
    if (len(x)>>20 < 15) and (fval > 1e-9): # skip for ResNet34(11mio)+ResNet50(24mio)
        combi_wsum += 1.0 / fval
        combiX_wsum += x * (1.0 / fval)
    return

def cosine(a: tt.Tensor, b: tt.Tensor) -> float:
    "cosine between tensor pair (debug only)"
    assert(len(a) == len(b)), "dot-product dim-conflict"
    d: float = tt.dot(a, b)
    return 0.0 if (tt.abs(d) < 1e-50) else d / (tt.norm(a) * tt.norm(b))

def combined_FX(model, train_data, test_data, cost_function, device, optimizer, fb_loss:float) -> tuple[float,float,float,float]:
    "print f(x_average) 2x"
    if (not booster_on) or (not elra_solver): # (combi_cnt < 1):
        return -99.99, -1.0, -1.0, 0.0
        
    x0: tt.Tensor = GetParam(model) # backup
    loss1, loss2, dl, cos = -99.999, -99.999, -1.0, 0.0
    assert(len(x0) > 0), "combined_FX (empty model)"
    global combiX_last, combiX_ldir
    
    #if (len(x0) < 250000) and (combi_wsum > 0.0): # print only (debug)
    #    SetParam(model, combiX_wsum / combi_wsum)
    #    loss2 = full_batch_loss(model, train_data, cost_function, device)

    # avg: float = combi_fx_sum / combi_cnt
    # avg_x: tt.Tensor = combiX_sum * (1.0 / combi_cnt)
    _, avg, avg_x = optimizer.GetParamAvg(True)
    if (avg_x is None): return -99.99, -1.0, -1.0, 0.0
    SetParam(model, avg_x)
    _, loss1 = full_batch_loss(model, train_data, cost_function, device)
    dtype1 = tt.bfloat16 # bfloat16 works on CPU
    if (combiX_last is not None):
        avg_x = avg_x.to(dtype=dtype1)
        dx: tt.Tensor = avg_x - combiX_last
        dl = tt.norm(dx)
        if (combiX_ldir is not None): # dot/nn=cos
            cos = cosine(combiX_ldir, dx)
        combiX_ldir = dx
    combiX_last = None if (len(x0) > (12<<20)) else avg_x.to(dtype=dtype1)
    del avg_x
    
    test_accu, test_loss = -1.0, -1.0
    if (loss1 < fb_loss): # switch for combi-step-jump, use if benefitial
        if (test_data is not None):
            test_accu, test_loss = TestEval(model, test_data, cost_function, device) # anyway seen later
            print("combined_FX= %.6g & %.6g, (avg= %.4g > %.4g), dl=%.3g,cos=%.3f, (%.3f%% %.4f), good" %
                (loss1, loss2, avg, combi_fx_min, dl,cos, test_accu*100, test_loss))
        else:
            print("combined_FX= %.6g & %.6g, (avg= %.4g > %.4g), dl=%.3g,cos=%.3f, (-), good" %
                (loss1, loss2, avg, combi_fx_min, dl,cos))
        SetParam(model, x0) # optimizer.SetCombiX(avg_x, loss1)
    else:
        print("combined_FX= %.6g & %.6g, (avg= %.4g > %.4g), dl=%.3g,cos=%.3f, skip" %
            (loss1, loss2, avg, combi_fx_min, dl,cos))
        SetParam(model, x0) # restore
    return loss1, test_accu, test_loss, dl
   
prev_bs: int = 0 # previous batch size (to detect last batch)
hist_loss = tt.zeros(128, dtype=tt.float32, device=tt.device('cpu'))
hist_lpos: int = 0
hist_lsum: float = 0.0

def training_step(X:tt.tensor, y:tt.tensor, model:tt.nn.Module, loss_function, optimizer:tt.optim.Optimizer, RestSteps:int):
    "Training step for torch model."
    global prev_bs, hist_loss, hist_lpos, hist_lsum

    optimizer.zero_grad(set_to_none=True) # (optional: skip for batch-combining)

    # with autocast(): # run forward pass with autocast (mixed precision)
    loss = loss_function(model(X), y) # we use this f(x) for all f()
    loss_item: float = loss.item() # already fix
    
    if (isinstance(optimizer, SelfConstOptimTorch)): # P2M+C2M
        if True: # test
            pos: int = hist_lpos & 127 # mod 128
            hist_lsum += loss_item - hist_loss[pos]
            hist_loss[pos] = loss_item
            hist_lpos += 1
        if optimizer.revert_step(loss_item): # TODO (Nov-2023) allow for gradient skip for revert-steps (P2M)
            optimizer.step(loss_item)
        else:
            if (RestSteps > 0) or (len(y) >= prev_bs): # last batch needs same size (else ignore)
                # combined_X(model, loss_item)
                loss.backward() # computes gradient for P2M+C2M
                optimizer.step(loss_item) #, batch=X, y=y) # P2M+C2M (X+y unused)
            else:
                print("(SkipLastBatch, bs=%d<%d, f=%.6f)" % (len(y), prev_bs, loss_item), flush=True) # skip last+short batch
            # HINT: "Pytorch: Working with Unscaled Gradients" is recommended for P2M+C2M
        prev_bs = len(y)
    else:
        loss.backward() # computes gradient for normal opt (Adam etc)
        optimizer.step()

    return loss_item


def train_epoch(dataloader, model, optimizer, cost_function, device):
    "train single epoch"
    losses, batches = [], []
    steps = []
    f_calls, g_calls = [], []
    last_time: float = -1.0
    tl_tmp: str = ""

    global booster_on, utmt_DS
    if (elra_solver): optimizer.GetParamAvg(booster_on) # reset
    # combined_X(None, 0.0) # clean
    # utmt_DS.FullShuffle(0) # todo:test
    # print(utmt_DS.GetDsHash()) # check data integrity by sums
    # utmt_DS.SetMiniBatchSize(0, shuffle=True) # reorder batches
    # utmt_DS.ShuffleBatch(rewind=True)
    
    batch_idx: int = 0
    batch_max: int = len(dataloader) - 1 # 0..max
    # if (not elra_solver): model.train()
    img_type: tt.dtype = tt.get_default_dtype() # tt.float32
    # ldf = open("path_dom.dat", "a")

    for batch_idx, (data, target) in enumerate(dataloader): # old+ok

        loss: float = training_step(data.to(device, img_type), target.to(device), model, cost_function, optimizer, batch_max - batch_idx)
        #if (99 == batch_idx % 100):
        #    Interpolx(model, cost_function, dataloader)
        #else:
        #    Interpolx(model, None, None)
        # WriteDominant(GetParam(model), log=ldf, t=batch_idx) # test (slow)
    # while 1: # new Shuffler-Loop
        #mbs, data, target = utmt_DS.ShuffleBatch() # new
        #if (mbs < 1): break
        #loss: float = training_step(data, target, model, cost_function, optimizer) # new

        norm100: float = 100.0 / len(dataloader)
        # print("** %d %.4e" % (batch_idx, loss)) # 1st epoch print(step,loss)
            
        if 1: # len(data) != len(dataloader.dataset): # why
            batch : int = batch_idx + 1
            tnow : float = time_time()
            dt : float = tnow - last_time
            if (dt >= 5.0): # progress print interval
                if "" == tl_tmp:
                    print('Epoch progress: [%d/%d(%.0f%%)]\tBatch avg. train loss:\t%.6g' %
                        (batch * len(data), len(dataloader.dataset),
                            batch * norm100, loss) )
                else:
                    tl_tmp = (" %.6g" % loss) # += !
                    print('Epoch progress: [%.1f%%]\tBatches train losses: %s' %
                        (batch * norm100, tl_tmp) )
                last_time = tnow
                tl_tmp = ""
            else:
                tl_tmp += ("%.3g," % loss)
        losses.append(loss)
        batches.append(batch_idx)

        if False and (type(optimizer) is not SelfConstOptimTorch): # needed ?
            optimizer.state["o_calls"] += 1
            optimizer.state["f_calls"] += 1
            optimizer.state["g_calls"] += 1

        step: int = optimizer.state["o_calls"]
        # f, g = optimizer.state["f_calls"], optimizer.state["g_calls"]

        steps.append(step)
        f_calls.append(optimizer.state["f_calls"])
        g_calls.append(optimizer.state["g_calls"])

        if loss == float('inf'):
            break
    
    optimizer.zero_grad(set_to_none=True) # less memory during test full-batch
    # if (not elra_solver): model.eval() # issue: batch normalize + booster_on
    # WriteDsHist(True, ldf)
    # ldf.close()
    return losses, batches, steps, f_calls, g_calls


def train(train_data:DataLoader, test_data:DataLoader, 
          model, cost_function, optimizer, max_epochs:int = 1000, target_loss:float = 0.0, 
          batch_size:int = 999999999, device = tt.device("cpu"), log = None):
    "train model with train_data"
    batch_size = batch_size if batch_size < 999999999 else len(X) # (9x9) = inf-like float('inf')
    global utmt_TS, utmt_DS, booster_on
    tt.set_printoptions(precision=4, linewidth=150)

    losses, batches = [], []
    epochs = []
    types, steps = [], []
    f_calls, g_calls = [], []
    test_loss_min: float = float('inf')

    epoch: int = 1

    optimizer.state["o_calls"] = 0
    optimizer.state["f_calls"] = 0
    optimizer.state["g_calls"] = 0

    CheckElraSolver(optimizer)
    ParamLoad(model, fn="", nofilewarn=False) # "startx_000.pt"
    FastenDataloader(train_data, test_data, maxMB = 800, device = device) # here DL switch-off
    # if (not elra_solver): model.eval() # E.g. dropout layers will be disabled during evaluation and batchnorm layers will use the running stats instead of the batch statistics to normalize the activation. The gradient computation will not be changed or disabled. !!
    if (tt.cuda.is_available()):
        print("torch.cuda.MB:", tt.cuda.memory_allocated()>>20, tt.cuda.memory_reserved()>>20)
    
    # Initial Training loss auf allen Trainings Daten, F(parameter satz)=Loss
    _, loss = full_batch_loss(model, train_data, cost_function, device)
    print(datetime2.now().strftime("[%H:%M:%S]"), end=" ")
    print('Start training: \t\tInit. avg. train loss:\t%.6f' % (loss), flush=True)
    loss0: float = loss
    
    losses.append(loss)
    batches.append(None)
    epochs.append(0)
    types.append("train")
    steps.append(0)
    f_calls.append(0)
    g_calls.append(0)

    # last_epoch_index: int = 0
    # log = open("history.txt", "a")

    while loss > target_loss and epoch <= max_epochs and loss < abs(loss0)*3: # stop criteria
        dt1: float = time_time()
        epoch_losses, epoch_batches, step, fs, gs = train_epoch(train_data, model, optimizer, cost_function, device)
        dt1 = time_time() - dt1
        mean_hist_loss: float = tt.mean(hist_loss).item()
        if (dt1 > 15.0): print("(end epoch %d, f128=%.3f, dt=%.1fs)" % (epoch, mean_hist_loss, dt1), flush=True)
        
        losses.extend(epoch_losses)
        batches.extend(epoch_batches)
        epochs.extend(len(epoch_losses) * [epoch])
        types.extend(len(epoch_losses) * ["batch"])
        steps.extend(step)
        f_calls.extend(fs)
        g_calls.extend(gs)

        if float('inf') in epoch_losses:
            break
        # F(nach epoch)
        dt2: float = time_time()
        # model_c2 = copy.deepcopy(model)
        accu, loss = full_batch_loss(model, train_data, cost_function, device)
        cx1,cx2,cx3,cx4 = combined_FX(model, train_data, test_data, cost_function, device, optimizer, loss) # new
        dt2 = time_time() - dt2
        if (dt2 > 100.0): print(datetime2.now().strftime("[%H:%M:%S]"), end=" ", flush=True)
        print('Finished epoch %d: \t\tFinal avg. train loss:\t%.6f (%.3f,ac=%.2f%%)' % (epoch, loss, mean_hist_loss, accu*100))
        if (dt2 > 900.0): ParamSave(model, fn="epoch_tmp.pt")

        if False: # still needed ?
            losses.append(loss)
            batches.append(None)
            epochs.append(epoch)
            types.append("train")
            steps.append(step[-1])
            f_calls.append(fs[-1])
            g_calls.append(gs[-1])

        if test_data is not None:
            # test_loss : float = full_batch_loss(model, test_data, cost_function, device)
            test_accu, test_loss = TestEval(model, test_data, cost_function, device)
            test_loss_min = min(test_loss_min, test_loss)
            print("Test set: Average loss: %.4f(>%.4f), accu: %.3f%%, dt=%.3f+%.3fs\n" % (test_loss, test_loss_min, test_accu*100,dt1,dt2), flush=True)
            if log is not None:
                log.write("%d,%.4g,%.4f,%.6g,(%.4g:%.4f:%.4g:%.3g),%s,%s\n" %
                (epoch, loss,test_accu,test_loss, cx1,cx2,cx3,cx4, StatListStr(epoch_losses),gstat.statist_GetStr()))
                if (dt2 > 55.0): log.flush()
            # ?? how to get alpha + beta here ??

            losses.append(test_loss)
            batches.append(None)
            epochs.append(epoch)
            types.append("test")
            steps.append(step[-1])
            f_calls.append(fs[-1])
            g_calls.append(gs[-1])

        #if losses[-1] >= losses[last_epoch_index] - losses[last_epoch_index] * 0.01: # speed up benchmarks and break when no improvement compared to previous test loss is seen
        #    break

        #if('converged' in optimizer.state and optimizer.state['converged']): # use p2min / c2min converged flag
        #    break

        # last_epoch_index: int = len(losses) - 1
        
        if (epoch == 1): # decide once
            n: int = sum(p.numel() for p in model.parameters())
            booster_on = ((n>>20) < 20) or ((tt.cuda.get_device_properties(0).total_memory >> 30) > 10) # low-dim or high-gpu-ram
        
        if (tt.cuda.device_count() > 1): sleep(1.5) # relax system due to multi-gpu issue

        epoch += 1
    
    ParamSave(model) # for later reuse
    
    # Release Tensor Memory
    utmt_TS.InitVectors()
    utmt_DS.InitVectors()
    # log.close()
    return losses, batches, epochs, types, steps, f_calls, g_calls

# EoF.
