# DataLoaderFast.py (2024)
# includes own DataLoader for constant Data (test + full-batch)

import numpy as np # only: class DataLdrFast
import torch as tt
from torch.utils.data import DataLoader # TensorDataset


class DataLdrFast:
    dlf_Instances: int = 0
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
        self.dlf_label_max = -1 # Classes - 1
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
            import glob
            from os import remove
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
        # self.dlf_mb_size = 0 # keep gpu-batch-size
        # self.dlf_label_max = -1 # keep classes/labels
        self.dlf_EpochCount = 0
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
            return (0, None, None)
        if (not elem): elem = self.dlf_init_mbs # keep from loaded source (elem==0)
        if (elem >= self.dlf_samples):
            self.dlf_pos_fast = self.dlf_samples
            return (self.dlf_samples, self.dlf_Images, self.dlf_Labels)
        if (elem % 8) and (not self.dlf_pos_fast):
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
        if (device is tt.device('cpu')):
            self.dlf_Images = self.dlf_Images.to(dtype=img_type)
            self.dlf_Labels = self.dlf_Labels.to(dtype=tt.uint8) # int16 + uint8
            return

        self.dlf_Images = self.dlf_Images.to(device) # .to(tt.bfloat16)
        self.dlf_Labels = self.dlf_Labels.to(dtype=tt.uint8).to(device)
        self.dlf_Images = self.dlf_Images.to(dtype=img_type) # here or at usage (memory vs runtime)
        # print(self.dlf_samples, self.dlf_Images.device, self.dlf_Labels.device); exit()
        # print(self.dlf_samples, self.dlf_Images.dtype, self.dlf_Labels.dtype); exit()

    def ImportFromFile(self, fn: str, device=tt.device('cpu')) -> int:
        from os.path import isfile
        if (len(fn) < 2) or (not isfile(fn)):
            return -1
        dsnum, self.dlf_Images, self.dlf_Labels = tuple(tt.load(fn, weights_only=True))
        assert(len(self.dlf_Images) == len(self.dlf_Labels)), "ImageCount <> LabelCount"
        self._list2props( dsnum.tolist() )
        assert(self.dlf_samples == len(self.dlf_Labels)), "SampleCountVar <> LabelCount"
        pix_sz: int = self.dlf_Images.element_size()
        self.dlf_sample_elem = tt.numel(self.dlf_Images[0])
        i: int = (self.dlf_sample_elem * self.dlf_samples * pix_sz) >> 20 # .bfloat16
        print("DatasetFromCache(n=%d, cls=%d, lab[%s], es=%d+%d, sz=%dMB)." % (self.dlf_samples, self.dlf_label_max+1, self.GetLabelStr(8), pix_sz, self.dlf_Labels.element_size(), i)) # optional
        self._dataset2device(device)
        return 0 # used RAM [MB]

    def Import_DataLoader(self, data:DataLoader, num_classes:int, is_train:bool=False,\
            maxMB:int=600, device=tt.device('cpu')) -> int:
        if (maxMB <= 0) or (not isinstance(data, DataLoader)) or (len(data) < 1):
            return -1
        from os.path import isfile
        maxMB = 600 # test
        dsfn: str = DataLdrFast.CacheFilename(data) + ".pt"
        dlf_listXY = [] # was self.
        self.dlf_samples = len(data.dataset) # 60K
        self.dlf_init_mbs = DataLdrFast.EstimBatchSize(data)
        self.dlf_is_train = is_train
        i: int = 0
        max_byte:  int = maxMB << 20
        size_byte: int = 0
        max_lab: int = -1 # max(labels)
        classes: int = num_classes # len(data.dataset.classes) if hasattr(data.dataset, "classes") else 1
        self.dlf_label_max = classes - 1
        assert(self.dlf_label_max > 0), "no labels"
        np.random.seed(123) # todo

        if (len(dsfn) > 1) and isfile(dsfn):
            # max_lab = self.dlf_label_max
            self.ImportFromFile(dsfn, device=device)
            if (self.dlf_samples <= 20000): # (not is_train):
                maxl: int = -1
                for _, y in data: maxl = max(maxl, y.max().item())
                assert(maxl == self.dlf_label_max), "Bad DataLoaderCacheFile! (delete ds_*.tmp.pt)"
            dsfn = "" # no save after load
            return 0 # loaded from file cache

        if (self.dlf_samples == 81167) or (self.dlf_samples >= 1200000):
            self.dlf_label_max = 1000 - 1 # detect ImageNet1k
            self.dlf_samples = 0 # mark as unused
            return 0

        dlf_Xsize = tt.zeros([len(data)], dtype=tt.int32)
        for X, y in data:
            size_byte += tt.numel(X) * X.element_size() # + tt.numel(y) * y.element_size()
            # print("X0:", tt.numel(X), X.element_size(), X.size(0), X.shape); exit(0) # X0: 2408448 4(float32) 16(bs) [16, 3, 224, 224]
            dlf_Xsize[i] = X.size(0); i += 1
            max_lab = max(max_lab, y.max().item()) # y.dtype==int64
            if (size_byte <= max_byte):
                y2 = y.to(dtype=tt.int16, copy=True) # int16 + uint8
                X2 = X.to(dtype=self.dlf_ImgType, copy=True) # optional/halfing
                dlf_listXY.append( (X2.to(device), y2) )
            else:
                if (max_lab < (1<<14)): # 15-bit limit below
                    dlf_listXY.append( (None, y.to(dtype=tt.int16, copy=True)) )
                else:
                    dlf_listXY.append( (None, None) ) # e.g. ImageNet1k (>8 bit)

        assert(self.dlf_label_max == int(max_lab)), "DataLoader not consistent"
        assert(max_lab > 0), "no labels"
        # print("Warn: No Labels in Dataset (n=%d, c=%d) !" % (self.dlf_samples, classes))
        if (size_byte > max_byte):
            print("Warn: %d MB > %d MB, c=%d, skip=%d!" % (size_byte>>20, max_byte>>20, sum(dlf_Xsize), i))
            if (is_train):
                self.ClearData()
                return -5
            else:
                self.dlf_only_lab = True # only test-data (no shuffle)

        if (len(dlf_listXY) != len(dlf_Xsize)): # Error:IDL(50<>391,i=391)!
            print("Error:IDL(%d<>%d,i=%d)!" % (len(dlf_listXY), len(dlf_Xsize), i)); exit()

        self.dlf_sample_elem = tt.numel(X[0])
        assert(max_lab > 0), "No/empty Labels!"
        self.dlf_init_mbs = int(dlf_Xsize[0]) # overwrite estimate
        xshape: list[int] = list(X.shape)
        xshape[0] = 0 if (self.dlf_only_lab) else self.dlf_samples

        if (self.dlf_label_max >= 1<<16): # todo: use int16 (danger) then
            print("Error:IDL(n=%d,classes=%d>U16)!" % (self.dlf_samples, self.dlf_label_max))
            self.ClearData()
            self.dlf_only_lab = False
            return -2

        pos: int = 0
        self.dlf_Labels = tt.zeros([self.dlf_samples], dtype=tt.int16, device=tt.device('cpu')) # uint8 + int16

        if (self.dlf_only_lab):
            for _, y in dlf_listXY:
                nxt: int = min(self.dlf_samples, pos + len(y))
                self.dlf_Labels[pos: nxt] = y
                pos = nxt
            print("DLF:only_labels, s=%d, c=%d" % (self.dlf_samples, max_lab+1))
            self.dlf_Images = tt.zeros(0)
            self.dlf_Labels = self.dlf_Labels.to(dtype=tt.int16) # caution: model needs 8+64
            self.dlf_samples = 0
            return 0

        self.dlf_Images = tt.zeros(xshape, dtype=self.dlf_ImgType) # .bfloat16, [1024,1,8,8]
        for X, y in dlf_listXY:
            nxt: int = min(self.dlf_samples, pos + len(y))
            # if (not self.dlf_only_lab):
            self.dlf_Images[pos: nxt] = X.to(dtype=self.dlf_ImgType) # .bfloat16
            self.dlf_Labels[pos: nxt] = y
            pos = nxt
        del dlf_listXY
        self.dlf_Labels = self.dlf_Labels.to(dtype=tt.uint8) # int16 issue

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
        if not (self.dlf_ShuffleCnt % 2): return # skip epochs
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

# EoF.
