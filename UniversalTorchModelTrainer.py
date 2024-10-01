# UniversalTorchModelTrainer.py (2024)
# (included own DataLoader for constant Data)

import torch as tt
from torch.utils.data import DataLoader # TensorDataset
# from torch.nn.parallel import DataParallel # Multi-GPU (new)
# from torch.cuda.amp import autocast, GradScaler
from Cos2MinTorchFunctionOptimizer import ElraOptimizer # ELRA_class.py (tbd 2024)
from time import time as time_time
# from os import path # remove, replace, (only path.isfile)
# import glob, pickle, copy
from math import inf # isnan

from DataLoaderFast import DataLdrFast
# class DataLdrFast: was here

class PandasDataFrame:
    def __init__(self):
        self.losses = []
        self.batches = []
        self.epochs = []
        self.types = []
        self.steps = []
        self.f_calls = []
        self.g_calls = []
        return

    def AppendFrame(self, loss:float, _bt, ep:int, tstr:str, stp:int, fs:int, gs:int) -> None:
        self.losses.append(loss)
        self.batches.append(None)
        self.epochs.append(ep)
        self.types.append(tstr)
        self.steps.append(stp)
        self.f_calls.append(fs)
        self.g_calls.append(gs)
        return

    def ExtendFrame(self, epoch_losses:list[float], epoch_batches:list[int], ep:int, _tstr:str, stp:int, fs:int, gs:int) -> None:
        self.losses.extend(epoch_losses)
        self.batches.extend(epoch_batches)
        self.epochs.extend(len(epoch_losses) * [ep])
        self.types.extend(len(epoch_losses) * ["batch"]) # _tstr
        self.steps.extend(stp)
        self.f_calls.extend(fs)
        self.g_calls.extend(gs)
        return

    def ReturnFrame(self):
        return self.losses, self.batches, self.epochs, self.types, self.steps, self.f_calls, self.g_calls

def CopyLgsFile() -> None:
    "backup file once per epoch"
    from shutil import copyfile
    from os.path import isfile
    fn: str = "state_lgs.txt"
    if isfile(fn):
        copyfile(fn, "epoch_lgs.txt")

def GetGradMax(model: tt.nn.Module) -> float:
    "inf-norm of gradients"
    # if (model is None): return -1.0
    nrm: tt.Tensor = tt.zeros(1, device=next(model.parameters()).device)
    for param in model.parameters():
        if (param.grad.data is not None):
            nrm = tt.max(nrm, tt.linalg.norm(param.grad.data.view(-1), ord=inf)) # max(abs(x))
    return nrm.item()

def GetParamxMax(model: tt.nn.Module, device) -> tuple[float, float]:
    "inf-norm of x (param)"
    if (model is None): return -1.0
    nrm2: tt.Tensor = tt.zeros(1, device=device)
    nrmM: tt.Tensor = tt.zeros(1, device=device)

    for param in model.parameters():
        pdv = param.data.view(-1)
        nrm2 += tt.square(tt.linalg.norm(pdv, ord=2.0))
        nrmM = tt.max(nrmM, tt.linalg.norm(pdv, ord=inf)) # max(abs(x))
    return tt.sqrt(nrm2).item(), nrmM.item()

#def FreezeBatchNorm(model: tt.nn.Module) -> None:
#    "unused", # model = ResNet50()
#    for m in model.modules():
#        if isinstance(m, tt.nn.BatchNorm2d):
#            m.eval()
#    return

def LimitBatchNorm(model: tt.nn.Module, mean_max: float) -> None:
    "Limit BatchNorm"
    c: int = 0
    i: int = 0
    e: int = 0
    maxn: float = 0.0
    # assert(mean_max > 0.0), "positive"

    for m in model.modules():
        if isinstance(m, tt.nn.BatchNorm2d):
            mn: float = tt.linalg.vector_norm(m.running_mean, ord=inf).item()
            if (mn < 1e38): maxn = max(mn, maxn) # nan/inf issue
            else: e += 1
            if (mn > mean_max) and (mean_max >= 1.0):
                mi, ma = m.running_mean.min().item(), m.running_mean.max().item()
                vmin, vmax = m.running_var.min().item(), m.running_var.max().item()
                c += 1
                tt.clamp(m.running_mean, min=-mean_max, max=mean_max, out=m.running_mean)
                print("BNL(%d,%.3g<%.3g,%.3g<%.3g)" % (i, mi,ma, vmin,vmax), end=" ")
            else:
                if (mn > 100.0) and (mean_max < 1.0): print("[%d,%.2g]" % (i, mn), end="")
        i += 1

    if (mean_max < 1.0) or (c > 0): print("LimitBatchNorm(%d/%d, n<%.3g, e=%d)" % (c, i, maxn, e))
    return

def ExportBatchNorm(model: tt.nn.Module) -> list[tt.Tensor]:
    "Export BatchNorm to Tensor-List"
    c: int = 0
    bnl = []

    for m in model.modules():
        if isinstance(m, tt.nn.BatchNorm2d):
           c += len(m.running_mean)
           bnl.append(m.running_mean.clone())
           bnl.append(m.running_var.clone())

    print("ExportBatchNorm(lay=%d, nums=%d)" % (len(bnl), c * 2))
    return bnl

def ImportBatchNorm(model: tt.nn.Module, bn_list) -> None:
    "Import BatchNorm from Tensor-List"
    if (bn_list is None) or (len(bn_list) < 1):
        for m in model.modules():
            if isinstance(m, tt.nn.BatchNorm2d):
                m.reset_running_stats()
    else:
        j: int = 0
        for m in model.modules():
            if isinstance(m, tt.nn.BatchNorm2d):
                assert(len(m.running_mean) == len(bn_list[j])), "wrong Tensor size"
                m.running_mean = bn_list[j + 0].clone()
                m.running_var  = bn_list[j + 1].clone()
                j += 2
        assert(len(bn_list) == j), "wrong list size"
    return

#def ResetBatchNorm(model: tt.nn.Module) -> None:
#    "Reset BatchNorm (recover from NaN)"
#    i, j = 0, 0
#    for m in model.modules():
#        if isinstance(m, tt.nn.BatchNorm2d):
#           print("BN(%d:%d,mn=%.6f,vn=%.6f), " % (j, len(m.running_mean), m.running_mean.norm().item(), m.running_var.norm().item()), end="")
#           m.reset_running_stats()
#           j += 1
#        i += 1
#    print(" BN.done=%d/%d" % (j, i))
#    # net.load_state_dict(copy_net.state_dict())

#@tt.no_grad()
#def get_wd_params(model: tt.nn.Module) -> None:
#    "unused"
#    # Parameters must have a defined order.
#    # No sets or dictionary iterations.
#    # See https://pytorch.org/docs/stable/optim.html#base-class
#    # Parameters for weight decay.
#    # all_params = tuple(model.parameters())
#    i, j = 0, 0
#    wd_params = [] # list()
#    for m in model.modules():
#        if isinstance( m, (
#                        tt.nn.Linear,
#                        tt.nn.Conv1d, tt.nn.Conv2d, tt.nn.Conv3d,
#                        tt.nn.ConvTranspose1d, tt.nn.ConvTranspose2d, tt.nn.ConvTranspose3d,
#            ),
#        ):
#            # wd_params.append(m.weight)
#            j += 1
#        i += 1
#    print("(WeightDecay: %d/%d)" % (j, i))
#    return

def SetParam(model, par: tt.Tensor) -> int:
    "x-tensor into model"
    assert(len(par) > 0), "empty Tensor"
    model.train() # needed if net uses BatchNorm
    # normalisation layers use per-batch statistics + (activates Dropout layers)
    s = e = 0
    for p in model.parameters():
        e += tt.numel(p)
        p.data = tt.reshape(par[s:e], p.size()).to(p.device, p.data.dtype)
        s = e
    return e # n=dim

def ParamLoad(model, fn:str="", nofilewarn:bool=True) -> bool:
    "load x vector from disk" # TODO
    n: int = sum(p.numel() for p in model.parameters())
    assert(n >= 1), "empty model"
    from os.path import isfile

    if (len(fn) < 2):
        fn = "startx_*.pt"
    fn = fn.replace('*', str(int(n)))
    if not isfile(fn):
        if nofilewarn:
            print("ParamLoad(n=%d:%s)=NoFile." % (n, fn))
        return False
    # tt.load(model, 'path')
    par: tt.Tensor = tt.load(fn, weights_only=True) # model.load_state_dict()
    assert(len(par) == n), "wrong dimension"

    print("########################################")
    SetParam(model, par)
    pn: float = tt.linalg.vector_norm(par).item()
    print("## ParamLoad(n=%d,av=%.3e,n2=%.3e)=OK" % (n, tt.mean(par), pn))
    print("########################################")
    return True

def GetParam(model) -> tt.Tensor:
    "model to x-tensor"
    # params = []
    # for p in model.parameters(): params.append( p.data.view(-1) )
    params = [param.data.view(-1) for param in model.parameters()]
    return tt.cat(params)

def ParamSave(model, fn:str="") -> bool:
    "store x vector on disk" # tested
    if (model is None):
        print("Warn:ParamSave(model=None), skip!")
        return False
    # tt.save(model, 'path')
    n: int = sum(p.numel() for p in model.parameters())
    # for p in model.parameters(): n += tt.numel(p)
    assert(n >= 1), "empty model"
    from os.path import isfile

    if (len(fn) < 2):
        fn = "lastx_*.pt"
    fn = fn.replace('*', str(int(n)))
    # if (path.isfile(fn)): print("ParamSave.overwrite(%s)" % fn)

    par: tt.Tensor = GetParam(model)
    tt.save(par, fn)

    pn: float = tt.linalg.vector_norm(par).item()
    print("ParamSave(%s,av=%.3e,n2=%.3e)=OK" % (fn, tt.mean(par), pn))
    return isfile(fn)

class MyGradScaler:
    "private GradScaler"
    def __init__(self, init_scale: float = 2.0**15, enabled:bool = True) -> None:
        self.good_cnt: int = 0
        self.dead_cnt: int = 0
        self.maxscale: float = 1.0
        self.scaling: float = float(init_scale)
        assert(self.scaling >= 1.0), "positive (1..1e6)"
        self.inv_scaling: float = 1.0 / self.scaling
        assert(enabled), "off = not implemented now"
        # self.enabled: bool = enabled

    def get_scale(self) -> float:
        "get actual scaling"
        return self.scaling

    def UpdateGradNorm(self, absmax: float) -> float:
        "update scale and return actual value"
        # absmax: float = GetGradMax(model)
        if (absmax < 1e999): # isinf+isnan
            if (absmax > self.maxscale):
                self.maxscale = absmax
            if (self.good_cnt < 100):
                self.good_cnt += 1
                return self.scaling # default way
            else:
                ret: float = self.scaling
                # print("UpdateGradNorm.upd, max=%.3g, scl=%.3g" % (self.maxscale, self.scaling))
                if (self.maxscale <= 65504 * 0.3):
                    self.scaling = min(2.0 * self.scaling, 65504.0) # float16max=65504
                    self.inv_scaling = 1.0 / self.scaling
                self.maxscale *= 0.5 # not perfect
                self.good_cnt = 0
                return ret
        else:
            print("UpdateGradNorm.Down, max=%.3g:%.3g, scl=%.3g" % (self.maxscale, absmax, self.scaling))
            self.scaling *= 0.5
            self.inv_scaling *= 2.0
            assert(self.scaling > 0.1), "zero scaling"
            self.maxscale = (self.maxscale * 0.5) if (self.maxscale < inf) else 0.0
            self.good_cnt = 0
            self.dead_cnt += 1
            return 0.0 # inf/nan in gradient

model_boost = None
last_x: tt.Tensor = None # debug only
diff_sum: tt.Tensor = None # debug only
elra_solver: bool = True
dog_solver: bool = None # DoG+LDoG
dog_averager = None # DoG+LDoG

# Creates a GradScaler once at the beginning of training.
scaler = None # GradScaler()

def CheckElraSolver(optim) -> bool:
    "verify ELRA class"
    global elra_solver, dog_solver
    assert(optim is not None), "input <> None"

    elra_solver = hasattr(optim, 'GetParamAvg')
    name: str = str( type (optim).__name__ )
    print("ELRA:", elra_solver, ", name=", name) # ElraOptimizer
    if (name.find('DoG') >= 0):
        dog_solver = True
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
        loss, _, _, _ = full_batch_loss(model, None, dl, cost_function, x0.device)
        d = 0.0 if (not i) else loss-last
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
        import numpy as np
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

def NormalEval_Labels(model, model2, data_loader:DataLoader, cost_func, device, labels) -> tuple[int,int, float,float]:
    "full batch via normal DataLoader + single Label tensor (half D2H-copy)"
    i: int = 0
    prints: bool = False
    img_type: tt.dtype = tt.get_default_dtype() # tt.float32
    corr1 = tt.zeros(1, dtype=tt.int64).to(device)
    corr2i: int = 0
    loss1 = tt.zeros(1, dtype=tt.float32).to(device)
    loss2 = 0.0
    count: int = len(data_loader.dataset)
    t: float = time_time()
    bs = data1 = None
    # print("NormalEval_Labels ===== ", len(labels), count)
    assert(len(labels) == count), "sample counts differ"
    pos: int = 0
    targets: tt.Tensor = labels.to(device) # 0..5 MB (int16 here)
    targets = targets.to(dtype=tt.int64) # optional uint8 (if classes<256)

    if (model2 is None):
        with tt.no_grad():
            for data0, target0 in data_loader:
                data = data1
                data1 = data0.to(device, img_type, non_blocking=True)
                if (bs is None):
                    bs = len(target0)
                    continue

                pos2: int = pos + bs
                target = targets[pos : pos2]
                pos = pos2

                # ResNet50
                # with autocast(device_type='cuda', dtype=tt.float16): # autocasting.
                output: tt.Tensor = model(data)
                loss1 += cost_func(output, target) # float16
                # test_loss += F.nll_loss(output, target, reduction="sum")
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).sum()

                i += 1
                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs # len(target)

            target = targets[pos : ]
            if len(target): # tail
                c1, l1 = single_eval(data1, target, model, cost_func)
                corr1 += c1; loss1 += l1 * len(target)
    else:
        corr2 = tt.zeros(1, dtype=tt.int64).to(device)
        loss2 = tt.zeros(1, dtype=tt.float32).to(device)

        with tt.no_grad():
            for data0, target0 in data_loader:
                data = data1
                data1 = data0.to(device, img_type, non_blocking=True)
                if (bs is None):
                    bs = len(target0)
                    continue

                pos2: int = pos + bs
                target = targets[pos : pos2]
                pos = pos2

                output: tt.Tensor = model(data)
                loss1 += cost_func(output, target)
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).sum()

                # + model2
                output = model2(data)
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr2 += pred.eq(target.view_as(pred)).sum()
                loss2 += cost_func(output, target) # len(target)

                i += 1
                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs # len(target)
            loss2 *= bs # len(target)

            target = targets[pos : ]
            if len(target): # tail
                c1, l1 = single_eval(data1, target, model, cost_func)
                corr1 += c1; loss1 += l1 * len(target)

                c1, l1 = single_eval(data1, target, model2, cost_func)
                corr2 += c1; loss2 += l1 * len(target)

        corr2i, loss2 = corr2.item(), loss2.item()

    if (prints):
        avg: float = -1.0 if (count < 1) else loss1.item() / count
        print(".\n..FBL(%.4g+%.3g-%d)" % (avg, -1.0, 0), end=" ")
    # print("NormalEval, inf-errors=", err, count)

    return corr1.item(), corr2i, loss1.item(), loss2

def NormalEval(model, model2, data_loader:DataLoader, cost_func, device) -> tuple[int,int, float,float]:
    "full batch via normal DataLoader"
    i: int = 0
    prints: bool = False
    img_type: tt.dtype = tt.get_default_dtype() # tt.float32
    corr1 = tt.zeros(1, dtype=tt.int64).to(device)
    corr2i: int = 0
    loss1 = tt.zeros(1, dtype=tt.float32).to(device)
    loss2 = 0.0
    count: int = len(data_loader.dataset)
    t: float = time_time()
    bs = None
    data1 = target1 = None

    if (model2 is None):
        with tt.no_grad():
            for data0, target0 in data_loader:
                data, target = data1, target1
                data1, target1 = data0.to(device, img_type, non_blocking=True), \
                    target0.to(device, non_blocking=True)
                if (bs is None):
                    bs = len(target0)
                    continue

                # ResNet50
                # with autocast(device_type='cuda', dtype=tt.float16): # autocasting.
                output: tt.Tensor = model(data)
                loss1 += cost_func(output, target) # float16
                # test_loss += F.nll_loss(output, target, reduction="sum")
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).sum()

                i += 1
                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs # len(target)

            if len(target1): # tail
                c1, l1 = single_eval(data1, target1, model, cost_func)
                corr1 += c1; loss1 += l1 * len(target1)
    else:
        corr2 = tt.zeros(1, dtype=tt.int64).to(device)
        loss2 = tt.zeros(1, dtype=tt.float32).to(device)

        with tt.no_grad():
            for data0, target0 in data_loader:
                data, target = data1, target1
                data1, target1 = data0.to(device, img_type, non_blocking=True), \
                    target0.to(device, non_blocking=True)
                if (bs is None):
                    bs = len(target0)
                    continue

                output: tt.Tensor = model(data)
                loss1 += cost_func(output, target)
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).sum()
                # loss1 += cf # len(target)

                # + model2
                output = model2(data)
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr2 += pred.eq(target.view_as(pred)).sum()
                loss2 += cost_func(output, target) # len(target)

                i += 1
                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs # len(target)
            loss2 *= bs # len(target)

            if len(target1): # tail
                c1, l1 = single_eval(data1, target1, model, cost_func)
                corr1 += c1; loss1 += l1 * len(target1)

                c1, l1 = single_eval(data1, target1, model2, cost_func)
                corr2 += c1; loss2 += l1 * len(target1)

        corr2i, loss2 = corr2.item(), loss2.item()

    if (prints):
        avg: float = -1.0 if (count < 1) else loss1.item() / count
        print(".\n..FBL(%.4g+%.3g-%d)" % (avg, -1.0, 0), end=" ")
    # print("NormalEval, inf-errors=", err, count)

    return corr1.item(), corr2i, loss1.item(), loss2

def FastCacheEval(model, model2, fdl: DataLdrFast, cost_function, device) -> tuple[int,int, float,float]:
    "full batch from fast data-loader-cache"
    count: int = fdl.dlf_samples # len(data_loader.dataset)
    assert(count > 0), "DataLdrFast available"
    assert(len(fdl.dlf_Images) > 0), "DataLdrFast, no samples"
    n: int = sum(p.numel() for p in model.parameters())
    bsize: int = 1024 if (n != 23910152) else 16 # 24mio=Resnet50=TIN
    if (n == 11271432): bsize = 64 # TIN-RN18

    loss1 = tt.zeros(1, dtype=tt.float32).to(device)
    loss2: float = 0.0
    corr1: tt.Tensor = tt.zeros(1, dtype=tt.int64).to(device)
    corr2i: int = 0

    fdl.DirectBatch(-1) # rewind

    if (model2 is None):
        with tt.no_grad():
            # for data, target in utmt_TS.dlf_listXY: # todo: iterator
            while 1:
                mbs, data, target = fdl.DirectBatch(bsize)
                if (not mbs): break
                output: tt.Tensor = model(data) # TIN crash here (OOM)
                loss1 += cost_function(output, target) * mbs
                # assert(cf < 1e30), "Mx1"
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).sum() # ,int(utmt_TS_Xsize[i])
    else:
        corr2: tt.Tensor = tt.zeros(1, dtype=tt.int64).to(device)
        loss2 = tt.zeros(1, dtype=tt.float32).to(device)
        with tt.no_grad():
            # for data, target in utmt_TS.dlf_listXY: # todo: iterator
            while 1:
                mbs, data, target = fdl.DirectBatch(bsize)
                if (not mbs): break
                output: tt.Tensor = model(data) # TIN crash here (OOM)
                loss1 += cost_function(output, target) * mbs
                # assert(cf < 1e30), "Mx2"
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).sum()

                output: tt.Tensor = model2(data)
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr2 += pred.eq(target.view_as(pred)).sum()
                loss2 += cost_function(output, target) * len(target)
        corr2i = corr2.item()
        loss2  = loss2.item()

    # print("FastCacheEval, inf-errors=", err, count)
    fdl.dlf_EpochCount += 1 # Test
    return corr1.item(), corr2i, loss1.item(), loss2

def TestEval(model, model2, test_loader:DataLoader, cost_function, device) -> tuple[float,float,float,float]:
    "calc Accuracy + TestLoss over full test-dataset"
    # model.eval() --
    if (test_loader is None) or (len(test_loader.dataset) < 1):
        return -1.0, -1.0

    global utmt_TS
    utmt_TS_samples: int = utmt_TS.dlf_samples
    # only_lab: bool = utmt_TS.dlf_only_lab
    test_loss: float = 0.0 # tt.zeros(1, dtype=tt.float32).to(device)
    test_loss2: float = 0.0
    count: int = len(test_loader.dataset)
    dt: float = time_time()

    if len(test_loader.dataset) != utmt_TS_samples or utmt_TS.dlf_only_lab: # slower
        if utmt_TS.dlf_only_lab:
            corr, corr2, test_loss, test_loss2 = NormalEval_Labels(
                model, model2, test_loader, cost_function, device, utmt_TS.dlf_Labels)
        else:
            corr, corr2, test_loss, test_loss2 = NormalEval(
                model, model2, test_loader, cost_function, device)

    else: # tensors already in device-memory/GPU
        corr, corr2, test_loss, test_loss2 = \
            FastCacheEval(model, model2, utmt_TS, cost_function, device)

    print("(%d)TL=%.3fs," % (-0, time_time() - dt), end=" ") # debug
    if (count > 0):
        s: float = 1.0 / float(count)
        assert(corr > 0), "test accu > 0.0"
        return corr * s, test_loss * s,  corr2 * s, test_loss2 * s
    return float('nan'), float('nan'),  float('nan'), float('nan')

def FastenDataloader(train_data:DataLoader, test_data:DataLoader, num_classes:int, maxMB:int=800, device=tt.device('cpu')) -> None:
    "Intial Load Dataset into CUDA/GPU for FullBatchOperation"
    global utmt_TS, utmt_DS
    dt: float = time_time()

    if (maxMB <= 0) or (train_data is None): # or (str(device).find("cuda") < 0):
        if (train_data is None):
            utmt_TS.InitVectors()
            utmt_DS.InitVectors()
        else:
            print("FastenDataloader(skip, dev=%s)." % (str(device)))
        return # dont skip here, even 2x faster on CPU !

    if 1: # bypass (cache) timm-DL to be faster
        # classes: int = len(test_data.dataset.classes) if hasattr(test_data.dataset, "classes") else -1
        print("FastenDataloader(%d+%d), cls=%d .." % (len(train_data.dataset), len(test_data.dataset), num_classes))
        utmt_TS.Import_DataLoader(test_data, num_classes, is_train=False, maxMB=950, device=device)
        if (utmt_TS.dlf_samples > 0): # both or none
            utmt_DS.Import_DataLoader(train_data, num_classes, is_train=False, maxMB=1400, device=device) # DISABLE DS HERE
        else: # for optim.SetClasses()
            utmt_DS.dlf_init_mbs = DataLdrFast.EstimBatchSize(train_data)
            utmt_DS.dlf_label_max = utmt_TS.dlf_label_max
        dt = time_time() - dt
        print("FastenDataloader(%d+%d), dt=%.3fs.\n" % (utmt_DS.dlf_samples, utmt_TS.dlf_samples, dt))
        DataLdrFast.CheckPair(utmt_DS, utmt_TS) # FilePairConflicts
        # print(utmt_DS.GetDsHash(), utmt_TS.GetDsHash()) #;exit() # debug
        return

    print("FastenDataloader(%d+%d)=SKIP.\n" % (len(train_data.dataset), len(test_data.dataset)))
    return

def full_batch_loss(model, model2, data_loader:DataLoader, cost_function, device) -> tuple[float,float,float,float]:
    "calc. full batch loss (over all train data), only for printing (no effect on solver)"
    if (data_loader is None) or (len(data_loader.dataset) < 1):
        return -1.0
    global utmt_DS
    i: int = 0
    t0: float = time_time()
    # dc: int = tt.cuda.device_count()
    # accumulate loss over batches
    total_loss: float = 0.0
    total_loss2: float = 0.0
    count: int = len(data_loader.dataset)
    # print("full_batch_loss", utmt_DS.dlf_only_lab, utmt_DS.dlf_Labels)

    if (utmt_DS.dlf_samples > 1): # fast - train
        assert(len(data_loader.dataset) <= utmt_DS.dlf_samples) # shrinked < Gpu_DL
        corr, corr2, total_loss, total_loss2 = \
            FastCacheEval(model, model2, utmt_DS, cost_function, device)
        i = 1

    if (not i): # tensors not in device-memory/GPU
        if utmt_DS.dlf_only_lab:
            corr, corr2, total_loss, total_loss2 = NormalEval_Labels(
                model, model2, data_loader, cost_function, device, utmt_DS.dlf_Labels)
        else:
            corr, corr2, total_loss, total_loss2 = \
                NormalEval(model, model2, data_loader, cost_function, device)

    # utmt_DS.dlf_EpochCount += 1 # Train
    t0 = time_time() - t0
    if (t0 > 50.0): # or (err > 0):
        print("sec=%.0f/%.1f%%(%d), " % (t0, 100.0*count/len(data_loader.dataset),-0), end=" ")
    # print(len(data_loader.dataset), len(data_loader)); exit() # [256..,last=96], 60000 235
    if (count > 0):
        s: float = 1.0 / float(count)
        if (model2 is None):
            return corr * s, float(total_loss) * s,  -1.0, inf
        else:
            return corr * s, float(total_loss) * s, corr2 * s, float(total_loss2) * s
    return float('nan'), float('nan'),  float('nan'), float('nan')

# <2m:215:26784> /home/telematik/.venv/lib/python3.11/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
#  return F.conv2d(input, weight, bias, self.stride,

booster_on: bool = False
# for combined_X() - to be removed here
#combiX_sum = combiX_wsum = None
#combi_fx_min, combi_fx_sum, combi_wsum = 0.0, 0.0, 0.0
#combi_cnt: int = 0

#def combined_X(model, fval:float) -> None:
#    "test only: combined x (param) for late improvement" # to be replaced by GetParamAvg()
#    return # debug only
#    global combiX_sum, combiX_wsum, combi_fx_min, combi_fx_sum, combi_fx_min, combi_wsum, combi_cnt
#    if (model is None): # reset
#        combiX_sum, combiX_wsum, combi_cnt = None, None, 0
#        return
#
#    x: tt.Tensor = GetParam(model)
#    if (len(x)>>20 > 20) and ((tt.cuda.get_device_properties(0).total_memory >> 30) < 11): # dim>20mio AND vram<15GB
#        return # skip both for ResNet50(24mio)
#
#    combi_cnt += 1
#    if (combiX_sum is None): # first
#        combiX_sum = x.clone()
#        combi_fx_min, combi_fx_sum = fval, fval
#        if (len(x)>>20 < 15) and (fval > 1e-9): # optional
#            combi_wsum = 1.0 / fval
#            combiX_wsum = x * combi_wsum
#        else:
#            combi_wsum = 0.0
#        return
#    combi_fx_min = min(combi_fx_min, fval)
#    combi_fx_sum += fval
#    combiX_sum += x
#    if (len(x)>>20 < 15) and (fval > 1e-9): # skip for ResNet34(11mio)+ResNet50(24mio)
#        combi_wsum += 1.0 / fval
#        combiX_wsum += x * (1.0 / fval)
#    return

def cosine(a: tt.Tensor, b: tt.Tensor) -> float:
    "cosine between tensor pair (debug only)"
    # return tt.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-30)
    assert(len(a) == len(b)), "dot-product dim-conflict"
    d: float = tt.dot(a, b)
    return 0.0 if (tt.abs(d) < 1e-6) else d / (tt.norm(a) * tt.norm(b))

def PrepareBoostModel(model:tt.nn.Module, device2, optim) -> tuple[tt.nn.Module,int,float]:
    "internal init"
    global model_boost
    boost: bool = booster_on and elra_solver
    if not boost: return None, 0, 0.0

    bcnt, avg, avg_x = optim.GetParamAvg(True)
    if (avg_x is None):
        print("Warn:FBL.GetParamAvg(c=%d,f=%.3g), x=None, skip-boost!" % (bcnt, avg))
        return None, 0, 0.0

    model2: tt.nn.Module = model_boost
    assert(model  is not None), "need source model"
    assert(model2 is not None), "need 2nd model"

    if (device2 is not None): model2.to(device2)

    ImportBatchNorm(model2, ExportBatchNorm(model)) # copy BN
    SetParam(model2, avg_x)
    # model2.eval() # optional (batch norm update)
    return model2, bcnt, avg

def LossRatio(loss_trn:float, loss_val:float) -> float:
    return loss_val/loss_trn if (loss_trn > 0.0) else loss_val-loss_trn

def combined_FBL(model, train_fast, test_data, cost_func, device, optim) -> tuple[float,float,float,float]:
    "Calc 4 Loss+Accu f(x + boost) x2 (train + test)"

    model2, bcnt, avg = PrepareBoostModel(model, None, optim)
    # boost: bool = booster_on and elra_solver

    assert len(train_fast) > 0, "empty dataset train"
    assert len(test_data)  > 0, "empty dataset test"
    accu1t, loss1t, accu2t, loss2t = full_batch_loss(model, model2, train_fast, cost_func, device)
    accu1v, loss1v, accu2v, loss2v = TestEval(model, model2, test_data, cost_func, device)

    t2t: float = LossRatio(loss1t, loss1v)
    print("*Train1 (%.3f%% %.6g), Test1 (%.3f%% %.6g), t2t=%.2g, n=%d" % (accu1t*100, loss1t, accu1v*100, loss1v, t2t, bcnt))
    if (model2 is not None):
        t2t = LossRatio(loss2t, loss2v)
        print("  *Boost2 (%.3f%% %.6g), Test2 (%.3f%% %.6g), t2t=%.2g, AvLf=%.3g" % (accu2t*100, loss2t, accu2v*100, loss2v, t2t, avg))
        optim.TellTrainBoostLoss(loss2t)
    else:
        accu2t = loss2t = accu2v = loss2v = -1.0
    return accu1t, loss1t, accu2t, loss2t,  accu1v, loss1v, accu2v, loss2v

smp_return: tuple = None
def thread_eval(x, y, model, cost_func) -> None:
    "internal: threaded SMP/DDP loss"
    global smp_return

    output: tt.Tensor = model(x)
    # loss = cost_func(output, y) # * len(y)
    pred: tt.Tensor  = output.max(1, keepdim=True)[1]
    # corr = pred.eq(y.view_as(pred)).sum()

    smp_return = (pred.eq(y.view_as(pred)).sum(), cost_func(output, y))
    return

def single_eval(x, y, model, cost_func) -> None:
    "internal: single loss+accu"

    output: tt.Tensor = model(x)
    # loss = cost_func(output, y) # * len(y)
    pred: tt.Tensor  = output.max(1, keepdim=True)[1]
    # corr = pred.eq(y.view_as(pred)).sum()

    return pred.eq(y.view_as(pred)).sum(), cost_func(output, y)

def calc_FBL_m2d2(model1, model2, data_loader, cost_func, device1, device2) -> tuple[float,float, float,float]:
    "full-batch-eval for 2x model at 2 devices"

    global smp_return
    from threading import Thread
    count: int = len(data_loader.dataset)
    assert(count > 1), "empty dataset (FBL)"
    img_type: tt.dtype = tt.get_default_dtype()
    loss1 = tt.zeros(1, dtype=tt.float32).to(device1)
    loss2 = tt.zeros(1, dtype=tt.float32).to(device2)
    corr1 = tt.zeros(1, dtype=tt.int64).to(device1)
    corr2 = tt.zeros(1, dtype=tt.int64).to(device2)
    bs = None
    smp_return = None # tuple(2)

    with tt.no_grad():
        for data0, target0 in data_loader:
            x1, y1 = data0.to(device1, img_type, non_blocking=True), \
                target0.to(device1, non_blocking=True)
            x2, y2 = data0.to(device2, img_type, non_blocking=True), \
                target0.to(device2, non_blocking=True)
            if (bs is None):
                bs = len(target0)
            else:
                if len(target0) != bs: break

            smp_return = None # tuple(2)
            th1 = Thread(target=thread_eval, args=(x2,y2, model2, cost_func))
            th1.start() # ((((

            output: tt.Tensor = model1(x1)
            loss1 += cost_func(output, y1) # * len(target0)
            pred: tt.Tensor  = output.max(1, keepdim=True)[1]
            corr1 += pred.eq(y1.view_as(pred)).sum()

            th1.join(timeout=40.0) # 40s, ))))
            # assert(smp_return is not None), "debug: calc_FBL_m2d2"
            c2, l2 = smp_return
            corr2 += c2
            loss2 += l2

    data0 = None
    loss1 *= bs; loss2 *= bs
    tlen: int = len(target0)

    if tlen and tlen < bs:
        c2, l2 = single_eval(x1,y1, model1, cost_func)
        corr1 += c2; loss1 += l2 * tlen

        c2, l2 = single_eval(x2,y2, model2, cost_func)
        corr2 += c2; loss2 += l2 * tlen

    smp_return = None
    s: float = 1.0 / count

    return corr2.item() * s, loss1.item() * s, corr2.item() * s, loss2.item() * s

def combined_FBL_smp(model, train_fast, test_data, cost_func, device, device2, optim) -> tuple[float,float,float,float]:
    "parallel (SMP = 2 models on 2 GPUs) calc of FullBatchLoss for normal + boost"
    model2, bcnt, avg = PrepareBoostModel(model, device2, optim)
    dt: float = time_time()
    if (model2 is not None):
        print("combined_FBL_smp(%s + %s), <<<<<" % (str(device), str(device2)))
        accu1t, loss1t, accu2t, loss2t = calc_FBL_m2d2(model, model2, train_fast, cost_func, device, device2)
        accu1v, loss1v, accu2v, loss2v = calc_FBL_m2d2(model, model2, test_data, cost_func, device, device2)
        dt = time_time() - dt
        print("combined_FBL_smp(dt=%.2f) done.  >>>>>" % dt)
        print("*Train1 (%.3f%% %.6g), Test1 (%.3f%% %.6g), t2t=%.2g, n=%d" % (accu1t*100, loss1t, accu1v*100, loss1v, LossRatio(loss1t, loss1v), bcnt))
        print("  *Boost2 (%.3f%% %.6g), Test2 (%.3f%% %.6g), t2t=%.2g, AvLf=%.3g" % (accu2t*100, loss2t, accu2v*100, loss2v, LossRatio(loss2t, loss2v), avg))
        optim.TellTrainBoostLoss(loss2t)
    else:
        accu1t, loss1t, accu2t, loss2t = full_batch_loss(model, None, train_fast, cost_func, device)
        accu1v, loss1v, accu2v, loss2v = TestEval(model, None, test_data, cost_func, device)
        print("*Train1 (%.3f%% %.6g), Test1 (%.3f%% %.6g), t2t=%.2g, n=%d" % (accu1t*100, loss1t, accu1v*100, loss1v, LossRatio(loss1t, loss1v), bcnt))
        accu2t = loss2t = accu2v = loss2v = -1.0
    return accu1t, loss1t, accu2t, loss2t,  accu1v, loss1v, accu2v, loss2v

class LossHistRb:
    "ringbuffer for running average of loss history"
    def __init__(self) -> None:
        self.hist_loss = tt.zeros(128, dtype=tt.float32, device=tt.device('cpu'))
        self.hist_lpos: int = 0
        # self.hist_lsum: float = 0.0
        return

    def add_loss(self, loss: float) -> None:
        # pos: int = self.hist_lpos & 127 # mod 128
        # self.hist_lsum += loss - self.hist_loss[pos].item()
        self.hist_loss[self.hist_lpos & 127] = loss
        self.hist_lpos += 1
        return

    def get_mean_loss(self) -> float:
        return tt.mean(self.hist_loss).item()

utmt_LossHist: LossHistRb = LossHistRb()

def CalcLossLimit(loss0: float) -> float:
    "loss threshold for retrace"
    return loss0 * 1.5 if (loss0 > 0.0) else (loss0 + 1.0) # optim.GetLossLimit()

def train_step_adam(X:tt.tensor, y:tt.tensor, model:tt.nn.Module, loss_func, optim:tt.optim.Optimizer, limitf:float, no_scaler) -> float:
    "Training step for Adam/Lion/SGD/DOG."
    global utmt_LossHist # hist_loss, hist_lpos

    optim.zero_grad(set_to_none=True) # (optional: skip for batch-combining)

    loss = loss_func(model(X), y)
    loss_item: float = loss.item() # already fix

    if (loss_item < 1e999): # isnan(loss_item), 1e999==inf (fastest)
        utmt_LossHist.add_loss(loss_item)
    else:
        n2, nM = GetParamxMax(model, loss.device)
        print("Warn:training_step(loss=nan)!", loss_item, n2, nM)

    if True:
        loss.backward() # computes gradient for normal opt (Adam etc)
        optim.step()
        if (dog_averager is not None): dog_averager.step() # DoG+LDoG

    return loss_item

def training_step_scaler(X:tt.tensor, y:tt.tensor, model:tt.nn.Module, loss_func, optim:tt.optim.Optimizer, limitf:float, scaler) -> float:
    "Training step for torch model + scaler, ELRA (P2M+C2M)"
    global utmt_LossHist # hist_loss, hist_lpos

    # optim.zero_grad(set_to_none=True) # (optional: skip for batch-combining)
    model.zero_grad(set_to_none=True) # 1:1

    # with autocast(): # run forward pass with autocast (mixed precision)
    loss = loss_func(model(X), y)
    loss_item: float = loss.item() # already fix

    if (loss_item < limitf): # isnan(loss_item), 1e999==inf (fastest)
        utmt_LossHist.add_loss(loss_item) # !MP
    else:
        n2, nM = GetParamxMax(model, loss.device)
        print("Warn:training_step(loss=nan)!", loss_item, limitf, n2, nM)
        optim.step_retrace(loss_item)
        return loss_item

    # MyGradScaler:
    loss *= scaler.scaling
    loss.backward()  # computes gradient for P2M+C2M
    absmax: float = GetGradMax(model)
    if (absmax < 1e999): # no overfloaw step, early skip
        optim.step(loss_item, scaler.inv_scaling)
    scaler.UpdateGradNorm(absmax)

    # torch\cuda\amp\grad_scaler\GradScaler
    #scaler.scale(loss).backward()
    #scaler.step(optim, loss_item, 1.0)
    #scaler.update()
    # Python311\site-packages\torch\cuda\amp\grad_scaler.py

    return loss_item

def training_step(X:tt.tensor, y:tt.tensor, model:tt.nn.Module, loss_func, optim:tt.optim.Optimizer, limitf:float, unused) -> float: 
    "Training step for torch model, ELRA (P2M+C2M)"
    global utmt_LossHist # hist_loss, hist_lpos

    optim.zero_grad(set_to_none=True) # (optional: skip for batch-combining)

    # with autocast(): # run forward pass with autocast (mixed precision)
    loss = loss_func(model(X), y)
    loss_item: float = loss.item() # already fix

    if (loss_item < limitf): # isnan(loss_item), 1e999==inf (fastest)
        utmt_LossHist.add_loss(loss_item)
    else:
        n2, nM = GetParamxMax(model, loss.device)
        print("Warn:training_step(loss=nan)!", loss_item, limitf, n2, nM)
        optim.step_retrace(loss_item)
        return loss_item

    loss.backward()  # computes gradient for P2M+C2M
    optim.step_noscale(loss_item)

    # print("(SkipLastBatch, bs=%d<%d, f=%.6f)" % (len(y), utmt_LossHist.prev_bs, loss_item), flush=True) # skip last+short batch
    return loss_item

def thread_step_smp(X:tt.tensor, y:tt.tensor, model:tt.nn.Module, loss_func, p0, limitf:float, scaling:float, device1) -> float:
    "Training step for torch model + scaler + MP, ELRA (P2M+C2M)"
    global smp_return
    # scaling: float = scaler.scaling # freeze scaler

    model.zero_grad(set_to_none=False) # model2
    if (p0 is not None): SetParam(model, p0) # sync param 1->2

    loss = loss_func(model(X), y)
    loss_item: float = loss.item()
    # print("thread_step_smp", loss_item, type(smp_return))

    if not (loss_item < limitf): # 1e999==inf (fastest)
        n2, nM = GetParamxMax(model, loss.device)
        print("Warn:training_step(loss2=nan), SMP!", loss_item, limitf, n2, nM)
        # optim.step_retrace(loss_item)
        smp_return = (loss_item, None)
        return

    # MyGradScaler:
    loss *= scaling
    loss.backward()  # computes gradient for P2M+C2M
    absmax: float = GetGradMax(model)
    if (absmax < 1e999): # no overflow step, early skip
        # optim.step(loss_item, None) # lgs_dev=None)
        grads = [param.grad.data.view(-1) for param in model.parameters()]
        smp_return = (loss_item, (tt.cat(grads) * (1.0/scaling)).to(device1, non_blocking=True) )
        return
    # scaler.UpdateGradNorm(absmax) # not in MP

    print("model2: loss=%.3e, |grad|=%.3e, SMP!" % (loss_item, absmax))
    smp_return = (None, None)
    return

def step_smp(x1,y1, x2,y2, model1,model2, optim, cost_func, device1,device2, limitf:float, p0, thread) -> bool:
    "internal: parallel step() SMP"
    # from threading import Thread # Lock
    global smp_return, scaler

    fin, _ = optim.CheckNextStep()
    if fin: # final/real step expected (no SMP/DDP)
        loss1 = training_step_scaler(x1,y1, model1, cost_func, optim, limitf, scaler)
        p1 = optim.GetParam(True, device2)
        # assert(p1 is not None), "expect real step" # debug test
        return loss1, False, p1

    p1 = smp_return = None # tuple(2)
    th1 = thread(target=thread_step_smp, args=(x2,y2, model2, cost_func, p0, limitf, scaler.scaling, device1))
    th1.start() # ((((

    loss1 = training_step_scaler(x1,y1, model1, cost_func, optim, limitf, scaler)
    p1 = optim.GetParam(True, device2)
    # thread_step_smp(data2[0: bs], target2[0: bs], model2, cost_func, None, limitf, scaler, device1)
    th1.join(timeout=40.0) # 40s, ))))
    assert(smp_return is not None), "train thread timeout (increase timeout for huge nets)"

    if (p1 is not None): # drop 2nd result
        return loss1, False, p1 # reuse x2,y2

    loss2, grad = smp_return
    if (grad is not None):
        optim.step_noscale(loss2, grad)
    else:
        if (loss2 is not None): optim.step_retrace(loss2)
    p1 = optim.GetParam(True, device2)
    
    return loss1, True, p1 # ok (both threads used)

def SelectStepCall(elra_solver:bool, scaler) -> callable:
    "switch step() callable"
    if (elra_solver):
        tstep = training_step if (scaler is None) else training_step_scaler
    else:
        tstep = train_step_adam
    assert callable(tstep), "need function for step()"
    return tstep

def train_epoch_smp(dataloader, model1, model2, optim, cost_func, device1, device2, loss0: float):
    "MultiGpu training (SMP)"
    assert(device2 is not None), "needs 2 GPUs"
    assert(model2 is not None), "needs 2 models"
    global smp_return, scaler
    from threading import Thread # inside step_smp()

    tstep = SelectStepCall(elra_solver, scaler) # training_step_scaler()
    _, collects = optim.CheckNextStep()
    # assert(collects >= 2), "else useless with SMP/DDP" # zero before init
    assert(scaler is not None), "todo (only 16bit for now)"

    img_type: tt.dtype = tt.get_default_dtype()
    limitf: float = CalcLossLimit(loss0)
    ldl: int = len(dataloader)
    assert(ldl > 2), "empty dataloader"
    pp100: float = 100.0 / ldl
    nextpp, ldlp = 0, (ldl * 2) // 100 # progress piece
    ldlp = max(10, ldlp)
    prev_bs: int|None = None
    redo1 = None
    redo_cnt = err2cnt = 0
    p1: tt.Tensor = None

    smp_return = None # tuple(2)
    SetParam(model2, GetParam(model1).to(device2)) # sync 1->2
    # Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context...
    print("train_epoch_smp(%s + %s, bc=4x%d)  <<<<<" % (str(device1), str(device2), ldl), flush=True)
    t1 = dt = time_time()

    for batch_idx, (data0, target0) in enumerate(dataloader):
        if (prev_bs is None):
            prev_bs = len(target0) # new multi batches
            bs = prev_bs >> 2
            bs2 = bs*2
        else:
            if (len(target0) != prev_bs): break # skip tail
        # reduce cudaMemcpy(H2D) by multi-batch transfer
        x1, y1 = data0[0: bs2].to(device1, img_type, non_blocking=True), target0[0: bs2].to(device1, non_blocking=True)
        x2, y2 = data0[bs2:  ].to(device2, img_type, non_blocking=True), target0[bs2:  ].to(device2, non_blocking=True)

        x2a, y2a = x2[0:bs], y2[0:bs] # device2: 1st half
        l1, r2ok, p1 = step_smp(x1[0:bs],y1[0:bs], x2a,y2a, model1, model2, optim, cost_func, device1, device2, limitf, p1, Thread)
        if not r2ok:
            if (redo1 is None):
                redo1 = (x2a.to(device1, non_blocking=True), y2a.to(device1, non_blocking=True))
            else:
                x1a, y1a = redo1
                _, r2ok, p1 = step_smp(x1a,y1a, x2a,y2a, model1, model2, optim, cost_func, device1, device2, limitf, p1, Thread)
                redo1, x1a, y1a = None, None, None # release GPU tensor
                if not r2ok:
                    redo1 = (x2a.to(device1, non_blocking=True), y2a.to(device1, non_blocking=True))
                    err2cnt += 1
            redo_cnt += 1

        x2a, y2a = x2[bs:], y2[bs:] # device2: 2nd half
        l2, r2ok, p1 = step_smp(x1[bs:],y1[bs:], x2a,y2a, model1, model2, optim, cost_func, device1, device2, limitf, p1, Thread)
        if not r2ok:
            if (redo1 is None):
                redo1 = (x2a.to(device1, non_blocking=True), y2a.to(device1, non_blocking=True))
            else:
                x1a, y1a = redo1
                _, r2ok, p1 = step_smp(x1a,y1a, x2a,y2a, model1, model2, optim, cost_func, device1, device2, limitf, p1, Thread)
                redo1, x1a, y1a = None, None, None # release GPU tensor
                if not r2ok:
                    redo1 = (x2a.to(device1, non_blocking=True), y2a.to(device1, non_blocking=True))
                    err2cnt += 1
            redo_cnt += 1   

        if (batch_idx >= nextpp):
            t: float = time_time()
            t1, t = t, t - t1
            print("Epoch progress: [%d/%d=%.0f%%]  Loss1=%.3e, dt=%.1fs" % (batch_idx, ldl, pp100*batch_idx, (l1+l2) * 0.5, t), flush=True)
            nextpp += ldlp

    x2 = y2 = None
    if (redo1 is not None):
        x1a, y1a = redo1
        l1 = tstep(x1a, y1a, model1, cost_func, optim, limitf, scaler)

    tail: int = len(target0)
    if tail and (len(target0) < prev_bs):
        x1, y1 = data0.to(device1, img_type, non_blocking=True), target0.to(device1, non_blocking=True)
        pos, nxt = 0, bs
        while (nxt <= tail):
            l1 = tstep(x1[pos:nxt], y1[pos:nxt], model1, cost_func, optim, limitf, scaler)
            pos, nxt = nxt, nxt+bs
        # print("(skip tail of %d samples)" % len(target0))

    dt = time_time() - dt
    print("train_epoch_smp(bc=%d, redo=%d+%d, dt=%.2f) done.  >>>>>" % (batch_idx, redo_cnt, err2cnt, dt), flush=True)
    redo1, p1, smp_return = None, None, None
    model1.zero_grad(set_to_none=True)
    model2.zero_grad(set_to_none=True)
    return [0], [0], [0], [0], [0]

def train_epoch(dataloader, model, optim, cost_func, device, loss0: float):
    "train single epoch"
    losses, batches = [], []
    steps = []
    f_calls, g_calls = [], []
    last_time: float = -1.0
    tl_tmp: str = ""

    global scaler # booster_on, utmt_DS
    limitf: float = CalcLossLimit(loss0)
    tstep = SelectStepCall(elra_solver, scaler)
    if (elra_solver):
        optim.GetParamAvg(booster_on) # reset
        # print("limitf=", limitf, loss0) # 3.5

    # utmt_DS.FullShuffle(0) # todo:test
    # print(utmt_DS.GetDsHash()) # check data integrity by sums
    # utmt_DS.SetMiniBatchSize(0, shuffle=True) # reorder batches
    # utmt_DS.ShuffleBatch(rewind=True)

    batch_idx: int = 0
    # batch_max: int = len(dataloader) - 1 # 0..max
    # if (not elra_solver): model.train()
    img_type: tt.dtype = tt.get_default_dtype() # tt.float32
    # ldf = open("path_dom.dat", "a")
    norm100: float = 100.0 / len(dataloader)
    prev_bs: int = 0
    bs: int = 2
    loss: float = 0.0
    i: int = 0
    # data0, target0 = None, None

    for batch_idx, (data0, target0) in enumerate(dataloader): # old+ok
        # reduce cudaMemcpy(H2D) by multi-batch transfer
        data, target = data0.to(device, img_type, non_blocking=True), target0.to(device, non_blocking=True)
        if (len(target) >= prev_bs):
            if (not prev_bs): # == 0
                prev_bs = len(target) # new multi batches
                bs = prev_bs >> 2 # // 4
                bs2, bs3 = bs*2, bs*3
            loss = tstep(data[0: bs], target[0: bs], model, cost_func, optim, limitf, scaler)
            loss = tstep(data[bs: bs2], target[bs: bs2], model, cost_func, optim, limitf, scaler)
            loss = tstep(data[bs2: bs3], target[bs2: bs3], model, cost_func, optim, limitf, scaler)
            loss = tstep(data[bs3: ],  target[bs3: ],  model, cost_func, optim, limitf, scaler)
        else: # requires: drop_last=False
            pos, left = 0, len(target)
            while (left >= bs):
                end: int = pos + bs
                loss = tstep(data[pos: end], target[pos: end],\
                            model, cost_func, optim, limitf, scaler)
                left, pos = left - bs, end
            if (left > 0):
                print("(SkipLastBatch, bs=%d<%d<%d, f=--)" % (left, len(target), prev_bs), flush=True)
            break # optional

        #if (len(target) >= prev_bs): # early skip (drop_last) # batch_max - batch_idx)
        #    loss = tstep(data.to(device, img_type), target.to(device), model, cost_func, optim, limitf, scaler)
        #else:
        #    print("(SkipLastBatch, bs=%d<%d, f=--)" % (len(target), prev_bs), flush=True)
        #prev_bs = len(target)
        #if (99 == batch_idx % 100): Interpolx(model, cost_func, dataloader)
        #else: Interpolx(model, None, None)
        # WriteDominant(GetParam(model), log=ldf, t=batch_idx) # test (slow)
    # while 1: # new Shuffler-Loop
        #mbs, data, target = utmt_DS.ShuffleBatch()
        #if (mbs < 1): break
        #loss: float = training_step(data, target, model, cost_func, optim) # new

        if (i < 10):
            i += 1
        else:
            i = 0
            tnow : float = time_time()
            dt : float = tnow - last_time
            if (dt >= 5.0): # progress print interval
                batch : int = batch_idx + 1
                if "" == tl_tmp:
                    print('Epoch progress: [%d/%d=%.0f%%]\tBatch avg. train loss:\t%.6g' %
                        (batch * len(data), len(dataloader.dataset),
                            batch * norm100, loss) )
                else:
                    tl_tmp = (" %.6g" % loss) # += !
                    print('Epoch progress: [%.1f%%]\tBatches train losses: %s' %
                        (batch * norm100, tl_tmp) )
                last_time = tnow
                tl_tmp = ""

        # if False and (type(optim) is not ElraOptimizer): # needed ?
        #     optim.state["o_calls"] += 1
        #     optim.state["f_calls"] += 1
        #     optim.state["g_calls"] += 1
        # f_calls.append(optim.state["f_calls"])
        # g_calls.append(optim.state["g_calls"])

        # step: int = optim.state["o_calls"]
        # f, g = optim.state["f_calls"], optim.state["g_calls"]

        if False:
            steps.append(optim.state["o_calls"])
            losses.append(loss)
            batches.append(batch_idx)

        if not (loss < 1e99):
            break

    f_calls.append(-1)
    g_calls.append(-1)
    steps.append(-1)
    losses.append(-1)
    batches.append(-1)

    n2, nM = GetParamxMax(model, device)
    print("GetParamxMax(bs=%d) = %.3g > %.3g" % (bs, n2, nM))
    if (scaler is not None):
        print("GradScaler.get_scale() = %.3g" % scaler.get_scale())

    optim.zero_grad(set_to_none=True) # less memory during test full-batch
    # if (not elra_solver): model.eval() # issue: batch normalize + booster_on
    LimitBatchNorm(model, 0.0) # print peak only
    return losses, batches, steps, f_calls, g_calls

def WriteParams(model, num:int, tmp: bool) -> None:
    "backup params=x (epoch wise)"
    from os.path import exists
    if exists("params_tmp"):
        assert(num >= 0), "neg. epoch"
        assert(model is not None), "empty model"
        if not (num % 4):
            fn:str = "params_tmp/epoch_" + str(int(num)) + ".pt"
            ParamSave(model, fn)
    else:
        if (num <= 2): print("Hint:WriteParams(no-folder)=skip.")
        if (tmp): ParamSave(model, fn="epoch_tmp.pt")
    return

def GetOtherDevice(model, device, elra_solver:bool):
    "intern: other MultiGpuDevice (SMP/DDP)"
    from copy import deepcopy
    device2 = None # tt.device('cpu')

    if not tt.cuda.is_available(): return None, None

    # return device, deepcopy(model) # debug/test on single-gpu
    return None, None # return (use single gpu only) !!!!! (comment out to SMP/DDP)

    did: int = tt.zeros(0, device=device).get_device()
    dc: int = tt.cuda.device_count()
    if did < 0: return None, None # cpu-only
    if dc < 2: return None, None # single gpu

    m = [tt.cuda.mem_get_info(i) for i in range(dc)] # (free_bytes, device_bytes)
    used_mb = [(m[i][1]-m[i][0])>>20 for i in range(dc)]
    dev0mb:int = m[0][1] >> 20 # MB physical VRAM

    if elra_solver:
        if (model is None):
            device2 = tt.device("cuda", (did + 1) % dc)
        else:
            if sum(p.numel() for p in model.parameters()) > (1<<20):
                device2 = tt.device("cuda", (did + 1) % dc)

    print("(Multi-GPU detected) %dx%dMB, used:%s, d2=%s" % (dc, dev0mb, str(used_mb), str(device2)))
    model2 = None if (model is None or device2 is None) else deepcopy(model).to(device2)
    return device2, model2

def train(dataloaders: tuple,
          model, cost_func, optim, max_epochs:int = 1000, target_loss:float = 0.0, 
          batch_size:int = 0, device = tt.device('cpu'), logf = None):
    "train model with train_data"
    train_data:DataLoader = dataloaders[0]
    test_data: DataLoader = dataloaders[1]
    train_fast:DataLoader = dataloaders[2] if (len(train_data.dataset) > 60000) else dataloaders[0]
    num_classes: int = dataloaders[3]

    from statistic_helper import GlobalStatist as gstat
    from datetime import datetime as datetime2
    from math import log
    from time import sleep
    global booster_on, model_boost, dog_averager, scaler, utmt_DS # utmt_TS, utmt_LossHist
    tt.set_printoptions(precision=4, linewidth=150)
    # batch_size = batch_size if batch_size < 999999999 else len(X) # (9x9) = inf-like inf
    print("BS = %d (%d/4)" % (batch_size, DataLdrFast.EstimBatchSize(train_data)))

    pdf: PandasDataFrame = PandasDataFrame()
    test_loss_min: float = inf

    epoch: int = 1

    optim.state["o_calls"], optim.state["f_calls"], optim.state["g_calls"] = 0, 0, 0

    # get_wd_params(model)
    CheckElraSolver(optim)
    if (str(tt.get_default_dtype()).find('float16') > 0):
        scaler = MyGradScaler(init_scale=2.0**15, enabled=True) # def=2^16
    if (dog_solver):
        from dog import PolynomialDecayAverager
        dog_averager = PolynomialDecayAverager(model)
    ParamLoad(model, fn="", nofilewarn=False) # "startx_000.pt"
    FastenDataloader(train_fast, test_data, num_classes, maxMB = 800, device = device) # here DL switch-off
    if (elra_solver): optim.SetClasses(num_classes, batch_size)
    # if (not elra_solver): model.eval() # E.g. dropout layers will be disabled during evaluation and batchnorm layers will use the running stats instead of the batch statistics to normalize the activation. The gradient computation will not be changed or disabled. !!
    device2 = None
    if tt.cuda.is_available():
        device2, model2 = GetOtherDevice(model, device, elra_solver)
            # model_dp = DataParallel(model, gpu_ids = [0,1])
        print("torch.cuda.MB:", tt.cuda.memory_allocated()>>20, tt.cuda.memory_reserved()>>20)

    # Initial Training loss auf allen Trainings Daten, F(parameter satz)=Loss
    a0, loss, _, _ = full_batch_loss(model, None, train_fast, cost_func, device)
    print(datetime2.now().strftime("[%H:%M:%S]"), end=" ")
    a0 *= num_classes
    if a0 > 1.5: print("### accu x classes = %.1f > 1 !!!" % a0)
    print('Start training: \t\tInit. avg. train loss:\t%.6f ln(%.1f)' % (loss, 2.718**loss), flush=True)
    loss0: float = log(num_classes) #loss # log(classes)
    # ResetBatchNorm(model)

    pdf.AppendFrame(loss, None, 0, "train", 0, 0, 0)

    # last_epoch_index: int = 0
    # logf = open("history.txt", "a")

    while loss > target_loss and epoch <= max_epochs and loss < abs(loss0)*3: # stop criteria
        if (elra_solver): optim.SetLgsDevice(device) # restore GPU-RAM (5x Tensor)
        dt1: float = time_time()
        if (device2 is None):
            epoch_losses, epoch_batches, step, fs, gs = train_epoch(train_data, model, optim, cost_func, device, loss0)
        else: # MultiGpu
            epoch_losses, epoch_batches, step, fs, gs = train_epoch_smp(train_data, model, model2, optim, cost_func, device, device2, loss0)
        dt1 = time_time() - dt1
        mean_hist_loss: float = utmt_LossHist.get_mean_loss()
        if (dt1 > 15.0): print("(end epoch %d, f128=%.3f, dt=%.1fs)" % (epoch, mean_hist_loss, dt1), flush=True)

        # pdf.ExtendFrame(epoch_losses, epoch_batches, epoch, "batch", step, fs, gs) # still needed ?

        # F(nach epoch)
        if elra_solver: optim.SetLgsDevice(tt.device('cpu')) # save GPU-RAM (3x Tensor)
        dt2: float = time_time()
        if (device2 is None):
            accu, loss, accu2t, loss2t,  accu1v, loss1v, accu2v, loss2v = combined_FBL(
                model, train_fast, test_data, cost_func, device, optim)
        else:
            accu, loss, accu2t, loss2t,  accu1v, loss1v, accu2v, loss2v = combined_FBL_smp(
                model, train_fast, test_data, cost_func, device, device2, optim)
        dt2 = time_time() - dt2
        if (dt2 > 100.0): print(datetime2.now().strftime("[%H:%M:%S]"), end=" ")
        print('Finished epoch %d/%d: \t\tFinal avg. train loss:\t%.6f (%.3f,ac=%.2f%%), dt=%.3f+%.3fs' %
            (epoch, max_epochs, loss, mean_hist_loss, accu*100, dt1, dt2), flush=True)
        WriteParams(model, epoch, dt2 > 900.0) # epoch_tmp.pt
        CopyLgsFile()

        # pdf.AppendFrame(loss, None, epoch, "train", step[-1], fs[-1], gs[-1]) # still needed ?

        if test_data is not None:
            # test_accu, test_loss, _, _ = TestEval(model, None, test_data, cost_func, device)
            # test_loss_min = min(test_loss_min, test_loss) = loss1v
            # print("Test set: Average loss: %.4f(>%.4f), accu: %.3f%%, dt=%.3f+%.3fs\n" % (test_loss, test_loss_min, test_accu*100,dt1,dt2), flush=True)
            if logf is not None:
                logf.write("%d,%.4g,%.4f,%.6g,(%.4g:%.4f:%.4g:-1),%s,%s\n" %
                (epoch, loss,accu1v,loss1v, loss2t,accu2v,loss2v, StatListStr(epoch_losses),gstat.statist_GetStr()))
                if (dt2 > 50.0): logf.flush()

            pdf.AppendFrame(loss1v, None, epoch, "test", step[-1], fs[-1], gs[-1]) # test_loss = loss1v

        #if losses[-1] >= losses[last_epoch_index] - losses[last_epoch_index] * 0.01: break # speed up benchmarks and break early

        #if('converged' in optim.state and optim.state['converged']): break # use p2min / c2min converged flag

        if (not booster_on) and (epoch >= 5) and elra_solver: # decide once
            n: int = sum(p.numel() for p in model.parameters())
            if (epoch == 10) or (25549352 == n): # ImgNet1k=25mio
                booster_on = ((n>>20) < 22) or ((tt.cuda.get_device_properties(0).total_memory >> 30) > 10) # low-dim or high-gpu-ram
                print("Booster=", booster_on)
                if (booster_on):
                    model_boost = ElraOptimizer.CopyModel(model, mode='default') # 'default','reduce-overhead',..

        if (tt.cuda.device_count() > 1): sleep(1.5) # relax system due to multi-gpu issue

        epoch += 1

    ParamSave(model) # for later reuse
    # ResetBatchNorm(model)
    if (booster_on):
        ParamSave(model_boost, fn="final_boost.pt")

    # Release Tensor Memory
    FastenDataloader(None, None, 0)
    # logf.close()
    return pdf.ReturnFrame() # losses, batches, epochs, types, steps, f_calls, g_calls

# EoF.
