# UniversalTorchModelTrainer.py (2024)
# (included own DataLoader for constant Data)

import torch as tt
from torch.utils.data import DataLoader # TensorDataset
from torch import nn, cuda
from torch import cat as tt_cat
from torch import no_grad, zeros, get_default_dtype, float32, int64
# from torch.nn.parallel import DataParallel # Multi-GPU (new)
# from torch.cuda.amp import autocast, GradScaler
from Cos2MinTorchFunctionOptimizer import ElraOptimizer # ELRA_class.py (tbd 2024)
from time import time as time_time
# from os import path # remove, replace, (only path.isfile)
# from accelerate import Accelerator
# from accelerate.utils import gather_object
from math import inf, nan # isnan
device_cpu: tt.device = tt.device('cpu')
cuda_device_count:int = cuda.device_count()

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
    if isfile(fn): copyfile(fn, "epoch_lgs.txt")
    return

def GetGradMax(model: nn.Module) -> float:
    "inf-norm of gradients - unused"
    tt_max = tt.max
    #if dev is None: dev = next(model.parameters()).device
    #nrm: tt.Tensor = tt_max[p.grad.data.flatten().abs().max() for p in model.parameters() if p.grad.data is not None]

    nrm: tt.Tensor = None  # zeros(1, device=dev)
    for p in model.parameters():
        pgd = p.grad.data
        # assert pgd is not None
        if pgd is not None:
            if nrm is None:
                nrm = pgd.flatten().norm(1e999) # .abs().max()
                nrm_item: float = nrm.item()
                if not (nrm_item < inf): return nrm_item # early skip
            else:
                nrm = tt_max(nrm, pgd.flatten().norm(1e999)) # .abs().max()
            # ord=inf, view(-1) max(abs(x))
    return -1.0 if nrm is None else nrm.item()

def GetParamxMax(model: nn.Module, device) -> tuple[float, float]:
    "inf-norm of x (param)"
    if model is None: return -1.0
    nrm2: tt.Tensor = zeros(1, device=device)
    nrmM: tt.Tensor = zeros(1, device=device)
    # norm = tt.linalg.norm
    tt_square, tt_max = tt.square, tt.max

    for param in model.parameters():
        pdv = param.data.flatten() # view(-1)
        nrm2 += tt_square(pdv.norm())  # ord=2.0
        nrmM = tt_max(nrmM, pdv.norm(1e999)) # .abs().max() # max(abs(x)), norm(inf)
    return nrm2.sqrt().item(), nrmM.item()

#def FreezeBatchNorm(model: nn.Module) -> None:
#    "unused", # model = ResNet50()
#    for m in model.modules():
#        if isinstance(m, nn.BatchNorm2d):
#            m.eval()
#    return

def BatchNormMoment(model: nn.Module, momentum: float) -> None:
    "BatchNorm2d, change momentum, default = 0.1, freeze = 0.0"
    if model is None: return
    BatchNorm2d = nn.BatchNorm2d
    assert momentum >= 0.0, "BatchNorm2d"
    msum:float = 0.0 # m.running_mean
    for m in model.modules():
        if isinstance(m, BatchNorm2d) and m.track_running_stats:
            rn = m.running_mean.norm()
            # print("Mom.", m.momentum, round(rn.item(), 4), end="; ") # class float
            # m.momentum = momentum
            msum += rn
    print("BatchNormMoment.sum=%.3f, mom=(%.1E)" % (float(msum), momentum))
    return

def LimitBatchNorm(model: nn.Module, mean_max: float) -> None:
    "Limit BatchNorm"
    c = i = e = 0
    maxn: float = 0.0
    BatchNorm2d = nn.BatchNorm2d
    # assert(mean_max > 0.0), "positive"
    tt_clamp, vector_norm = tt.clamp, tt.linalg.vector_norm

    for i, m in enumerate(model.modules()):
        if isinstance(m, BatchNorm2d) and m.track_running_stats:
            m_running_mean = m.running_mean  # ord=inf
            mn: float = m_running_mean.norm(inf).item()
            if (mn < 1e38): maxn = max(mn, maxn) # nan/inf issue
            else: e += 1
            if (mn > mean_max) and (mean_max >= 1.0):
                m_running_var = m.running_var
                mi, ma = m_running_mean.min().item(), m_running_mean.max().item()
                vmin, vmax = m_running_var.min().item(), m_running_var.max().item()
                c += 1
                tt_clamp(m_running_mean, min=-mean_max, max=mean_max, out=m.running_mean)
                print("BNL(%d,%.3g<%.3g,%.3g<%.3g)" % (i, mi,ma, vmin,vmax), end=" ")
            else:
                if (mn > 100.0) and (mean_max < 1.0): print("[%d,%.2g]" % (i, mn), end="")

    if (mean_max < 1.0) or (c): print("LimitBatchNorm(%d/%d, n<%.3g, e=%d)" % (c, i, maxn, e))
    return

def ExportBatchNorm(model: nn.Module, silent:bool=False) -> list[tuple]:
    "Export BatchNorm to Tensor-List"
    BatchNorm2d = nn.BatchNorm2d
    bnl = [(m.running_mean.clone(), m.running_var.clone()) \
        if isinstance(m, BatchNorm2d) and m.track_running_stats else None for m in model.modules()]
    # for b in bnl: b.to(device_cpu)
    c:int = sum([b is not None for b in bnl])
    if not silent: print("ExportBatchNorm(lay=%d/%d)" % (c, len(bnl)))
    return bnl

    # c, bnl = 0, []
    # for m in model.modules():
    #     if isinstance(m, BatchNorm2d):
    #        c += len(m.running_mean)
    #        bnl.append(m.running_mean.clone())
    #        bnl.append(m.running_var.clone())

    print("ExportBatchNorm(lay=%d, nums=%d)" % (len(bnl), c * 2))
    return bnl

def ImportBatchNorm(model: nn.Module, bn_list, device=None) -> None:
    "Import BatchNorm from Tensor-List"
    BatchNorm2d = nn.BatchNorm2d
    # model.load_state_dict(model0.state_dict())
    # tt.load(PATH, map_location=device, weights_only=True)
    if (bn_list is None) or not len(bn_list):
        for m in model.modules():
            if isinstance(m, BatchNorm2d):
                m.reset_running_stats()
    else:
        if device is None:
            for m, b in zip(model.modules(), bn_list):
                if b is not None:
                    # assert isinstance(m, BatchNorm2d)  # .clone()
                    m.running_mean, m.running_var = b  # b[0], b[1]
        else:
            for m, b in zip(model.modules(), bn_list):
                if b is not None:
                    m.running_mean.copy_(b[0].to(device))
                    m.running_var.copy_( b[1].to(device))
        return

        # j: int = 0
        # for m in model.modules():
        #     if isinstance(m, BatchNorm2d):
        #         assert len(m.running_mean) == len(bn_list[j]), "wrong Tensor size"
        #         m.running_mean = bn_list[j + 0].clone()
        #         m.running_var  = bn_list[j + 1].clone()
        #         j += 2
        # assert len(bn_list) == j, "wrong list size"
    return

#def ResetBatchNorm(model: nn.Module) -> None:
#    "Reset BatchNorm (recover from NaN)"
#    i, j = 0, 0
#    for m in model.modules():
#        if isinstance(m, nn.BatchNorm2d):
#           print("BN(%d:%d,mn=%.6f,vn=%.6f), " % (j, len(m.running_mean), m.running_mean.norm().item(), m.running_var.norm().item()), end="")
#           m.reset_running_stats()
#           j += 1
#        i += 1
#    print(" BN.done=%d/%d" % (j, i))
#    # net.load_state_dict(copy_net.state_dict())

#@no_grad()
#def get_wd_params(model: nn.Module) -> None:
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
#                        nn.Linear,
#                        nn.Conv1d, nn.Conv2d, nn.Conv3d,
#                        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
#            ),
#        ):
#            # wd_params.append(m.weight)
#            j += 1
#        i += 1
#    print("(WeightDecay: %d/%d)" % (j, i))
#    return

def SetParam(model: nn.Module, par: tt.Tensor) -> None:
    "x-tensor into model"
    assert par.numel(), "empty Tensor"
    model.train()  # needed if net uses BatchNorm
    # normalisation layers use per-batch statistics + (activates Dropout layers)
    par = par.to(next(model.parameters()).device, get_default_dtype())

    s = e = 0
    for p in model.parameters():
        e += p.numel()
        p.data = par[s:e].reshape(p.shape) # .to(p.device, p.data.dtype)
        s = e
    return  # n=dim

def ParamLoad(model: nn.Module, fn:str="", nofilewarn:bool=True) -> bool:
    "load x vector from disk"  # TODO
    n: int = sum(p.numel() for p in model.parameters())
    assert(n >= 1), "empty model"
    from os.path import isfile

    if len(fn) < 2:  fn = "startx_*.pt"
    fn = fn.replace('*', str(int(n)))
    if not isfile(fn):
        if nofilewarn:
            print("ParamLoad(n=%d:%s)=NoFile." % (n, fn))
        return False
    # tt.load(model, 'path')
    par: tt.Tensor = tt.load(fn, weights_only=True) # model.load_state_dict()
    assert par.numel() == n, "wrong dimension"

    print("########################################")
    SetParam(model, par)
    pn: float = par.norm().item()  # vector_norm
    print("## ParamLoad(n=%d,av=%.3e,n2=%.3e)=OK" % (n, par.mean(), pn))
    print("########################################")
    return True

def GetParam(model: nn.Module) -> tt.Tensor:
    "model to x-tensor"
    # optimizer.state_dict('params')
    # with no_grad():
    return tt_cat( [par.data.flatten() for par in model.parameters()] )

def ParamSave(model: nn.Module, fn: str="") -> bool:
    "store x vector on disk" # tested
    if model is None:
        print("Warn:ParamSave(model=None), skip!")
        return False
    # tt.save(model, 'path')
    n: int = sum(p.numel() for p in model.parameters())
    assert(n >= 1), "empty model"
    from os.path import isfile

    if len(fn) < 2: fn = "lastx_*.pt"
    fn = fn.replace('*', str(int(n)))
    # if (path.isfile(fn)): print("ParamSave.overwrite(%s)" % fn)

    par: tt.Tensor = GetParam(model)
    tt.save(par, fn)

    pn: float = par.norm().item()  # vector_norm
    print("ParamSave(%s,av=%.3e,n2=%.3e)=OK" % (fn, par.mean(), pn))
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
        "get actual scaling - unused"
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
                self.maxscale *= 0.5  # not perfect
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
            return 0.0  # inf/nan in gradient

model_boost = None
diff_sum: tt.Tensor = None # debug only
elra_solver: bool = True
dog_solver: bool = None # DoG+LDoG
dog_averager = None # DoG+LDoG

# Creates a GradScaler once at the beginning of training.
scaler = None  # GradScaler()

def CheckElraSolver(optim) -> bool:
    "verify ELRA class"
    global elra_solver, dog_solver
    assert(optim is not None), "input <> None"

    elra_solver = hasattr(optim, 'GetParamAvg')
    name: str = str( type (optim).__name__ )
    print("ELRA:", elra_solver, ", name=", name) # ElraOptimizer
    if 'DoG' in name:
        dog_solver = True
    return elra_solver

def PrintParabola(abc, x1:float) -> None:
    "debug: check fit polnomial, y= a*x², a=y/x²"
    from math import nan
    a, b, c = abc[0], abc[1], abc[2]
    xs: float = float(-0.5 * b/a) if (a*a > 1e-20) else nan
    xr: float = (xs / x1) if (x1*x1 > 1e-20) else nan
    y1: float = (a * x1 + b) * x1 + c
    ys: float = (a * xs + b) * xs + c
    print("Parabola[%d](%.3e,%.3e,%.3e), (f1=%.3e,xs=%.3e:%.3f,ys=%.3e)" % (len(abc), a,b,c, y1,xs,xr,ys))
    return

# debug stuff
last_x: tt.Tensor = None  # debug only
dataldr_debug = None # short debug cache
# debug_model = None

def Interpolx(model, cost_func) -> None:
    "debug: valley cut loss"

    dl:DataLoader = dataldr_debug
    if dl is None: return
    global last_x
    x0 = GetParam(model)

    if last_x is None:
        last_x = x0
        return
    if x0.equal(last_x): return # collect-step

    from copy import deepcopy
    import numpy as np
    BatchNorm2d, tt_cat = nn.BatchNorm2d, tt.cat
    dev = x0.device
    scaling:float = 1.0 if scaler is None else min(scaler.scaling, 10000.0)
    inv_scaling = 1.0 / scaling
    len_dl:int = len(dl)
    bs:int = dl[0][1].numel()
    inv_len = 1.0 / len_dl

    print("##Interpolx: %dx%d" % (len_dl, bs), end=', ')
    m2 = deepcopy(model)
    bn0 = tuple([(m.running_mean.clone(), m.running_var.clone()) \
        if isinstance(m, BatchNorm2d) else None for m in model.modules()])
    full_step_len = x0.dist(last_x).item()
    full_vect = x0 - last_x
    realG = full_vect / full_vect.norm() # .item()
    g0n: tt.Tensor = x0 * 0.0  # normalized grad_0

    f01a = np.zeros(32, dtype=np.float32)
    f1: float = 0.1  # 2^(30/4) = 181
    for i in range(1, len(f01a) ):
        # f1: float = i * (1.0/10)
        f01a[i] = f1; f1 *= -2.0**0.25  # 0.15-27.2

    relx = f01a * full_step_len # np.linspace(0.0, full_step_len, 10+1)  # x-values
    last_loss = 0.0
    pos_loss, pos_grad = relx * 0.0, relx * 0.0  # np.arrays
    for f1 in f01a:  # range(10+1):
        m2.zero_grad(True)
        # tmp_loss, tmp_glen = tt.zeros(len_dl, device=dev), tt.zeros(len_dl, device=dev)
        tmp_loss = np.zeros(len_dl, dtype=np.float32)
        tmp_glen, tmp_gl2 = tmp_loss * 0.0, tmp_loss * 0.0
        par: tt.Tensor = last_x + f1 * full_vect # last_x * f1 + x0 * (1.0 - f1)
        SetParam(m2, par)
        for m, b in zip(m2.modules(), bn0):
            if b is not None:
                m, v = b
                m.running_mean, m.running_var = m.clone(), v.clone()
        gsum = None  # tensor
        for k, (X,y) in enumerate(dl):
            loss = cost_func(m2(X), y)
            tmp_loss[k] = loss.item()
            loss *= scaling
            loss.backward()
            g1 = tt_cat([p.grad.data.flatten() for p in m2.parameters()]) * inv_scaling
            # g1 = g1.to(dtype=tt.float32)
            m2.zero_grad(True)
            tmp_glen[k], tmp_gl2[k] = g1.norm().item(), g1.dot(realG).item()
            if gsum is None: gsum = g1
            else: gsum += g1

        # tmp_glen *= inv_scaling; gsum *= inv_scaling
        grad_norm: float = gsum.norm().item()
        loss_mean, loss_std = tmp_loss.mean(), tmp_loss.std()
        glm, agls = tmp_glen.mean(), grad_norm * inv_len
        # if not i: g0n, g0s = gsum, gsum.norm().item()  # once
        gls = tmp_gl2.mean() # / g0s  # sqrt(negative)
        # loss, _, _, _ = full_batch_loss(m2, None, dl, cost_function, x0.device)
        dy = 0.0 if (not i) else loss_mean-last_loss
        pos_loss[i], pos_grad[i] = loss_mean, gls # fvl.append(loss_mean)
        last_loss = loss_mean
        print("%.2f %.6f %.4e %.4e %.4e %.4e," % (f1, loss_mean, loss_std, dy, gls, glm), end='')

    # lossy = np.array(fvl)  # y-values = loss
    abc = np.polyfit(relx, pos_loss, 2, full=False)  # parabola(a,b,c)
    spx = np.nan if np.abs(abc[0]) < 1e-9 else float(-0.5*abc[1]/abc[0])
    k:float = abc[0] # (full_step_len**2)  # y= a*x², a=y/x²
    last_x = x0  # double usage (debug only)
    print("##cut(L=%.4e,%.4e,%.4e,k=%.4e)." % (pos_loss.mean(), spx, full_step_len, k), flush=True)
    # PrintParabola(abc, full_step_len)
    return

def WriteDsHist(reset:bool, log) -> None:
    "debug: histogram of epoch movement exponents"
    global diff_sum
    if (log is None) or (diff_sum is None):
        return
    if not log.closed:
        import numpy as np
        _ , e2 = tt.frexp(diff_sum) # get (int) float^2-exponents
        #    print("HIST(",np.min(e2),"<",np.max(e2),"),",np.mean(e2))
        #    print("P(1,50,99):",np.percentile(e2, 1),"<",np.percentile(e2, 50),"<",np.percentile(e2, 99))
        #    #np.median(e2, axis=None, overwrite_input=False)
        e2min, e2max = int(tt.min(e2)), int(tt.max(e2))
        bins: int = 1 + int(e2max - e2min)
        log.write("#hist=%d,s=%.3g,%.3g<%.3g,%d<%d," %
            (len(diff_sum),diff_sum.sum(), diff_sum.min(),diff_sum.max(), e2min, e2max))
        hist,_ = np.histogram(e2.numpy(), bins=bins, range=(e2min, e2max), density=False)
        # print(e2min, e2max, bins, hist);exit()
        log.write("%s\n" % str(hist))
        log.flush()
    if reset:
        diff_sum = None
    return

def WriteDominant(x: tt.Tensor, t:int = -1, log=None) -> int:
    "debug: plot strong components"
    global last_x, diff_sum
    if (x is None) or (x.numel() < 2): return 0
    len_x: int = x.numel()
    cpu = device_cpu
    # xmin, xmax = tt.min(x).item(), tt.max(x).item()
    # x = x.to(dtype=float)
    xsum, xnrm = x.sum().item(), x.norm().item()

    if (log is None):
        lf,fc = open("path_dom.dat", "a"), True
    else:
        lf,fc = log, False
    if (lf is None) or (lf.closed):
        print("WriteDominant(t=%d,n=%d):Error=fopen!" % (t, len_x))
        return -1

    if (last_x is None):
        print("Dom(0/%d, xs=%.3g,xn=%.3g, init)." % (len_x, xsum,xnrm))
        lf.write("##NEW,dim=%d,s=%.6g,n=%.6g\n" % (len_x, xsum, xnrm))
        if (fc): lf.close()
        last_x = x.clone()
        return 0

    assert len_x == last_x.numel(), "length(x_Tensors) differs"
    d, last_x = (x - last_x), x
    gmin, gmax = d.min().item(), d.max().item()
    gsum, gnrm = d.sum().item(), d.norm().item()

    th: float = max(abs(gmin), abs(gmax)) * 0.3
    db = d.abs() > th # [bool]
    nc: int = db.sum()
    d, db = d.to(cpu), db.cpu()

    if (diff_sum is None):
        diff_sum  = d*d
    else:
        diff_sum += d*d
    # print("Dom(%d/%d, xs=%.3g,xn=%.3g, ds=%.3g,dn=%.3g)." % (nc,len(x), xsum,xnrm, gsum,gnrm))
    lf.write("#t=%d,%d/%d, xs=%.3g,xn=%.3g, ds=%.3g,dn=%.3g, th=%.3g\n" %
        (t,nc,len_x, xsum,xnrm, gsum,gnrm, th))
    lf.write("-1,%.6g,%.6g\n" % (xsum, xnrm))

    if (gnrm > 0.0):
        x1 = x.to(cpu)
        for i in range(len_x):
            if db[i]:
                lf.write("%d,%.6g,%.3g\n" % (i, x1[i], d[i]))

    if fc: lf.close()
    return 1

# Device-Cache (new class) ..
utmt_DS: DataLdrFast = DataLdrFast()
utmt_TS: DataLdrFast = DataLdrFast()
utmt_VS: DataLdrFast = DataLdrFast()

def StatListStr(lst: list[float]) -> str:
    n : int = len(lst)
    if n < 2: return ("(len=%d<2!)" % n)
    tens = tt.tensor(lst)
    prv = tens[0]
    sum_ad : float = 0.0
    for i in range(1, n):
        v = tens[i]
        sum_ad += tt.abs(v - prv)
        prv = v
    val = -9.9 if (0.0==sum_ad) else (tens[n-1]-tens[0]) /sum_ad
    # print(len(lst),min(lst),sum(lst)/n,max(lst),val); exit(0)
    return ("(%d:%.3g<%.3g<%.3g:%.3f)" % (n, tens.min().item(), tens.mean().item(), tens.max().item(), val.item()))

def GrabBatches(data_loader:DataLoader, device, bc:int):
    "debug: keep some batches for valley cuts"

    img_type: tt.dtype = get_default_dtype()
    count: int = len(data_loader.dataset)
    count2: int = 0
    bc //= 4  # dl 4x
    ret = [None] * (bc * 4)

    for i, (X0, y0) in enumerate(data_loader):
        bs0: int = y0.numel()
        count2 += bs0
        if i >= bc: continue
        assert not (3 & bs0)  # mod 4
        if device is not None:
            X0, y0 = X0.to(dtype=img_type, device=device), y0.to(device)
        else:
            X0 = X0.to(dtype=img_type)
        i4: int = i * 4
        X, y = X0.split(bs0 >> 2), y0.split(bs0 >> 2)  # div 4
        for j in range(4):
            ret[i4 + j] = (X[j], y[j])
        # ret[i] = (X, y) if device is None else (X, y)

    print('GrabBatches:', len(data_loader), count, count2, len(ret))
    assert count == count2, 'len(DataLoader) strange'
    return tuple(ret)

model_output = None # (output.shape, output.dtype)
def NormalEval_Labels(model, model2, data_loader:DataLoader, cost_func, device, labels) -> tuple[int,int, float,float]:
    "full batch via normal DataLoader + single Label tensor (half D2H-copy)"
    tcat = tt_cat  # local ist faster
    prints: bool = False
    global model_output
    img_type: tt.dtype = get_default_dtype()  # float32
    corr1 = zeros(1, dtype=int64, device=device)
    corr2i: int = 0
    loss1 = zeros(1, dtype=float32, device=device)
    loss2 = 0.0
    output: tt.Tensor = None
    count: int = len(data_loader.dataset)
    t: float = time_time()
    bs = data1 = None
    # print("NormalEval_Labels ===== ", len(labels), count)
    assert labels.numel() == count, "sample counts differ"
    targets: tt.Tensor = labels.to(device) # 0..5 MB (int16 here)
    targets = targets.to(dtype=int64) # optional uint8 (if classes<256)
    tgl = outchk = None # targets.split(bs), output.chunk(2)

    if model2 is None:
        with no_grad():
            for i, (data0, target0) in enumerate(data_loader, -1):
                data = data1
                data1 = data0.to(device, img_type, non_blocking=True)
                if (bs is None):
                    bs = target0.numel()
                    bs2 = bs >> 1
                    tgl = targets.split(bs)
                    assert not (bs & 1), "bs not even"
                    continue

                # target, pos = targets[pos : pos + bs], pos2
                target = tgl[i]  # off by one

                # ResNet50
                # with autocast(device_type='cuda', dtype=tt.float16): # autocasting.
                if output is None:
                    output = tcat( [model(d) for d in data.chunk(2)] )
                    outchk = output.chunk(2)
                    # output = tcat( (model(data[:bs2]), model(data[bs2:])) )
                    model_output = (output.shape, output.dtype)
                else:
                    for o, d in zip(outchk, data.chunk(2)): o[:] = model(d)
                    # tcat( [model(d) for d in data.chunk(2)], out=output )
                    # output[:bs2], output[bs2:] = model(data[:bs2]), model(data[bs2:])
                loss1 += cost_func(output, target) # float16
                # test_loss += F.nll_loss(output, target, reduction="sum")
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).count_nonzero()

                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs  # len(target)

            if data1.numel():  # tail (shorter)
                output = None
                c1, l1 = single_eval_bs(data1, tgl[-1], model, cost_func, bs2)
                corr1 += c1; loss1 += l1  # * len(target)
                #c1, l1 = single_eval(data1[bs2:], target[bs2:], model, cost_func)
                #corr1 += c1; loss1 += l1
    else:
        corr2 = zeros(1, dtype=int64, device=device)
        loss2 = zeros(1, dtype=float32, device=device)

        with no_grad():
            for i, (data0, target0) in enumerate(data_loader, -1):
                data = data1
                data1 = data0.to(device, img_type, non_blocking=True)
                if (bs is None):
                    bs = target0.numel()
                    bs2 = bs >> 1
                    tgl = targets.split(bs)
                    assert not (bs & 1), "bs not even"
                    continue

                # target, pos = targets[pos : pos + bs], pos2
                target = tgl[i]  # off by one
                data_chunk = data.chunk(2)

                # output: tt.Tensor = model(data)
                if output is None:
                    output = tcat( [model(d) for d in data_chunk] )
                    outchk = output.chunk(2)
                else:
                    for o, d in zip(outchk, data_chunk): o[:] = model(d)
                    # tcat( [model(d) for d in data_chunk], out=output )
                    # output[:bs2], output[bs2:] = model(data[:bs2]), model(data[bs2:])
                loss1 += cost_func(output, target)
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).count_nonzero()

                # + model2
                for o, d in zip(outchk, data_chunk): o[:] = model2(d)
                # tcat( [model2(d) for d in data_chunk], out=output )
                # output[:bs2], output[bs2:] = model2(data[:bs2]), model2(data[bs2:])
                pred: tt.Tensor  = output.max(1, keepdim=True)[1]
                corr2 += pred.eq(target.view_as(pred)).count_nonzero()
                loss2 += cost_func(output, target)  # len(target)

                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs; loss2 *= bs  # len(target)

            if data1.numel():  # tail (shorter)
                output = None
                target = tgl[-1]  # targets[pos : ]
                for m, c, l in zip([model, model2], [corr1, corr2], [loss1,loss2]):
                    c1, l1 = single_eval_bs(data1, target, m, cost_func, bs2)
                    c += c1; l += l1
                    #c1, l1 = single_eval(data1[bs2:], target[bs2:], m, cost_func)
                    #c += c1; l += l1  # needs Tensor, not int
                # c1, l1 = single_eval(data1[:bs2], target[:bs2], model2, cost_func)
                # corr2 += c1; loss2 += l1 # * len(target)
                # c1, l1 = single_eval(data1[bs2:], target[bs2:], model2, cost_func)
                # corr2 += c1; loss2 += l1

        corr2i, loss2 = corr2.item(), loss2.item()

    if prints:
        avg: float = -1.0 if (count < 1) else loss1.item() / count
        print(".\n..FBL(%.4g+%.3g-%d)" % (avg, -1.0, 0), end=" ")

    return corr1.item(), corr2i, loss1.item(), loss2

def NormalEval(model, model2, data_loader:DataLoader, cost_func, device) -> tuple[int,int, float,float]:
    "full batch via normal DataLoader"
    tcat = tt_cat  # local ist faster
    prints: bool = False
    global model_output
    img_type: tt.dtype = get_default_dtype() # float32
    corr1 = zeros(1, dtype=int64, device=device)
    corr2i: int = 0
    loss1 = zeros(1, dtype=float32, device=device)
    loss2 = 0.0

    output: tt.Tensor = None
    pred: tt.Tensor = None
    count: int = len(data_loader.dataset)
    t: float = time_time()
    bs = None
    outchk = None  # output.chunk(2)
    data1 = target1 = None

    if model2 is None:
        with no_grad():
            for i, (data0, target0) in enumerate(data_loader):
                data, target = data1, target1
                data1, target1 = data0.to(device, img_type, non_blocking=True), \
                    target0.to(device, non_blocking=True)
                if (bs is None):
                    bs = target0.numel()
                    bs2 = bs >> 1
                    assert not (bs & 1), "bs not even"
                    continue

                # ResNet50
                # with autocast(device_type='cuda', dtype=tt.float16): # autocasting.
                if output is None:
                    output = tcat( [model(d) for d in data.chunk(2)] )
                    outchk = output.chunk(2)
                    # output = tcat( (model(data[:bs2]), model(data[bs2:])) )
                else:
                    for o, d in zip(outchk, data.chunk(2)): o[:] = model(d)
                    # tcat( [model(d) for d in data.chunk(2)], out=output )
                    # output[:bs2], output[bs2:] = model(data[:bs2]), model(data[bs2:])
                loss1 += cost_func(output, target)  # float16
                # test_loss += F.nll_loss(output, target, reduction="sum")
                pred = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).count_nonzero()

                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs  # len(target)

            if target1.numel():  # tail (shorter)
                output = None
                c1, l1 = single_eval_bs(data1, target1, model, cost_func, bs2)
                corr1 += c1; loss1 += l1 # * len(target1)
                #c1, l1 = single_eval(data1[bs2:], target1[bs2:], model, cost_func)
                #corr1 += c1; loss1 += l1 # * len(target1)
    else:
        corr2 = zeros(1, dtype=int64, device=device)
        loss2 = zeros(1, dtype=float32, device=device)

        with no_grad():
            for i, (data0, target0) in enumerate(data_loader):
                data, target = data1, target1
                data1, target1 = data0.to(device, img_type, non_blocking=True), \
                    target0.to(device, non_blocking=True)
                if (bs is None):
                    bs = target0.numel()
                    bs2 = bs >> 1
                    assert not (bs & 1), "bs not even"
                    continue

                data_chunk = data.chunk(2)
                if output is None:
                    output = tcat( [model(d) for d in data_chunk] )
                    model_output = (output.shape, output.dtype)
                    outchk = output.chunk(2)
                else:
                    for o, d in zip(outchk, data_chunk): o[:] = model(d)
                    # tcat( [model(d) for d in data.chunk(2)], out=output )
                    # output[:bs2], output[bs2:] = model(data[:bs2]), model(data[bs2:])
                loss1 += cost_func(output, target)
                pred = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).count_nonzero()
                # loss1 += cf # len(target)

                # + model2
                for o, d in zip(outchk, data_chunk): o[:] = model2(d)
                # tcat( [model2(d) for d in data_chunk], out=output )
                # output[:bs2], output[bs2:] = model2(data[:bs2]), model2(data[bs2:])
                pred = output.max(1, keepdim=True)[1]
                corr2 += pred.eq(target.view_as(pred)).count_nonzero()
                loss2 += cost_func(output, target) # len(target)

                if not (i & 15) and ((time_time() - t) >= 120.0):
                    print("<2m:%d:%d>" % (i, count), end=" ", flush=True)
                    prints, t = True, time_time()
            loss1 *= bs; loss2 *= bs  # len(target)

            if target1.numel():  # tail
                output = None
                for m, c, l in zip([model, model2], [corr1, corr2], [loss1,loss2]):
                    c1, l1 = single_eval_bs(data1, target1, m, cost_func, bs2)
                    c += c1; l += l1
                    #c1, l1 = single_eval(data1[bs2:], target1[bs2:], m, cost_func)
                    #c += c1; l += l1  # needs Tensor, not int
                # c1, l1 = single_eval(data1, target1, model, cost_func)
                # corr1 += c1; loss1 += l1  # * len(target1)

        corr2i, loss2 = corr2.item(), loss2.item()

    if prints:
        avg: float = -1.0 if (count < 1) else loss1.item() / count
        print(".\n..FBL(%.4g+%.3g-%d)" % (avg, -1.0, 0), end=" ")
    # print("NormalEval, inf-errors=", err, count)

    return corr1.item(), corr2i, loss1.item(), loss2

def FastCacheEval(model, model2, fdl: DataLdrFast, cost_function, device) -> tuple[int,int, float,float]:
    "full batch from fast data-loader-cache"
    count: int = fdl.dlf_samples  # len(data_loader.dataset)
    assert count > 0, "DataLdrFast available"
    assert fdl.dlf_Images.size(0) > 0, "DataLdrFast, no samples"
    n: int = sum(p.numel() for p in model.parameters())
    bsize: int = 1024 if (n != 23910152) else 16  # 24mio=Resnet50=TIN
    if (n == 11271432): bsize = 64  # TIN-RN18

    output: tt.Tensor = None
    pred: tt.Tensor = None

    loss1 = zeros(1, dtype=float32, device=device)
    loss2: float = 0.0
    corr1: tt.Tensor = zeros(1, dtype=int64, device=device)
    corr2i: int = 0

    _, zip_fdl = fdl.DirectBatchRead(bsize)
    # fdl.DirectBatch(-1)  # rewind

    if (model2 is None):
        with no_grad():
            for data, target in zip_fdl:
            # while True:
                # mbs, data, target = fdl.DirectBatch(bsize)
                # if (not mbs): break
                output = model(data) # TIN crash here (OOM)
                loss1 += cost_function(output, target) * target.numel()
                pred = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).count_nonzero()
    else:
        if device.index != next(model2.parameters()).device.index: model2.to(device)
        corr2: tt.Tensor = zeros(1, dtype=int64).to(device)
        loss2 = zeros(1, dtype=float32).to(device)
        with no_grad():
            for data, target in zip_fdl:
            # while True:
                # mbs, data, target = fdl.DirectBatch(bsize)
                # if (not mbs): break
                output = model(data) # TIN crash here (OOM)
                loss1 += cost_function(output, target) * target.numel()
                pred = output.max(1, keepdim=True)[1]
                corr1 += pred.eq(target.view_as(pred)).count_nonzero()

                output = model2(data)
                pred = output.max(1, keepdim=True)[1]
                corr2 += pred.eq(target.view_as(pred)).count_nonzero()
                loss2 += cost_function(output, target) * target.numel()

        corr2i, loss2 = corr2.item(), loss2.item()

    # print("FastCacheEval, inf-errors=", err, count)
    fdl.dlf_EpochCount += 1 # Test
    return corr1.item(), corr2i, loss1.item(), loss2

def TestEval(model, model2, test_loader:DataLoader, fdl, cost_function, device) -> tuple[float,float,float,float]:
    "calc Accuracy + TestLoss over full test-dataset"
    # model.eval() --
    if test_loader is None: return -1.0, -1.0, -1.0, -1.0
    count: int = len(test_loader.dataset)
    if count < 1: return -1.0, -1.0, -1.0, -1.0

    # global utmt_TS, utmt_VS
    # is_tst: bool = (utmt_TS.dlf_samples + utmt_VS.dlf_samples) > 0 and utmt_TS.dlf_samples == count
    # fdl: DataLdrFast = utmt_TS if is_tst else utmt_VS
    fdl_samples: int = 0 if fdl is None else fdl.dlf_samples
    # only_lab: bool = fdl.dlf_only_lab
    test_loss: float = 0.0  # zeros(1, dtype=float32).to(device)
    test_loss2: float = 0.0
    dt: float = time_time()

    if count != fdl_samples or fdl.dlf_only_lab:  # slower
        if model2 is not None:
            assert device.index == next(model2.parameters()).device.index, 'TestEval'
        if fdl.dlf_only_lab:
            corr, corr2, test_loss, test_loss2 = NormalEval_Labels(
                model, model2, test_loader, cost_function, device, fdl.dlf_Labels)
        else:
            corr, corr2, test_loss, test_loss2 = NormalEval(
                model, model2, test_loader, cost_function, device)

    else:  # tensors already in device-memory/GPU
        corr, corr2, test_loss, test_loss2 = \
            FastCacheEval(model, model2, fdl, cost_function, device)

    print("(%d)TL=%.3fs," % (-0, time_time() - dt), end=" ") # debug
    if count > 0:
        # assert(corr > 0), "test accu > 0.0"
        s: float = 1.0 / count
        if model2 is not None:
            return corr * s, test_loss * s,  corr2 * s, test_loss2 * s
        else:
            return corr * s, test_loss * s,  -1.0, -1.0

    from math import nan
    return nan, nan, nan, nan

def FastenDataloader(full_data:tuple, num_classes:int, maxMB:int=800, device=device_cpu) -> None:
    "Intial Load Dataset into CUDA/GPU for FullBatchOperation"
    global utmt_TS, utmt_DS, utmt_VS
    train_data, test_data, val_data = full_data  # DataLoader
    dt: float = time_time()

    if (maxMB <= 0) or (train_data is None): # or (str(device).find("cuda") < 0):
        if (train_data is None):
            utmt_TS.InitVectors()
            utmt_DS.InitVectors()
            utmt_VS.InitVectors()
        else:
            print("FastenDataloader(skip, dev=%s)." % (str(device)))
        return # dont skip here, even 2x faster on CPU !

    len_val_ds: int = -1 if val_data is None else len(val_data.dataset)
    if 1:  # bypass (cache) timm-DL to be faster
        # classes: int = len(test_data.dataset.classes) if hasattr(test_data.dataset, "classes") else -1
        print("FastenDataloader(%d+%d+%d), cls=%d .." % (len(train_data.dataset), len(test_data.dataset), len_val_ds, num_classes))
        utmt_TS.Import_DataLoader(test_data, num_classes, is_train=False, maxMB=950, device=device)
        if len_val_ds > 0:
            utmt_VS.Import_DataLoader(val_data, num_classes, is_train=False, maxMB=222, device=device)
        if (utmt_TS.dlf_samples > 0): # both or none
            utmt_DS.Import_DataLoader(train_data, num_classes, is_train=False, maxMB=1400, device=device) # DISABLE DS HERE
        else:  # for optim.SetClasses()
            utmt_DS.dlf_init_mbs = DataLdrFast.EstimBatchSize(train_data)
            utmt_DS.dlf_label_max = utmt_TS.dlf_label_max
        dt = time_time() - dt
        print("FastenDataloader(%d+%d+%d), dt=%.3fs.\n" % (utmt_DS.dlf_samples, utmt_TS.dlf_samples, utmt_VS.dlf_samples, dt))
        DataLdrFast.CheckPair(utmt_DS, utmt_TS) # FilePairConflicts
        # print(utmt_DS.GetDsHash(), utmt_TS.GetDsHash()) #;exit() # debug
        return

    print("FastenDataloader(%d+%d+%d)=SKIP.\n" % (len(train_data.dataset), len(test_data.dataset), len_val_ds))
    return

def full_batch_loss(model, model2, data_loader:DataLoader, cost_function, device) -> tuple[float,float,float,float]:
    "calc. full batch loss (over all train data), only for printing (no effect on solver)"
    if (data_loader is None) or not len(data_loader.dataset): return nan, nan, nan, nan
    global utmt_DS
    i: int = 0
    t0: float = time_time()
    # dc: int = cuda.device_count()
    total_loss: float = 0.0
    total_loss2: float = 0.0
    count: int = len(data_loader.dataset)

    if (utmt_DS.dlf_samples > 1):  # fast - train
        assert len(data_loader.dataset) <= utmt_DS.dlf_samples, "shrinked < Gpu_DL"
        corr, corr2, total_loss, total_loss2 = \
            FastCacheEval(model, model2, utmt_DS, cost_function, device)
        i = 1

    if not i:  # tensors not in device-memory/GPU
        if utmt_DS.dlf_only_lab:
            corr, corr2, total_loss, total_loss2 = NormalEval_Labels(
                model, model2, data_loader, cost_function, device, utmt_DS.dlf_Labels)
        else:
            corr, corr2, total_loss, total_loss2 = \
                NormalEval(model, model2, data_loader, cost_function, device)

    # utmt_DS.dlf_EpochCount += 1 # Train
    t0 = time_time() - t0
    if t0 > 50.0: print("sec=%.1f/100%%(-), " % (t0), end=" ")
    # 100.0*count/len(data_loader.dataset) # [256..,last=96],
    if count > 0:
        s: float = 1.0 / count
        if (model2 is None):
            return corr * s, float(total_loss) * s,  -1.0, inf
        else:
            return corr * s, float(total_loss) * s, corr2 * s, float(total_loss2) * s

    # from math import nan
    return nan, nan, nan, nan

# <2m:215:26784> /home/telematik/.venv/lib/python3.11/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
#  return F.conv2d(input, weight, bias, self.stride,

booster_on: bool = False

def cosine(a: tt.Tensor, b: tt.Tensor) -> float:
    "cosine between tensor pair (debug only)"
    # return nn.functional.cosine_similarity(a, b, dim=1, eps=1e-30)
    assert a.numel() == b.numel(), "dot-product dim-conflict"
    d: tt.Tensor = tt.dot(a, b)
    return 0.0 if (d.abs() < 1e-6) else (d / (a.norm() * b.norm())).item()

def PrepareBoostModel(model:nn.Module, device2, optim) -> tuple[nn.Module,int,float]:
    "internal init"
    boost: bool = booster_on and elra_solver
    if not boost: return None, 0, 0.0

    bcnt, avg, avg_x = optim.GetParamAvg(True)
    if avg_x is None:
        print("Warn:FBL.GetParamAvg(c=%d,f=%.3g), x=None, skip-boost!" % (bcnt, avg))
        return None, 0, 0.0

    global model_boost  # needed?
    model2: nn.Module = model_boost
    assert(model  is not None), "need source model"
    assert(model2 is not None), "need 2nd model"

    if (device2 is not None): model2.to(device2)

    ImportBatchNorm(model2, ExportBatchNorm(model), device=device2)  # copy BN
    SetParam(model2, avg_x)
    # model2.eval() # optional (batch norm update)
    return model2, bcnt, avg

def LossRatio(loss_trn:float, loss_val:float) -> float:
    return loss_val/loss_trn if (loss_trn > 0.0) else loss_val-loss_trn

def AccuRatio(accu_trn:float, accu_val:float) -> float:
    a1t, a1v = 1.0 - accu_trn, 1.0 - accu_val
    return a1v/a1t if (a1t > 0.0) else a1v-a1t

def combined_FBL(model, full_data:tuple, cost_func, device, optim) -> tuple[float,float,float,float]:
    "Calc 4 Loss+Accu f(x + boost) x2 (train + test)"
    # from copy import deepcopy
    model2, bcnt, avg = PrepareBoostModel(model, None, optim)
    # boost: bool = booster_on and elra_solver
    train_fast, test_data, val_data = full_data

    assert len(train_fast), "empty dataset train"
    assert len(test_data),  "empty dataset test"
    # model1 = deepcopy(model)
    accu1t, loss1t, accu2t, loss2t = full_batch_loss(model, model2, train_fast, cost_func, device)
    accu1v, loss1v, accu2v, loss2v = TestEval(model, model2, test_data, utmt_TS, cost_func, device)  # test
    if val_data is not None and len(val_data):
        a1val, l1val, a2val, l2val = TestEval(model, model2, val_data, utmt_VS, cost_func, device)  # val
        if elra_solver: optim.SetValidLoss(l1val, l2val)
        print('VAL: a=%.4f,l=%.4f, boost: a=%.4f,l=%.4f' %
            (a1val, l1val, a2val, l2val))

    t2t, a2a = LossRatio(loss1t, loss1v), AccuRatio(accu1t, accu1v)
    if elra_solver: optim.SetTrainLoss(loss1t, loss2t)
    print("*Train1 (%.3f%% %.6g), Test1 (%.3f%% %.6g), t2t=%.2f,a2a=%.2f, n=%d" %
        (accu1t*100, loss1t, accu1v*100, loss1v, t2t,a2a, bcnt))
    if model2 is not None:
        t2t, a2a = LossRatio(loss2t, loss2v), AccuRatio(accu2t, accu2v)
        print("  *Boost2 (%.3f%% %.6g), Test2 (%.3f%% %.6g), t2t=%.2f,a2a=%.2f, AvLf=%.3g" %
            (accu2t*100, loss2t, accu2v*100, loss2v, t2t,a2a, avg))
        if elra_solver: optim.TellTrainBoostLoss(loss2t)
        #for i in range(10+1):  # debug-2025
        #    t: float = i * 0.01
        #    xt = optim.GetBoostFuture(t)
        #    if xt is None: break
        #    SetParam(model2, xt)
        #    a3val, l3val, _, _ = TestEval(model2, None, val_data, utmt_VS, cost_func, device)
        #    print("Boost3s(%.2f) a=%.3f%%, l=%.4f." % (t, a3val*100, l3val))
        global last_x
        if last_x is not None: last_x = GetParam(model2) # debug
    else:
        accu2t = loss2t = accu2v = loss2v = -1.0
    return accu1t, loss1t, accu2t, loss2t,  accu1v, loss1v, accu2v, loss2v

smp_return: tuple = None
def thread_eval(x: tt.Tensor, y: tt.Tensor, model, cost_func) -> None:
    "internal: threaded SMP/DDP loss"

    # assert next(model.parameters()).device == x.device, "SMP device"
    # with no_grad():
    output: tt.Tensor = model(x)
    # loss = cost_func(output, y) # * len(y)
    pred: tt.Tensor  = output.max(1, keepdim=True)[1]
    # corr = pred.eq(y.view_as(pred)).sum()

    global smp_return
    smp_return = (pred.eq(y.view_as(pred)).count_nonzero(), cost_func(output, y))
    return

def single_eval(x: tt.Tensor, y: tt.Tensor, model, cost_func) -> tuple[int, float]:
    "internal: single loss+accu"
    n: int = y.numel()  # len(y)
    if not n: return 0, 0.0

    with no_grad():
        output: tt.Tensor = model(x)
    # loss = cost_func(output, y) # * len(y)
    pred: tt.Tensor  = output.max(1, keepdim=True)[1]
    # corr = pred.eq(y.view_as(pred)).sum()
    return pred.eq(y.view_as(pred)).count_nonzero(), cost_func(output, y) * n

def single_eval_bs(x: tt.Tensor, y: tt.Tensor, model, cost_func, bs:int) -> tuple[int, float]:
    "internal: single loss+accu (split into bs)"
    n: int = y.numel()  # len(y)

    if n <= bs:
        if not n: return 0, 0.0
        with no_grad():
            output: tt.Tensor = model(x)
        # loss = cost_func(output, y) # * len(y)
        # corr = pred.eq(y.view_as(pred)).sum()
    else:
        with no_grad():
            # for xx in x.split(bs): outs.append( model(xx) )
            output: tt.Tensor = tt_cat([ model(xx) for xx in x.split(bs) ])
    # all batches together
    pred: tt.Tensor  = output.max(1, keepdim=True)[1]
    return pred.eq(y.view_as(pred)).count_nonzero(), cost_func(output, y) * n

def calc_FBL_m2d2(model1, model2, data_loader, cost_func, device1, device2) -> tuple[float,float, float,float]:
    "full-batch-eval for 2x model at 2 devices"

    from threading import Thread
    count: int = len(data_loader.dataset)
    assert count > 1, "empty dataset (FBL)"
    assert model2 is not None, "debug smp"
    img_type: tt.dtype = get_default_dtype()
    loss1 = zeros(1, dtype=float32, device=device1)
    loss2 = zeros(1, dtype=float32, device=device2)
    corr1 = zeros(1, dtype=int64, device=device1)
    corr2 = zeros(1, dtype=int64, device=device2)
    bs = None
    global smp_return
    smp_return = None  # tuple(2)

    with no_grad():
        for data0, target0 in data_loader:
            x1, y1 = data0.to(device1, img_type, non_blocking=True), \
                target0.to(device1, non_blocking=True)
            x2, y2 = data0.to(device2, img_type, non_blocking=True), \
                target0.to(device2, non_blocking=True)
            if (bs is None):
                bs = target0.numel()
            else:
                if target0.numel() != bs: break

            smp_return = None  # tuple(2)
            th1 = Thread(target=thread_eval, args=(x2,y2, model2, cost_func))
            th1.start()  # ((((

            output: tt.Tensor = model1(x1)
            loss1 += cost_func(output, y1)  # * len(target0)
            pred: tt.Tensor  = output.max(1, keepdim=True)[1]
            corr1 += pred.eq(y1.view_as(pred)).count_nonzero()

            th1.join(timeout=40.0)  # 40s, ))))
            # assert(smp_return is not None), "debug: calc_FBL_m2d2"
            c2, l2 = smp_return  # !!!!
            corr2 += c2; loss2 += l2

    data0 = output = None
    loss1 *= bs; loss2 *= bs
    tlen: int = target0.numel()

    if tlen and tlen < bs:
        c2, l2 = single_eval(x1,y1, model1, cost_func)
        corr1 += c2; loss1 += l2  # * tlen

        c2, l2 = single_eval(x2,y2, model2, cost_func)
        corr2 += c2; loss2 += l2

    smp_return = None
    s: float = 1.0 / count

    return corr1.item() * s, loss1.item() * s, corr2.item() * s, loss2.item() * s

def combined_FBL_smp(model, full_data, cost_func, dev_list, optim) -> tuple[float,float,float,float]:
    "parallel (SMP = 2 models on 2 GPUs) calc of FullBatchLoss for normal + boost"
    train_fast, test_data, val_data = full_data
    device, device2 = dev_list[0], dev_list[1]
    model2, bcnt, avg = PrepareBoostModel(model, device2, optim)
    dt: float = time_time()

    if model2 is not None:
        print("combined_FBL_smp(%s + %s), <<<<<" % (str(device), str(device2)))
        accu1t, loss1t, accu2t, loss2t = calc_FBL_m2d2(model, model2, train_fast, cost_func, device, device2)
        accu1v, loss1v, accu2v, loss2v = calc_FBL_m2d2(model, model2, test_data, cost_func, device, device2)
        dt = time_time() - dt
        print("combined_FBL_smp(dt=%.2f) done.  >>>>>" % dt)
        print("*Train1 (%.3f%% %.6g), Test1 (%.3f%% %.6g), t2t=%.2g, n=%d" % (accu1t*100, loss1t, accu1v*100, loss1v, LossRatio(loss1t, loss1v), bcnt))
        print("  *Boost2 (%.3f%% %.6g), Test2 (%.3f%% %.6g), t2t=%.2g, AvLf=%.3g" % (accu2t*100, loss2t, accu2v*100, loss2v, LossRatio(loss2t, loss2v), avg))
        if elra_solver: 
            optim.TellTrainBoostLoss(loss2t)
            optim.SetTrainLoss(loss1t, loss2t)
        for i in range(10+1):
            t: float = i*0.1
            # xt = optim.GetBoostFuture(t)
    else:
        accu1t, loss1t, accu2t, loss2t = full_batch_loss(model, None, train_fast, cost_func, device)
        accu1v, loss1v, accu2v, loss2v = TestEval(model, None, test_data, utmt_TS, cost_func, device)
        print("*Train1 (%.3f%% %.6g), Test1 (%.3f%% %.6g), t2t=%.2g, n=%d" % (accu1t*100, loss1t, accu1v*100, loss1v, LossRatio(loss1t, loss1v), bcnt))
        accu2t = loss2t = accu2v = loss2v = -1.0
        if elra_solver: optim.SetTrainLoss(loss1t, loss1t)
    
    if val_data is not None and len(val_data):
        # model2.to(device)  # move back to device1 ??
        a1val, l1val, a2val, l2val = TestEval(model, model2, val_data, utmt_VS, cost_func, device)  # val
        if elra_solver: optim.SetValidLoss(l1val, l2val)
        print('VAL: a=%.4f,l=%.4f, boost: a=%.4f,l=%.4f' %
            (a1val, l1val, a2val, l2val))

    return accu1t, loss1t, accu2t, loss2t,  accu1v, loss1v, accu2v, loss2v

class LossHistRb:
    "ringbuffer for running average of loss history"
    def __init__(self) -> None:
        self.hist_loss = zeros(128, dtype=float32, device=device_cpu)
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
    return loss0 * 1.5 if (loss0 > 0.0) else (loss0 + 1.0)  # optim.GetLossLimit()

def train_step_adam(X:tt.Tensor, y:tt.Tensor, model:nn.Module, loss_func, optim:tt.optim.Optimizer, limitf:float, no_scaler) -> float:
    "Training step for Adam/Lion/SGD/DOG."
    global utmt_LossHist  # hist_loss, hist_lpos

    optim.zero_grad(set_to_none=True) # (optional: skip for batch-combining)

    loss = loss_func(model(X), y)
    loss_item: float = loss.item()  # already fix

    if (loss_item < 1e999):  # isnan(loss_item), 1e999==inf (fastest)
        utmt_LossHist.add_loss(loss_item)
    else:
        n2, nM = GetParamxMax(model, loss.device)
        print("Warn:training_step(loss=nan)!", loss_item, n2, nM)

    if True:
        loss.backward() # computes gradient for normal opt (Adam etc)
        optim.step()
        if (dog_averager is not None): dog_averager.step() # DoG+LDoG

    return loss_item

def training_step_scaler(X:tt.Tensor, y:tt.Tensor, model:nn.Module, loss_func, optim:tt.optim.Optimizer, limitf:float, scaler) -> float:
    "Training step for torch model + scaler, ELRA (P2M+C2M)"
    global utmt_LossHist # hist_loss, hist_lpos

    # optim.zero_grad(set_to_none=True) # (optional: skip for batch-combining)
    model.zero_grad(set_to_none=True)  # 1:1

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
    grads = tt_cat( [p.grad.data.flatten() for p in model.parameters()] )
    absmax: float = grads.norm(inf).item()  # GetGradMax(model)
    if (absmax < 1e999):  # no overflow step, early skip
        grads *= scaler.inv_scaling  # in-place
        optim.step_noscale(loss_item, grads)  # bug fix
        # optim.step(loss_item, scaler.inv_scaling)  # 2x gradient extraction (time)
    scaler.UpdateGradNorm(absmax)

    # torch\cuda\amp\grad_scaler\GradScaler
    #scaler.scale(loss).backward()
    #scaler.step(optim, loss_item, 1.0)
    #scaler.update()
    # Python311\site-packages\torch\cuda\amp\grad_scaler.py

    return loss_item

def training_step(X:tt.Tensor, y:tt.Tensor, model:nn.Module, loss_func, optim:tt.optim.Optimizer, limitf:float, unused) -> float: 
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

def thread_step_smp(X:tt.Tensor, y:tt.Tensor, model:nn.Module, loss_func, p0, limitf:float, scaling:float, device1) -> float:
    "Training step for torch model + scaler + MP, ELRA (P2M+C2M)"
    global smp_return
    # scaling: float = scaler.scaling # freeze scaler

    model.zero_grad(set_to_none=False) # model2
    if (p0 is not None): SetParam(model, p0) # sync param 1->2

    loss = loss_func(model(X), y)
    loss_item: float = loss.item()
    # print("thread_step_smp", loss_item, type(smp_return))

    if not (loss_item < limitf):  # 1e999==inf (fastest)
        n2, nM = GetParamxMax(model, loss.device)
        print("Warn:training_step(loss2=nan), SMP!", loss_item, limitf, n2, nM)
        # optim.step_retrace(loss_item)
        smp_return = (loss_item, None)
        return

    # MyGradScaler:
    loss *= scaling
    loss.backward()  # computes gradient for P2M+C2M
    grads = tt_cat( [p.grad.data.flatten() for p in model.parameters()] )
    absmax: float = grads.norm(inf).item()  # GetGradMax(model)

    if (absmax < 1e999):  # no overflow step, early skip
        # optim.step(loss_item, None) # lgs_dev=None)
        grads *= 1.0 / scaling  # in-place
        smp_return = (loss_item, grads.to(device1, non_blocking=True) )
        return
    # scaler.UpdateGradNorm(absmax) # not in MP

    print("model2: loss=%.3e, |grad|=%.3e, SMP!" % (loss_item, absmax))
    smp_return = (None, None)
    return

def step_smp(x1,y1, x2,y2, model1,model2, optim, cost_func, device1,device2, limitf:float, p0, thread) -> bool:
    "internal: parallel step() SMP"
    # from threading import Thread # Lock
    global smp_return # scaler

    fin, _ = optim.CheckNextStep()
    if fin:  # final/real step expected (no SMP/DDP)
        loss1 = training_step_scaler(x1,y1, model1, cost_func, optim, limitf, scaler)
        p1 = optim.GetParam(True, device2)
        return loss1, False, p1

    p1 = smp_return = None # tuple(2)
    th1 = thread(target=thread_step_smp, args=(x2,y2, model2, cost_func, p0, limitf, scaler.scaling, device1))
    th1.start() # ((((

    loss1 = training_step_scaler(x1,y1, model1, cost_func, optim, limitf, scaler)
    p1 = optim.GetParam(True, device2)
    # thread_step_smp(data2[0: bs], target2[0: bs], model2, cost_func, None, limitf, scaler, device1)
    th1.join(timeout=40.0) # 40s, ))))
    assert(smp_return is not None), "train thread timeout (increase timeout for huge nets)"

    if p1 is not None: return loss1, False, p1 # drop 2nd result, reuse x2,y2

    loss2, grad = smp_return
    if (grad is not None):
        optim.step_noscale(loss2, grad)
    else:
        if (loss2 is not None): optim.step_retrace(loss2)
    p1 = optim.GetParam(True, device2)
    
    return loss1, True, p1  # ok (both threads used)

def grad_smp(xy:tuple, model, p0: tt.Tensor, clsd:tuple) -> tuple:
    "multi-GPU gradient worker thread (SMP)"
    cost_func, limitf, scaling, dev0 = clsd
    X, y = xy
    model.zero_grad(set_to_none=False)
    if (p0 is not None): SetParam(model, p0)

    loss = cost_func(model(X), y)
    loss_item: float = loss.item()

    if not (loss_item < limitf): return loss_item, None

    # MyGradScaler:
    loss *= scaling
    loss.backward()  # computes gradient for P2M+C2M
    grads = tt_cat( [p.grad.data.flatten() for p in model.parameters()] )
    absmax: float = grads.norm(1e999).item()  # 1e999 = inf
    if (absmax < 1e999):
        grads *= 1.0 / scaling
        return loss_item, grads.to(dev0)
    return loss_item, None

def ManyCopy(p1: tt.Tensor, mod_list:list, dev_list:list) -> list:
    "copy param into multiple models (skip master)"
    if p1 is None: return [None] * len(dev_list)
    d2d = [p1.to(d, non_blocking=True) if i else None for i, (d) in enumerate(dev_list)]
    cuda.synchronize()
    return d2d
    for i, (m, d) in enumerate(zip(mod_list, d2d)):
        if not i: continue  # skip 1st device (master)
        SetParam(m, d)
    return

def SelectStepCall(elra_solver:bool, scaler) -> callable:
    "switch step() callable"
    if elra_solver:
        tstep = training_step if (scaler is None) else training_step_scaler
    else:
        tstep = train_step_adam
    assert callable(tstep), "need function for step()"
    return tstep

def train_epoch_smp(dataloader, mod_list:list, optim, cost_func, dev_list:tuple, loss0: float):
    "MultiGpu training (SMP)"
    # assert device2 is not None, "needs 2 GPUs"
    # assert model2 is not None, "needs 2 models"
    # global smp_return # scaler
    # from threading import Thread # inside step_smp()

    tstep = SelectStepCall(elra_solver, scaler) # training_step_scaler()
    _, collects = optim.CheckNextStep()
    # assert(collects >= 2), "else useless with SMP/DDP" # zero before init
    assert(scaler is not None), "todo (only 16bit for now)"
    from concurrent.futures import ThreadPoolExecutor # ProcessPoolExecutor
    tpe = ThreadPoolExecutor(max_workers=4)
    # executor = ProcessPoolExecutor(4)

    img_type: tt.dtype = get_default_dtype()
    limitf: float = CalcLossLimit(loss0)
    ldl: int = len(dataloader)
    assert ldl > 2, "empty dataloader"
    pp100: float = 100.0 / ldl
    nextpp, ldlp = 0, (ldl * 2) // 100  # progress piece
    ldlp = max(10, ldlp)
    prev_bs: int|None = None
    err2cnt = 0
    dc: int = len(dev_list)
    assert dc >= 2, "need multiple devices, 2 or 4"
    assert len(mod_list) == dc, "models <> devices"
    device1, model1, d2d0 = dev_list[0], mod_list[0], [None]*dc
    grad_dc: float = 1.0 / dc
    clsd = (cost_func, limitf, scaler.scaling, device1)
    p1: tt.Tensor = optim.GetParam(False, device_cpu) # GetParam(model1).to(device_cpu)
    assert p1 is not None, "optim.GetParam() => None"
    grads = tt.zeros(0, device=device1)  # tt.sum(,out=)
    # Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context...
    print("train_epoch_smp(%s, bc=4x%d)  <<<<<" % (str(dev_list), ldl), flush=True)
    res_list = xy_series = None
    t1 = dt = time_time()

    # if p1 is not None:
    ebn = ExportBatchNorm(model1, True)
    for m, d in zip(mod_list, ManyCopy(p1, None, dev_list)):
        if d is not None:
            SetParam(m, d)
            ImportBatchNorm(m, ebn, device=d)
    ebn = p1 = None

    for batch_idx, (data0, target0) in enumerate(dataloader):
        if prev_bs is None:
            prev_bs = target0.numel()  # new multi batches
            assert not (prev_bs % 32), "tensor core alignment (N*8 @FP16)"
            bs = prev_bs // (4 * dc)  # DIV by 4, single-GPU / step
            bs2 = bs * (4 // dc)  # batch-block / GPU
            bs2bs: bool = (bs2 == bs)  # is (4==dc)
        else:
            if target0.numel() != prev_bs: break  # skip tail

        # reduce cudaMemcpy(H2D) by multi-batch transfer
        xy_series = [None] * 4  # 4x32  (4 steps / memcopy)
        dev_pairs = [(x.to(d, img_type, non_blocking=True), y.to(d, non_blocking=True))
                for x,y,d in zip(data0.chunk(dc), target0.chunk(dc), dev_list)]
        xysp = [(x.split(bs), y.split(bs)) for x,y in dev_pairs]
        if bs2bs:  # 4 GPUs
            for i in range(4):  # i = splits per device (4)
                xy_series[i] = [(x[i], y[i]) for x,y in xysp]
        else:  # 2 GPUs (2 batches / h2d-copy), 2x2
            # xysp = [(x.split(bs), y.split(bs)) for x,y in dev_pairs] # 2x2
            assert 2 == len(xysp), "batch count internal smp"
            for i in range(4):  # i = splits per device (4)
                xy_series[i] = [(x[i], y[i]) for x,y in xysp]
        
        # print('xy_series', len(xy_series), type(xy_series[0][0][0])) # debug
        for xyd in xy_series:
            p1 = optim.GetParam(True, device_cpu)
            if p1 is None:
                res_list = [tpe.submit(grad_smp, p, m, None, clsd) for p, m in zip(xyd, mod_list)]
            else:
                #ebn = ExportBatchNorm(model1, True)  # optional (no effect, why?)
                #for m, d in zip(mod_list, ManyCopy(p1, None, dev_list)):
                #    if d is not None: ImportBatchNorm(m, ebn, device=d)
                #ebn = None
                d2d = ManyCopy(p1, None, dev_list); p1 = None
                res_list = [tpe.submit(grad_smp, p, m, t0, clsd) for p, m, t0 in zip(xyd, mod_list, d2d)]

            res_list = [r.result(timeout=30) for r in res_list]
            losses = tt.tensor([l[0] for l in res_list], dtype=float32, device=device_cpu)
            loss_item: float = losses.mean().item()
            if loss_item < limitf:
                gl = [l[1] for l in res_list]
                nogs: int = sum([g is None for g in gl])
                if nogs: print("SMP no_grads:", losses, nogs, clsd[2])
                else:
                    # gs:float = 1.0 / len(res_list)
                    # tt.sum(tt.stack([l[1].to(device=device1) for l in res_list]), dim=0, out=grads)
                    tt.add(gl[0], gl[1].to(device1), out=grads)  # device=cpu!
                    for l in gl[2: ]:
                        tt.add(grads, l.to(device1), out=grads)
                    grads *= grad_dc
                    optim.step_noscale(loss_item, grads) # skip: /len(res_list)
            else:
                optim.step_retrace(loss_item)

        if (batch_idx >= nextpp):
            t: float = time_time()
            t1, t = t, t - t1
            print("Epoch progress: [%d/%d=%.0f%%]  Loss1=%.3e, dt=%.1fs" %
                (batch_idx, ldl, pp100*batch_idx, loss_item, t), flush=True)
            nextpp += ldlp

    tpe.shutdown()
    res_list = grads = gl = x2 = y2 = None
    tail: int = target0.numel()
    if tail and (target0.numel() < prev_bs):
        data0, target0 = data0.to(device1, img_type, non_blocking=True), target0.to(device1, non_blocking=True)
        bs2 = bs * dc  # bs2=32
        for x, y in zip(data0.split(bs2), target0.split(bs2)):
            if y.numel() == bs2:
                l1 = tstep(x, y, model1, cost_func, optim, limitf, scaler)
            else:
                print("(skip tail of %d/%d samples)" % (y.numel(), target0.numel()))

    dt = time_time() - dt
    print("train_epoch_smp(bc=%d, [], dt=%.2f) done.  >>>>>" % (batch_idx, dt), flush=True)
    redo1 = p1 = smp_return = None
    model1.zero_grad(set_to_none=True)
    for m in mod_list:
        m.zero_grad(set_to_none=True)
    return [0], [0], [0], [0], [0]

def train_epoch(dataloader, model, optim, cost_func, device, loss0: float):
    "train single epoch"
    losses, batches = [], []
    steps = []
    f_calls, g_calls = [], []
    last_time: float = -1.0
    tl_tmp: str = ""

    # global scaler # booster_on, utmt_DS
    limitf: float = CalcLossLimit(loss0)
    tstep = SelectStepCall(elra_solver, scaler)
    if elra_solver: optim.GetParamAvg(booster_on) # reset

    # utmt_DS.FullShuffle(0) # todo:test
    # print(utmt_DS.GetDsHash()) # check data integrity by sums
    # utmt_DS.SetMiniBatchSize(0, shuffle=True) # reorder batches
    # utmt_DS.ShuffleBatch(rewind=True)

    len_ds: int = len(dataloader.dataset)
    # if (not elra_solver): model.train()
    img_type: tt.dtype = get_default_dtype() # float32
    # ldf = open("path_dom.dat", "a")
    norm100: float = 100.0 / len(dataloader)
    prev_bs: int = 0
    bs: int = 2
    loss: float = 0.0
    # global last_x; last_x = None  # debug

    for batch, (data0, target0) in enumerate(dataloader, 1): # old+ok
        # reduce cudaMemcpy(H2D) by multi-batch transfer
        data, target = data0.to(device, img_type, non_blocking=True), target0.to(device, non_blocking=True)
        if target.numel() >= prev_bs:
            if not prev_bs:  # == 0
                prev_bs = target.numel()  # new multi batches
                bs = prev_bs >> 2 # // 4

            for d, t in zip(data.split(bs), target.split(bs)):
                loss = tstep(d, t, model, cost_func, optim, limitf, scaler)
                #if batch < 12: Interpolx(model, cost_func) # debug-2025
            #loss = tstep(data_sp[], target[bs: bs2], model, cost_func, optim, limitf, scaler)
        else: # requires: drop_last=False
            left: int = target.numel() % bs
            for d, t in zip(data.split(bs), target.split(bs)):
                if t.numel() == bs:
                    tstep(d, t, model, cost_func, optim, limitf, scaler)
            if left:
                print("(SkipLastBatch, bs=%d<%d<%d, f=--)" % (left, target.numel(), prev_bs), flush=True)
            break  # optional

        #if (len(target) >= prev_bs): # early skip (drop_last) # batch_max - batch_idx)
        #    loss = tstep(data.to(device, img_type), target.to(device), model, cost_func, optim, limitf, scaler)
        #else:
        #    print("(SkipLastBatch, bs=%d<%d, f=--)" % (len(target), prev_bs), flush=True)
        #prev_bs = len(target)
        #if (99 == batch_idx % 100): Interpolx(model, cost_func, dataloader)
        #else: Interpolx(model, None, None)
        # WriteDominant(GetParam(model), log=ldf, t=batch_idx) # test (slow)
    # while True: # new Shuffler-Loop
        #mbs, data, target = utmt_DS.ShuffleBatch()

        if not (batch & 15):
            tnow : float = time_time()
            dt: float = tnow - last_time
            if (dt >= 5.0): # progress print interval
                if not tl_tmp:
                    print('Epoch progress: [%d/%d=%.0f%%]\tBatch avg. train loss:\t%.6g' %
                        (batch * target.numel(), len_ds, batch * norm100, loss) )
                else:
                    tl_tmp = (" %.6g" % loss) # += !
                    print('Epoch progress: [%.1f%%]\tBatches train losses: %s' %
                        (batch * norm100, tl_tmp) )
                last_time = tnow
                tl_tmp = ""

        #if False:
        #    steps.append(optim.state["o_calls"])
        #    losses.append(loss)
        #    batches.append(batch_idx)

        if not (loss < 1e99): break

    f_calls.append(-1)
    g_calls.append(-1)
    steps.append(-1)
    losses.append(-1)
    batches.append(-1)

    n2, nM = GetParamxMax(model, device)
    print("GetParamxMax(bs=%d) = %.3g > %.3g" % (bs, n2, nM))
    if scaler is not None:
        print("GradScaler.scaling = %.3g" % scaler.scaling)

    optim.zero_grad(set_to_none=True) # less memory during test full-batch
    # if (not elra_solver): model.eval() # issue: batch normalize + booster_on
    LimitBatchNorm(model, 0.0)  # print peak only
    return losses, batches, steps, f_calls, g_calls

def WriteParams(model, num:int, tmp: bool) -> None:
    "backup params=x (epoch wise)"
    from os.path import exists
    if exists("params_tmp"):
        assert num >= 0, "neg. epoch"
        assert model is not None, "empty model"
        if not (num % 4):
            fn:str = "params_tmp/epoch_" + str(int(num)) + ".pt"
            ParamSave(model, fn)
    else:
        if num <= 2: print("Hint:WriteParams(no-folder)=skip.")
        if tmp: ParamSave(model, fn="epoch_tmp.pt")
    return

def CreateModels(model, dev_list:list):
    "intern: create model instances for each GPU (SMP)"
    assert model is not None, "need master model"
    if len(dev_list) < 2: return [model]  # single GPU device
    assert next(model.parameters()).device == dev_list[0], "model1 @master-device"
    from copy import deepcopy

    return [deepcopy(model).to(d) if i else model for i, d in enumerate(dev_list)]

def GetOtherDevice(model, device, elra_solver:bool):
    "intern: other MultiGpuDevice (SMP/DDP)"
    # from copy import deepcopy
    device2 = None  # device_cpu

    if not cuda_device_count: return None
    # return device, deepcopy(model) # debug/test on single-gpu

    did: int = zeros(0, device=device).get_device()
    dc: int = cuda.device_count()
    if did < 0: return None  # cpu-only
    if dc < 2: return None  # single gpu

    return None # (use single gpu only) !!!!! (comment out to SMP/DDP)

    m = [cuda.mem_get_info(i) for i in range(dc)] # (free_bytes, device_bytes)
    used_mb = [(m[i][1]-m[i][0])>>20 for i in range(dc)]
    dev0mb: int = m[0][1] >> 20  # MB physical VRAM

    if elra_solver:
        if (model is None):
            device2 = tt.device("cuda", (did + 1) % dc)
        else:
            n: int = sum(p.numel() for p in model.parameters())
            if n > (1<<20):
                device2 = tt.device("cuda", (did + 1) % dc)

    print("(Multi-GPU detected) %dx%dMB, used:%s, d2=%s" % (dc, dev0mb, str(used_mb), str(device2)))
    # model2 = None if (model is None or device2 is None) else deepcopy(model).to(device2)
    return device2

def train(dataloaders: tuple,
          model, cost_func, optim, max_epochs:int = 1000, target_loss:float = 0.0, 
          batch_size:int = 0, device = device_cpu, logf = None):
    "train model with train_data"
    train_data:DataLoader = dataloaders[0]
    test_data: DataLoader = dataloaders[1]
    train_fast:DataLoader = dataloaders[2] # if (len(train_data.dataset) > 60000) else dataloaders[0]
    val_data: DataLoader = dataloaders[3]
    full_data = (train_fast, test_data, val_data)
    num_classes: int = dataloaders[-1]

    from statistic_helper import GlobalStatist as gstat
    from datetime import datetime as datetime2
    from math import log
    from time import sleep
    global booster_on, model_boost, scaler  # utmt_DS, utmt_TS
    tt.set_printoptions(precision=4, linewidth=150)
    # batch_size = batch_size if batch_size < 999999999 else len(X) # (9x9) = inf-like inf
    print("BS = %d (%d/4)" % (batch_size, DataLdrFast.EstimBatchSize(train_data)))

    # pdf: PandasDataFrame = PandasDataFrame()
    test_loss_min: float = inf
    epoch: int = 1

    global dataldr_debug
    optim.state["o_calls"], optim.state["f_calls"], optim.state["g_calls"] = 0, 0, 0

    # get_wd_params(model)
    CheckElraSolver(optim)
    if 'float16' in str(get_default_dtype()):
        scaler = MyGradScaler(init_scale=2.0**15, enabled=True) # def=2^16
    if dog_solver:
        from dog import PolynomialDecayAverager
        global dog_averager
        dog_averager = PolynomialDecayAverager(model)
    ParamLoad(model, fn="", nofilewarn=False) # "startx_000.pt"
    FastenDataloader(full_data, num_classes, maxMB = 800, device = device) # here DL switch-off
    # if elra_solver: optim.SetClasses(num_classes, batch_size)
    # if (not elra_solver): model.eval() # E.g. dropout layers will be disabled during evaluation and batchnorm layers will use the running stats instead of the batch statistics to normalize the activation. The gradient computation will not be changed or disabled. !!
    device2 = None
    if cuda_device_count > 1:
        device2 = GetOtherDevice(model, device, elra_solver)
        dev_list = (device, device2)
            # model_dp = DataParallel(model, gpu_ids = [0,1])
        mod_list = CreateModels(model, dev_list)
        print("torch.cuda.MB:", cuda.memory_allocated()>>20, cuda.memory_reserved()>>20)

    # Initial Training loss auf allen Trainings Daten, F(parameter satz)=Loss
    model.eval()  # affects dropout + batchnorm
    a0, loss, _, _ = full_batch_loss(model, None, train_fast, cost_func, device)
    if val_data is not None:
        av0, lv0, _, _ = TestEval(model, None, val_data, utmt_VS, cost_func, device)
        print('VAL-0: ds=%d, loss=%.4f, accu=%.4f%%' % (len(val_data.dataset), lv0, av0*100))
        if elra_solver: optim.SetValidLoss(lv0)
    model.train()  # default mode
    print(datetime2.now().strftime("[%H:%M:%S]"), end=" ")
    a0 *= num_classes
    if a0 > 1.5: print("### accu x classes = %.1f > 1 !!!" % a0)
    print('Start training: \t\tInit. avg. train loss:\t%.6f ln(%.1f)' % (loss, 2.718**loss), flush=True)
    loss0: float = log(num_classes)  #loss # log(classes)
    # ResetBatchNorm(model)

    # pdf.AppendFrame(loss, None, 0, "train", 0, 0, 0)

    # last_epoch_index: int = 0
    # logf = open("history.txt", "a")

    while loss > target_loss and epoch <= max_epochs and loss < abs(loss0)*3: # stop criteria
        if elra_solver: optim.SetLgsDevice(device) # restore GPU-RAM (5x Tensor)
        BatchNormMoment(model, 0.0)  # default=0.1, fix=0.0, 2**-12=2.44E-04
        dt1: float = time_time()
        if device2 is None:
            epoch_losses, epoch_batches, step, fs, gs = train_epoch(train_data, model, optim, cost_func, device, loss0)
        else: # MultiGpu
            epoch_losses, epoch_batches, step, fs, gs = train_epoch_smp(train_data, mod_list, optim, cost_func, dev_list, loss0)
        dt1 = time_time() - dt1
        mean_hist_loss: float = utmt_LossHist.get_mean_loss()
        if (dt1 > 15.0): print("(end epoch %d, f128=%.3f, dt=%.1fs)" % (epoch, mean_hist_loss, dt1), flush=True)
        BatchNormMoment(model, 1e-2)  # default=0.1

        # pdf.ExtendFrame(epoch_losses, epoch_batches, epoch, "batch", step, fs, gs) # still needed ?

        # F(nach epoch)
        if elra_solver: optim.SetLgsDevice(device_cpu) # save GPU-RAM (3x Tensor)
        dt2: float = time_time()
        if (device2 is None):
            accu, loss, accu2t, loss2t, accu1v, loss1v, accu2v, loss2v = combined_FBL(
                model, full_data, cost_func, device, optim)
        else:
            accu, loss, accu2t, loss2t,  accu1v, loss1v, accu2v, loss2v = combined_FBL_smp(
                model, full_data, cost_func, dev_list, optim)
        dt2 = time_time() - dt2
        if (dt2 > 100.0): print(datetime2.now().strftime("[%H:%M:%S]"), end=" ")
        print('Finished epoch %d/%d: \t\tFinal avg. train loss:\t%.6f (%.3f,ac=%.2f%%), dt=%.3f+%.3fs' %
            (epoch, max_epochs, loss, mean_hist_loss, accu*100, dt1, dt2), flush=True)
        WriteParams(model, epoch, dt2 > 900.0) # epoch_tmp.pt
        CopyLgsFile()

        # pdf.AppendFrame(loss, None, epoch, "train", step[-1], fs[-1], gs[-1]) # still needed ?

        if test_data is not None:
            # test_accu, test_loss, _, _ = TestEval(model, None, test_data, fdl, cost_func, device)
            # test_loss_min = min(test_loss_min, test_loss) = loss1v
            # print("Test set: Average loss: %.4f(>%.4f), accu: %.3f%%, dt=%.3f+%.3fs\n" % (test_loss, test_loss_min, test_accu*100,dt1,dt2), flush=True)
            if logf is not None:
                logf.write("%d,%.4g,%.4f,%.6g,(%.4g:%.4f:%.4g:-1),%s,%s\n" %
                (epoch, loss,accu1v,loss1v, loss2t,accu2v,loss2v, StatListStr(epoch_losses),gstat.statist_GetStr()))
                if (dt2 > 50.0): logf.flush()

            # pdf.AppendFrame(loss1v, None, epoch, "test", step[-1], fs[-1], gs[-1]) # test_loss = loss1v

        #if losses[-1] >= losses[last_epoch_index] - losses[last_epoch_index] * 0.01: break # speed up benchmarks and break early

        #if('converged' in optim.state and optim.state['converged']): break # use p2min / c2min converged flag

        if (not booster_on) and (epoch >= 5) and elra_solver:  # decide once
            n: int = sum(p.numel() for p in model.parameters())
            if (epoch == 10) or (25549352 == n):  # ImgNet1k=25mio
                booster_on = ((n>>20) < 22) or ((cuda.get_device_properties(0).total_memory >> 30) > 10)  # low-dim or high-gpu-ram
                print("Booster=", booster_on)
                if booster_on:
                    model_boost = ElraOptimizer.CopyModel(model, mode='default') # 'default','reduce-overhead',..
                    dataldr_debug = GrabBatches(train_data, device, 8)

        if cuda_device_count > 1: sleep(1.0) # relax system due to multi-gpu issue

        epoch += 1

    ParamSave(model) # for later reuse
    # ResetBatchNorm(model)
    if booster_on:
        ParamSave(model_boost, fn="final_boost.pt")

    # Release Tensor Memory
    FastenDataloader((None, None, None), 0)
    # logf.close()
    return # pdf.ReturnFrame() # losses, batches, epochs, types, steps, f_calls, g_calls

# EoF.
