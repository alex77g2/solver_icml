# Cos2MinTorchFunctionOptimizer.py (2025)
# rename: Cos2MinTorchFunctionOptimizer -> ELRA_class.py (tbd 2024)

import torch
from torch import cat as torch_cat
from torch.nn import Module
# import lib_grad_solve
# from math import isnan, isinf, inf
device_cpu: torch.device = torch.device('cpu')
compile_cnt: int = 0

def setModelParameter(theta: torch.Tensor, SetModelCfg:tuple) -> None:
    "set model param(theta) - new+faster"
    model, split, sizes = SetModelCfg
    # sp: tuple = torch.split(theta, self.SplitList)
    for p, t, s in zip(model.parameters(), theta.split(split), sizes):
        p.data = t.reshape(s)
        # p.data *= 0.0; p.data += t.reshape(s)  # in-place
        # p.data.flatten()[:] = t
    return

#def setModelParameter1(theta, model: Module, split:tuple) -> None:
#    "set model param(theta) - test"
#    # sp: tuple = torch.split(theta, self.SplitList)
#    for p, t in zip(model.parameters(), theta.split(split)):
#        p.data = t.reshape(p.size())
#    return
#
#def setModelParameter0(theta, model: Module, sp) -> None:
#    "set model param(theta)"
#    s = e = 0
#
#    for p in model.parameters():
#        e += p.numel()
#        # t, s = theta[s:e], p.size()
#        # params = torch.reshape(t, p.size())
#        p.data = theta[s:e].reshape(p.size())
#        # p.data.copy_(torch.reshape(theta[s:e], p.size()))
#        s = e
#    return

#def ExportBatchNorm(model: Module) -> list[tuple]:
#    "Export BatchNorm to Tensor-List (for retrace)"
#    from torch.nn import BatchNorm2d
#    return [(m.running_mean.clone(), m.running_var.clone()) \
#        if isinstance(m, BatchNorm2d) else None for m in model.modules()]

def ImportBatchNorm(model: Module, bnl) -> None:
    "Import BatchNorm from Tensor-List (for retrace)"
    if bnl is None: return
    for m, b in zip(model.modules(), bnl):
        if b is not None:
            m.running_mean, m.running_var = b
    return

# was: SelfConstOptimTorch
class ElraOptimizer(torch.optim.Optimizer):

    from enum import Enum
    class Mode(Enum):
        c2min = 1
        # c2min_check = 2
        p2min = 3 # default

    def __init__(self, params, model, batch_size: int, classes: int, lr:float = 1e-5, mode:Mode = Mode.p2min, loss = None, wd:float = 1.0) -> None:

        defaults = {}
        super().__init__(params, defaults)
        if (mode != ElraOptimizer.Mode.c2min):
            import lib_grad_solve  # P2M (default)
            self.__optim_step = lib_grad_solve.SelfConstOptim.p2min_step
        else:
            import lib_grad_solve_c2m as lib_grad_solve  # C2M
            self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_pv_step

        assert model is not None, "need torch model"
        assert batch_size >= 1, "positive number (seen by GPU)"
        assert classes > 1, "labels"
        dim: int = sum(p.numel() for p in model.parameters())  # model.parameters()
        assert dim > 0, "model dim count"

        self.SCO = lib_grad_solve.SelfConstOptim
        if wd >= 0.0 and wd < 0.5:
            print('# change weight_decay=%.6f ==> %.6f !' % (wd, 1.0 - wd))
            wd = 1.0 - wd  # should be 0.999 .. 1.000

        self.__optim_instance = lib_grad_solve.SelfConstOptim(dim, batch_size, classes, lr=lr, wd=wd)

        #if (mode == ElraOptimizer.Mode.c2min):
        #    self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_pv_step
        # elif mode == ElraOptimizer.Mode.c2min_check:
        #    self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_greedy_step
        #elif (mode == ElraOptimizer.Mode.p2min):
        #    self.__optim_step = lib_grad_solve.SelfConstOptim.p2min_step
        self.step_call = getattr(self.__optim_instance, self.__optim_step.__name__)
        assert callable(self.step_call), "function needed: step()"
        # print(model.state_dict()); exit()

        self.__model = model
        # self.__loss: float|None = loss
        self.batch_size: int = batch_size  # 8, 32, .. TODO: transfer into LGS
        # self.classes: int = classes
        # print(params) # generator object Module.parameters
        self.elra_TFO_dim: int = None
        self.param_bak: torch.Tensor = None  # backup (cache)
        # self.device = torch.device("cpu") # default
        self.CalcParaAvg: bool = None # Booster on/off (False=None)
        self.BoosterTarget: int = 240 # 0 = off (even)
        self.ParaLossSum: float = 0.0
        self.ParaAvgCount: int = 0
        self.ParaLossCount: int = 0
        self.ParaAvgTensor: torch.Tensor = None
        self.FullBoostTensor: torch.Tensor = None # last complete
        self.LastBoostTensor = None  # debug: print progress(x)
        self.FullBoostLoss: float = 0.0
        self.BoostCount: int = 0
        self.LastEpochSteps: int = 0
        self.EpochSteps: int = 0  # since last Booster
        self.RetraceCount: int = 0
        # self.ChangedParam: torch.Tensor = None
        self.ChangedParam: bool = False
        self.SetModelCfg = None  # (model, SplitList, ParSizes)
        # self.SplitList: tuple = None  # (int, int, ..)
        # self.ParSizes = None
        self.BatchNormBak = None

        self.InitialParam()
        # self.SetClasses(classes, batch_size)
        return

    #def revert_step(self, loss: float) -> bool:
    #    "UNUSED: P2min: revert step check (future)"
    #    if (self.elra_TFO_dim is None):
    #        return False # do initial step anyway
    #
    #    if not (loss < 1e999): # isnan(loss) or isinf(loss), 1e999==inf (fastest)
    #        self.__skip_next_grad = True
    #        return True # never allow NaN
    #
    #    ret = getattr(self.__optim_instance, self.__optim_step.__name__) (
    #                None, loss, None )
    #    self.__skip_next_grad = ret
    #    return ret # (default=False)

    @staticmethod
    def CopyModel(model: Module, mode:str='default', device=None) -> Module:
        "create 2nd model, for booster"
        from copy import deepcopy
        from os import path
        import torch._dynamo
        from torch._dynamo.testing import CompileCounter
        assert model is not None, "needs model"
        fn: str = "compile_skip"

        model_boost = deepcopy(model) # ELRA-Booster
        device1 = next(model.parameters()).device
        if (device is not None):
            print("model.device =", str(device1), ", boost =", str(device))
            model_boost.to(device)

        if (mode == 'off'):
            print("deepcopy(model), no compile(), mode =", mode)
            return model_boost

        cc_fc: int = CompileCounter().frame_count
        #if (cc_fc < 1): # 0 even for compiled models ?
        #    print("deepcopy(model), no compile(),", mode, cc_fc)
        #    return model_boost

        if path.isfile(fn):
            # https://github.com/pytorch/pytorch/issues/128121
            # 2nd compile of deepcopy(model) fails on multiple ubuntu-pc (fatal error: Python.h: file not found)
            print("Detected Compile Blocker (for issue #128121) !", fn)
            return model_boost

        assert hasattr(torch, 'compile'), "requires PyTorch 2.x (2024)"
        if False: # path.exists("/etc/"): # Windows not supported (May 2024)
            global compile_cnt
            n: int = sum(p.numel() for p in model.parameters())
            if not len(mode): mode = 'default'
            print("compiling 2nd model... (takes a ~minute)", n)
            torch._dynamo.config.suppress_errors = True
            compile_crash: bool = False
            torch._dynamo.reset()
            # backend='cudagraphs' ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
            compile_cnt += 1
            try:
                model_boost = torch.compile(model_boost, mode=mode) # mode='reduce-overhead') # reduce-overhead fails for ImgNet + lama2
                print("CompileCounter() =", compile_cnt, CompileCounter().frame_count)
            except Exception as inst:
                compile_crash = True
                print("Exception in torch.compile() !!", inst)
                model_boost = deepcopy(model) # again, w/o compile
            if compile_crash:
                print("Create Compile Blocker!", fn)
                f = open(fn, "x")
                f.close()

        return model_boost

    # def GetLossLimit(self) -> float:
    #    "unused"
    #    return self.__optim_instance.GetLossLimit()

    def SetParam(self, x: torch.Tensor) -> None:
        "debug: set x (jump)"
        assert x.numel() == self.elra_TFO_dim, "Tensor size conflict"
        assert self.param_bak.device == x.device, "Tensor device conflict"
        y = x.clone()
        self.param_bak = y
        setModelParameter(y, self.SetModelCfg)
        self.__optim_instance.LGS_SoftReset()
        return

    def GetParam(self, only_new:bool, device:torch.device) -> torch.Tensor|None:
        "get param x (used for MultiGpu only)"
        param_bak = self.param_bak

        # if param_bak is None:
            # self.InitialParam()  # only SMP
        assert self.elra_TFO_dim is not None, "model w/o param"

        if only_new and not self.ChangedParam: return None
        self.ChangedParam = False

        if device == device_cpu:  # .cpu()
            return param_bak.to(device_cpu, copy=True).pin_memory()
        return param_bak.to(device, non_blocking=True, copy=True) 

    def SetValidLoss(self, loss:float, boost:float = None) -> None:
        "set valid_loss (optional)"
        self.__optim_instance.SetValidLoss(loss, boost)
        return

    def SetTrainLoss(self, loss:float, boost:float = None) -> None:
        "set valid_loss (optional)"
        self.__optim_instance.SetTrainLoss(loss, boost)
        return

    def TellTrainBoostLoss(self, train_loss:float) -> None:
        "tell train_loss of boost-params (if avail)"
        self.__optim_instance.LGS_TellTrainBoostLoss(train_loss)
        return

    def SetClasses(self, classes: int, gpubatchsize: int) -> None:
        "inform solver about class-count (helpful for noise estim.)"
        assert type(self.__optim_instance).__name__ == "SelfConstOptim", "ELRA-LGS only"
        print("(SetClasses already in init)")  # will be removed soon
        # assert self.elra_TFO_dim is None, "set before first solver.step()"
        # getattr(self.__optim_instance, self.SCO.LGS_SetClasses.__name__) (classes, gpubatchsize)
        return

    def CheckNextStep(self) -> tuple[bool, int]:
        "check if next step is real vs. collect-only (only for MultiGpu SMP/DDP)"
        return self.__optim_instance.LGS_CheckNextStep()

    def SetLgsDevice(self, dev: torch.device = None) -> None:
        "move LGS-Tensors to device to free GPU ram during fullbatch (end of epoch)"
        assert type(self.__optim_instance).__name__ == "SelfConstOptim", "ELRA-LGS only"
        if self.elra_TFO_dim is None: return
        if self.elra_TFO_dim > (20 << 20):  # RN34 = 21mio
            getattr(self.__optim_instance, self.SCO.LGS_SetDevice.__name__) (dev)
        return

    def CalcBoostTensor(self) -> None:
        "epoch x average (internal)"
        if (self.ParaAvgCount < 2):  # reset: should never happen
            print("CalcBoostTensor:reset", self.ParaAvgCount)
            self.ParaAvgTensor, self.ParaAvgCount = None, 0
            self.FullBoostTensor = None
            return

        self.FullBoostLoss = 0.0 if (self.ParaLossCount < 1) else \
            (self.ParaLossSum / self.ParaLossCount)

        self.ParaAvgTensor *= 1.0 / self.ParaAvgCount  # in-place average
        if True:  # debug: print progress(x)
            pat, lbt = self.ParaAvgTensor, self.LastBoostTensor
            self.BoostCount += 1
            if self.BoostCount < 10:
                fn:str = str('boost_%02d.pt' % self.BoostCount)
                # pat.save(fn) # debug
            if lbt is not None:
                d = pat.dist(lbt).item()
                e = pat.dist(self.param_bak).item()  # x(now) <> boost(now)
                print("BoostDistX = %.3E / %d, dxb=%.3E" % (d, self.ParaAvgCount, e))
            self.LastBoostTensor = self.FullBoostTensor

        self.ParaAvgTensor, self.FullBoostTensor = None, self.ParaAvgTensor # .to(device=torch.device('cpu'))
        self.ParaLossCount, self.ParaLossSum = 0, 0.0
        # print("BoostCalc:", self.ParaAvgCount, self.LastEpochSteps) # debug
        self.ParaAvgCount = 0
        return

    def GetBoostFuture(self, t: float) -> torch.Tensor:
        "debug: try boosted prediction (future), 0 < t < 2?"
        if not (t > 0.0):
            # print("+++ GetBoostFuture1 +++", type(self.FullBoostTensor), t)
            assert 0.0 == t, "positive prediction point (future)"
            return self.FullBoostTensor
        # print("+++ GetBoostFuture2 +++", type(self.LastBoostTensor), t)
        if self.LastBoostTensor is None: return None
        # pred = now + t*(now - last)
        return (self.FullBoostTensor * (1.0 + t)) - (self.LastBoostTensor * t)

    def GetParamAvg(self, enable: bool):
        "control local param averaging + reset avg. + return vector (for booster)"
        assert type (self.__optim_instance).__name__ == "SelfConstOptim", "ELRA only"
        loss: float = self.FullBoostLoss
        count: int = self.EpochSteps
        self.CalcParaAvg = True if enable else None
        if enable:
            assert(self.BoosterTarget >= 2), "averaged step count"

        if (count > 0): # once per epoch
            print("GetParamAvg: c=%d, l=%.6f, r=%d" % (count, loss, self.RetraceCount))
            self.EpochSteps, self.LastEpochSteps = 0, count

        if (not enable) or (count < 4): # <4 steps/epoch
            # self.FullBoostTensor = None  # avoid double boost usage
            return 0, loss, None

        if (self.FullBoostTensor is None) and (self.ParaAvgCount > 0): # not enough real steps
            if (self.BoosterTarget < 999999999):
                print("GetParamAvg:BoostEpochSwitch(%d<%d,rm=%d)" %
                    (count, self.BoosterTarget, self.ParaAvgCount))
                self.BoosterTarget = 999999999 # never within epoch (internal constant, e.g. INT_MAX)
                # self.ParaAvgTensor, self.ParaAvgCount = None, 0 # forget incomplete sum
                # return count, loss, None # this epoch w/o boost
            if (self.ParaAvgCount >= 2): # minimum guess (only for very few steps/epoch)
                self.CalcBoostTensor() # create FullBoostTensor if possible
            else:
                print("Warn:SkipLowBoost(pac=%d, epoS=%d)" % (self.ParaAvgCount, count))

        return count, loss, self.FullBoostTensor  # no reset
        # ret_vect, self.FullBoostTensor = self.FullBoostTensor, None
        # return count, loss, ret_vect  #.to(self.param_bak.device),  None = NoNewBoostAvail

    def InitialParam(self) -> None:
        "internal: initial export from model to this class"
        model = self.__model
        assert model is not None, "need torch model"

        ParSizes = tuple( [par.shape for par in model.parameters()] )
        params = [par.data.flatten() for par in model.parameters()]
        SplitList = tuple( [x.numel() for x in params] )
        # print([len(x) for x in params], [x.numel() for x in params]); exit()
        self.SetModelCfg = (model, SplitList, ParSizes)
        params_n, params = torch_cat(params), None
        self.elra_TFO_dim = params_n.numel()
        print("ParamCount = %d, norm = %.4g, lay=%d/2, dev=%s" % (self.elra_TFO_dim, params_n.norm().item(), len(SplitList), str(params_n.device))) # device=CPU
        self.param_bak = params_n
        self.ChangedParam = True
        return

    def FirstCycle(self, loss:float, scale:float, grad = None) -> None:
        "internal: first step is special - unused"

        # self.InitialParam()

        # for param in self.__model.parameters(): # TODO make this more torch like, see e.g. https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
        #     params.append(param.data.view(-1)) # why we get back params here (only init)
        #     grads.append(param.grad.data.view(-1))
        # self.__model.zero_grad() # (still empty first run)

        # Calls our solver with x (params), function value (loss), gradient (grads)
        if grad is None:
            assert scale > 0.0, "first step exploded"
            grads = [par.grad.data.flatten() for par in self.__model.parameters()]
            grad = torch_cat(grads) * scale
            grads = None
        params_n, _, _ = self.step_call(self.param_bak, loss, grad)
        print("FirstCycle", loss, type(params_n))
        if (params_n is not None):
            # assert 0, "FirstCycle-x"
            # print("ParamCount = %d, loss = %.6f, dev=%s" % (len(params_n), loss, str(params_n.device))) # device=CPU
            # setModelParameter(params_n, self.SetModelCfg) # missing
            self.param_bak = params_n
            self.ChangedParam = True
        return

    def BoosterUpdate(self, param: torch.Tensor) -> None:
        "internal: average x-param"
        if (self.ParaAvgTensor is not None): # (self.ParaAvgCount >= 1):
            self.ParaAvgTensor += param
        else:
            self.ParaAvgTensor = param.clone()  # clone needed
        self.ParaAvgCount += 1

        if (self.ParaAvgCount >= self.BoosterTarget):  # > 3 and even
            self.CalcBoostTensor()
        return

    def zero_grad(self, set_to_none:bool=True) -> None:
        "model.zero_grad() = clear gradients (only for compatible api)"
        self.__model.zero_grad(set_to_none)
        return

    def step(self, loss: float, scale: float|None = None) -> None:
        "ELRA (C2M + P2M) step (usualy float16)"

        # get X (params) and G (grads)
        # retrace: bool = False
        param_bak = self.param_bak

        # self.__optim_step.__name__ = "p2min_step"
        if not (loss < 1e999): # 1e999=inf, check outside
            print("NoGrad: loss=%.3e, x.norm=%.3g !" % (loss, param_bak.norm()))
            params_n, _, retrace = self.step_call(  # getattr(..) (
                param_bak, loss, None )
        else:
            grads = [par.grad.data.flatten() for par in self.__model.parameters()]
            if (scale is not None): # (scale != 1.0):
                params_n, _, retrace = self.step_call( # self.__optim_instance.p2min_step(
                    param_bak, loss, torch_cat(grads) * scale)
            else:
                params_n, _, retrace = self.step_call(
                    param_bak, loss, torch_cat(grads) )
            grads = None
        #else:  # first cycle
        #    self.FirstCycle(loss, 1.0 if (scale is None) else scale)
        #    return

        self.ParaLossSum += loss
        self.ParaLossCount += 1

        if params_n is None: return

        # no update step (collect average)
        setModelParameter(params_n, self.SetModelCfg)
        self.ChangedParam = True
        self.EpochSteps += 1  # moving steps

        if retrace:
            self.RetraceCount += 1
            self.param_bak = params_n
            # self.param_bak.copy_(params_n)
            ImportBatchNorm(self.__model, self.BatchNormBak)
            return

        # self.BatchNormBak = ExportBatchNorm(self.__model)  # new 25.02.2025

        # (not retrace)
        if self.CalcParaAvg is not None: # and (self.param_bak is not None): # Booster
            self.BoosterUpdate(param_bak)

        self.param_bak = params_n
        # self.param_bak.copy_(params_n)

        return  # self.ParaAvgCount # BoosterFill

    def step_retrace(self, loss: float) -> None:
        "ELRA (C2min + P2min) retrace step (no gradient)"
        # return self.step(loss, None)

        param_bak = self.param_bak
        assert param_bak is not None, "normal cycle (not first)"
        
        print("NoGrad: loss=%.3e, x.norm=%.3g !" % (loss, param_bak.norm()))
        params_n, _, retrace = self.step_call( param_bak, loss, None )

        assert params_n is not None, "retrace is not coolect-only"
        setModelParameter(params_n, self.SetModelCfg)
        self.EpochSteps += 1  # moving steps

        assert retrace, "retrace = retrace"
        self.RetraceCount += 1
        self.param_bak = params_n
        self.ChangedParam = True
        # self.param_bak.copy_(params_n)
        ImportBatchNorm(self.__model, self.BatchNormBak)
        return

    def step_noscale(self, loss: float, grad: torch.Tensor=None) -> None:
        "ELRA (C2min + P2min) normal step (no GradScaler, scale=1, mainly float32)"
        # return self.step(loss, 1.0)
        
        retrace: bool = False
        param_bak = self.param_bak

        # if True:  # normal cycle (faster)
        if not (loss < 1e999): # 1e999=inf, (self.__skip_next_grad)
            return self.step_retrace(loss)

        if grad is None:
            # for param in self.__model.parameters():
            #     grads.append(param.grad.data.view(-1))
            grads = [par.grad.data.flatten() for par in self.__model.parameters()]
            params_n, _, retrace = self.step_call( # self.__optim_instance.p2min_step(
                    param_bak, loss, torch_cat(grads))
            grads = None
        else:  # MultiGpu SMP only
            params_n, _, retrace = self.step_call(param_bak, loss, grad)

        self.ParaLossSum += loss
        self.ParaLossCount += 1

        if params_n is None: return

        # update step (no collect average)
        setModelParameter(params_n, self.SetModelCfg)
        self.ChangedParam = True
        self.EpochSteps += 1  # moving steps
        self.param_bak = params_n

        if retrace:
            self.RetraceCount += 1
            # self.param_bak.copy_(params_n)
            ImportBatchNorm(self.__model, self.BatchNormBak)
            return

        # self.BatchNormBak = ExportBatchNorm(self.__model)  # new 25.02.2025

        # (not retrace)
        if self.CalcParaAvg is not None: # and (param_bak is not None): # Booster
            self.BoosterUpdate(param_bak)

        return

# class ElraOptimizer
