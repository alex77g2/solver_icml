# Cos2MinTorchFunctionOptimizer.py (2024)
# rename: Cos2MinTorchFunctionOptimizer -> ELRA_class.py (tbd 2024)

import torch
# import lib_grad_solve
# import numpy as np
# from math import isnan, isinf, inf

compile_cnt: int = 0

def setModelParameter(theta, model: torch.nn.Module) -> None:
    "set model param(theta)"
    s = e = 0

    for p in model.parameters():
        e += torch.numel(p)
        # t, s = theta[s:e], p.size()
        # params = torch.reshape(t, p.size())
        p.data = torch.reshape(theta[s:e], p.size())
        # p.data.copy_(torch.reshape(theta[s:e], p.size()))
        s = e
    return


# was: SelfConstOptimTorch
class ElraOptimizer(torch.optim.Optimizer):

    from enum import Enum
    class Mode(Enum):
        c2min = 1
        # c2min_check = 2
        p2min = 3 # default

    def __init__(self, params, model, lr:float = 1e-5, mode:Mode = Mode.p2min, loss = None, wd:float = 0.0) -> None:

        defaults = {}
        super().__init__(params, defaults)
        if (mode != ElraOptimizer.Mode.c2min):
            import lib_grad_solve # P2M (default)
            self.__optim_step = lib_grad_solve.SelfConstOptim.p2min_step
        else:
            import lib_grad_solve_c2m as lib_grad_solve # C2M
            self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_pv_step

        self.SCO = lib_grad_solve.SelfConstOptim

        self.__optim_instance = lib_grad_solve.SelfConstOptim(lr=lr, wd=wd)

        #if (mode == ElraOptimizer.Mode.c2min):
        #    self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_pv_step
        # elif mode == ElraOptimizer.Mode.c2min_check:
        #    self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_greedy_step
        #elif (mode == ElraOptimizer.Mode.p2min):
        #    self.__optim_step = lib_grad_solve.SelfConstOptim.p2min_step
        self.step_call = getattr(self.__optim_instance, self.__optim_step.__name__)
        assert callable(self.step_call), "function needed: step()"
        assert(model is not None), "need torch model"

        self.__model = model
        self.__loss = loss
        # print(params) # generator object Module.parameters
        self.elra_TFO_dim: int = None
        self.param_bak: torch.Tensor = None # backup
        # self.device = torch.device("cpu") # default
        self.CalcParaAvg: bool = None # Booster on/off (False=None)
        self.BoosterTarget: int = 240 # 0 = off (even)
        self.ParaLossSum: float = 0.0
        self.ParaAvgCount: int = 0
        self.ParaLossCount: int = 0
        self.ParaAvgTensor: torch.Tensor = None
        self.FullBoostTensor: torch.Tensor = None # last complete
        self.FullBoostLoss: float = 0.0
        self.LastEpochSteps: int = 0
        self.EpochSteps: int = 0 # since last Booster
        self.RetraceCount: int = 0
        self.ChangedParam: bool = False
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
    def CopyModel(model: torch.nn.Module, mode:str='default', device=None) -> torch.nn.Module:
        "create 2nd model, for booster"
        from copy import deepcopy
        from os import path
        import torch._dynamo
        from torch._dynamo.testing import CompileCounter
        assert(model is not None), "needs model"
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

        assert(hasattr(torch, 'compile')), "requires PyTorch 2.x (2024)"
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

    def GetParam(self, only_new:bool, device:torch.device) -> torch.Tensor|None:
        "get param x (used for MultiGpu only)"

        if (self.param_bak is None):
            self.InitialParam()
            assert(self.param_bak is not None), "model hw/o param"

        if (only_new and not self.ChangedParam): return None
        self.ChangedParam = False

        return self.param_bak.to(device, non_blocking=True, copy=True)

    def TellTrainBoostLoss(self, train_loss:float) -> None:
        "tell train_loss of boost-params (if avail)"
        self.__optim_instance.LGS_TellTrainBoostLoss(train_loss)
        return

    def SetClasses(self, classes: int, gpubatchsize: int) -> None:
        "inform solver about class-count (helpful for noise estim.)"
        assert(self.elra_TFO_dim is None), "set before first solver.step()"
        assert(type (self.__optim_instance).__name__ == "SelfConstOptim"), "ELRA-LGS only"
        getattr(self.__optim_instance, self.SCO.LGS_SetClasses.__name__) (classes, gpubatchsize)
        return

    def CheckNextStep(self) -> tuple[bool, int]:
        "check if next step is real vs. collect-only (only for MultiGpu SMP/DDP)"
        return self.__optim_instance.LGS_CheckNextStep()

    def SetLgsDevice(self, dev: torch.device = None) -> None:
        "move LGS-Tensors to device to free GPU ram during fullbatch (end of epoch)"
        assert(type (self.__optim_instance).__name__ == "SelfConstOptim"), "ELRA-LGS only"
        if (self.elra_TFO_dim is None): return
        if self.elra_TFO_dim > (20 << 20): # RN34 = 21mio
            getattr(self.__optim_instance, self.SCO.LGS_SetDevice.__name__) (dev)
        return

    def CalcBoostTensor(self) -> None:
        "epoch x average (internal)"
        if (self.ParaAvgCount < 2): # reset: should never happen
            self.ParaAvgTensor, self.ParaAvgCount = None, 0
            self.FullBoostTensor = None
            return

        self.FullBoostLoss = 0.0 if (self.ParaLossCount < 1) else \
            (self.ParaLossSum / self.ParaLossCount)

        self.ParaAvgTensor *= 1.0 / self.ParaAvgCount # in-place
        self.ParaAvgTensor, self.FullBoostTensor = None, self.ParaAvgTensor # .to(device=torch.device('cpu'))
        self.ParaLossCount, self.ParaLossSum = 0, 0.0
        # print("BoostCalc:", self.ParaAvgCount, self.LastEpochSteps) # debug
        self.ParaAvgCount = 0
        return

    def GetParamAvg(self, enable: bool):
        "control local param averaging + reset avg. + return vector (for booster)"
        assert(type (self.__optim_instance).__name__ == "SelfConstOptim"), "ELRA only"
        loss: float = self.FullBoostLoss
        count: int = self.EpochSteps
        self.CalcParaAvg = True if enable else None
        if enable:
            assert(self.BoosterTarget >= 2), "averaged step count"

        if (count > 0): # once per epoch
            print("GetParamAvg: c=%d, l=%.6f, r=%d" % (count, loss, self.RetraceCount))
            self.EpochSteps, self.LastEpochSteps = 0, count

        if (not enable) or (count < 4): # <4 steps/epoch
            self.FullBoostTensor = None
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

        ret_vect, self.FullBoostTensor = self.FullBoostTensor, None
        return count, loss, ret_vect # .to(self.param_bak.device),  None = NoNewBoostAvail

    def InitialParam(self) -> None:
        "internal: initial export from model to this class"
        assert(self.__model is not None), "need torch model"

        if True: # (self.param_bak is None):
            params = [param.data.view(-1) for param in self.__model.parameters()]
            params_n = torch.cat(params)
            params = None
            print("ParamCount = %d, norm = %.4g, dev=%s" % (len(params_n), params_n.norm().item(), str(params_n.device))) # device=CPU
            self.elra_TFO_dim = len(params_n)
            self.param_bak = params_n
            self.ChangedParam = True
        return

    def FirstCycle(self, loss:float, scale:float) -> None:
        "internal: first step is special"

        self.InitialParam()
        grads  = [param.grad.data.view(-1) for param in self.__model.parameters()]

        # params, grads = [], []
        # for param in self.__model.parameters(): # TODO make this more torch like, see e.g. https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
        #     params.append(param.data.view(-1)) # why we get back params here (only init)
        #     grads.append(param.grad.data.view(-1))
        # self.__model.zero_grad() # (still empty first run)

        assert(scale > 0.0), "first step exploded"
        # Calls our solver with x (params), function value (loss), gradient (grads)
        params_n, _, _ = self.step_call(
            self.param_bak, loss, torch.cat(grads) * scale)

        if (params_n is not None):
            # print("ParamCount = %d, loss = %.6f, dev=%s" % (len(params_n), loss, str(params_n.device))) # device=CPU
            self.param_bak = params_n
            self.ChangedParam = True
        return

    def BoosterUpdate(self, param: torch.Tensor) -> None:
        "internal: average x-param"
        if (self.ParaAvgTensor is not None): # (self.ParaAvgCount >= 1):
            self.ParaAvgTensor += param
        else:
            self.ParaAvgTensor = param.clone() # clone needed
        self.ParaAvgCount += 1

        if (self.ParaAvgCount >= self.BoosterTarget): # > 3 and even
            self.CalcBoostTensor()
        return

    def step(self, loss: float, scale: float|None = None) -> None:
        "ELRA (C2M + P2M) step (usualy float16)"

        # get X (params) and G (grads)
        retrace: bool = False

        if (self.elra_TFO_dim is not None): # normal cycle (faster)
            # self.__optim_step.__name__ = "p2min_step"
            if not (loss < 1e999): # 1e999=inf, check outside
                print("NoGrad: loss=%.3e, x.norm=%.3g !" % (loss, self.param_bak.norm()))
                params_n, _, retrace = self.step_call( # getattr(..) (
                    self.param_bak, loss, None )
            else:
                # grads = [] # only grad (reuse param)
                grads = [param.grad.data.view(-1) for param in self.__model.parameters()]
                if (scale is not None): # (scale != 1.0):
                    params_n, _, retrace = self.step_call( # self.__optim_instance.p2min_step(
                        self.param_bak, loss, torch.cat(grads) * scale)
                else:
                    params_n, _, retrace = self.step_call(
                        self.param_bak, loss, torch.cat(grads) )
                grads = None
        else: # first cycle
            self.FirstCycle(loss, 1.0 if (scale is None) else scale)
            return
        # self.__skip_next_grad = False # optional

        self.ParaLossSum += loss
        self.ParaLossCount += 1

        if (params_n is not None): # no update step (collect average)
            setModelParameter(params_n, self.__model) # set __model.parameters
            self.ChangedParam = True
            self.EpochSteps += 1 # moving steps

            if (retrace):
                self.RetraceCount += 1
                self.param_bak = params_n
                # self.param_bak.copy_(params_n)
                return

            # (not retrace)
            if self.CalcParaAvg is not None: # and (self.param_bak is not None): # Booster
                self.BoosterUpdate(self.param_bak)

            self.param_bak = params_n
            # self.param_bak.copy_(params_n)

        return # self.ParaAvgCount # BoosterFill

    def step_retrace(self, loss: float) -> None:
        "ELRA (C2min + P2min) retrace step (no gradient)"
        # return self.step(loss, None)

        assert (self.elra_TFO_dim is not None), "normal cycle (not first)"
        
        print("NoGrad: loss=%.3e, x.norm=%.3g !" % (loss, self.param_bak.norm()))
        params_n, _, retrace = self.step_call( self.param_bak, loss, None )

        assert (params_n is not None), "retrace is not coolect-only"
        setModelParameter(params_n, self.__model) # set __model.parameters
        self.EpochSteps += 1 # moving steps

        assert(retrace), "retrace = retrace"
        self.RetraceCount += 1
        self.param_bak = params_n
        self.ChangedParam = True
        # self.param_bak.copy_(params_n)
        return

    def step_noscale(self, loss: float, grad: torch.Tensor=None) -> None:
        "ELRA (C2min + P2min) normal step (no GradScaler, scale=1, mainly float32)"
        # return self.step(loss, 1.0)
        
        retrace: bool = False

        if (self.elra_TFO_dim is not None): # normal cycle (faster)
            # self.__optim_step.__name__ = "p2min_step"
            if not (loss < 1e999): # 1e999=inf, (self.__skip_next_grad)
                return self.step_retrace(loss)
            else:
                if (grad is None):
                    # grads = [] # only grad (reuse param)
                    # for param in self.__model.parameters():
                    #     grads.append(param.grad.data.view(-1))
                    grads = [param.grad.data.view(-1) for param in self.__model.parameters()]
                    params_n, _, retrace = self.step_call( # self.__optim_instance.p2min_step(
                            self.param_bak, loss, torch.cat(grads))
                    grads = None
                else: # MultiGpu SMP only
                    params_n, _, retrace = self.step_call(self.param_bak, loss, grad )
        else: # first cycle
            self.FirstCycle(loss, 1.0)
            return
        # self.__skip_next_grad = False # optional

        self.ParaLossSum += loss
        self.ParaLossCount += 1

        if (params_n is not None): # no update step (collect average)
            setModelParameter(params_n, self.__model) # set __model.parameters
            self.ChangedParam = True
            self.EpochSteps += 1 # moving steps

            if (retrace):
                self.RetraceCount += 1
                self.param_bak = params_n
                # self.param_bak.copy_(params_n)
                return

            # (not retrace)
            if self.CalcParaAvg is not None: # and (self.param_bak is not None): # Booster
                self.BoosterUpdate(self.param_bak)

            self.param_bak = params_n
            
        return

# class ElraOptimizer
