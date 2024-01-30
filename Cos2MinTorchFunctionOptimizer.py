# Cos2MinTorchFunctionOptimizer.py (2023)

import torch
import lib_grad_solve
# import numpy as np


def setModelParameter(theta, model: torch.nn.Module, dev) -> None:
    # (faster + less code)
    s : int = 0
    e : int = 0

    # use_cuda : bool = torch.cuda.is_available()
    if (dev is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CPU for FnktBench
    else:
        device = dev
    # theta = torch.tensor(theta)

    for p in model.parameters():
        e += torch.numel(p)
        # t = theta[s:e]
        # s = p.size()
        # params = torch.reshape(t, p.size())
        p.data = torch.reshape(theta[s:e], p.size()).to(device)
        # ToDo: Konsistenz mit allen Benchmark-Files
        s = e

def dummy_predict(theta , model:torch.nn.Module, loss = None, batch = None, y = None) -> float:
    # ("UNUSED:dummy_predict")
    setModelParameter(theta, model)
    assert(0)
    with torch.no_grad():

        f_x = model() if batch is None else model(batch)
        L = f_x if loss is None else loss(f_x, y)

    return L.item() # np.array(L.item())


def dummy_grad(theta , model:torch.nn.Module, loss = None, batch = None, y = None):
    # ("UNUSED:dummy_grad")
    assert(0)
    setModelParameter(theta, model)

    f_x = model() if (batch is None) else model(batch)
    L = f_x if (loss is None) else loss(f_x, y)
    # print("L=", L.item())
    L.backward() # our gradient (+ somewhere else)

    grads = [] #torch.Tensor()#[] # this block costs 30% runtime !!!
    for param in model.parameters():
        grads.append(param.grad.data.view(-1)) #org
    # print("1:",type(grads),len(grads[1]))
    grads = torch.cat(grads)
    # print("2:",type(grads),grads.shape)
    model.zero_grad()

    return grads, L.item() # np.array(grads.cpu())

class SelfConstOptimTorch(torch.optim.Optimizer):

    from enum import Enum
    class Mode(Enum):
        c2min = 1
        # c2min_check = 2
        p2min = 3

    def __init__(self, params, model, lr:float = 1e-6, mode:Mode = Mode.c2min, loss = None) -> None:

        defaults = {}
        super().__init__(params, defaults)

        self.__optim_instance = lib_grad_solve.SelfConstOptim(lr=lr)
        self.__skip_next_grad: bool = False
        # self.__GG: float = -2.1
        # self.__calcGG: bool = False
        self.__next_combi_x: bool = False

        if (mode == SelfConstOptimTorch.Mode.c2min):
            self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_pv_step
        #elif mode == SelfConstOptimTorch.Mode.c2min_check:
        #    self.__optim_step = lib_grad_solve.SelfConstOptim.c2min_greedy_step
        elif (mode == SelfConstOptimTorch.Mode.p2min):
            self.__optim_step = lib_grad_solve.SelfConstOptim.p2min_step

        self.__model = model
        self.__loss = loss
        # print(params) # generator object Module.parameters
        self.elra_TFO_dim: int = 0
        self.param_bak: torch.Tensor = None # backup
        # self.device = torch.device("cpu") # default
        self.CalcParaAvg: bool = False # Booster on/off
        self.ParaLossSum: float = 0.0
        self.ParaAvgCount: int = 0
        self.ParaLossCount: int = 0
        self.ParaAvgTensor: torch.Tensor = None
        return

    def revert_step(self, loss: float) -> bool:
        "P2min: revert step check" # future
        if (self.elra_TFO_dim < 1):
            return False # do initial step anyway
            
        ret = getattr(self.__optim_instance, self.__optim_step.__name__) (
                    None, loss, None )
        self.__skip_next_grad = ret
        return ret # (default=False)
        
    def SetCombiX(self, xavg: torch.Tensor, loss: float) -> None:
        "set combined x (epoch averaged param)"
        assert(0) # unused
        assert(len(xavg) > 0)
        self.param_bak = xavg.clone()
        self.__next_combi_x = True
        self.__skip_next_grad = False
        #_, _, _ = getattr(self.__optim_instance, self.__optim_step.__name__) (
        #            xavg, loss, None, combi_step=True )
        return
        
    def GetParamAvg(self, enable:bool):
        "control local param averaging + reset avg. + return vector (for booster)"
        loss: float = self.ParaLossSum
        count: int = self.ParaAvgCount
        self.CalcParaAvg = enable
        ret_vect, self.ParaAvgTensor = self.ParaAvgTensor, None
        self.ParaAvgCount, self.ParaLossSum = 0, 0.0
        if (self.ParaLossCount >= 1):
            loss /= self.ParaLossCount
        self.ParaLossCount = 0
        if (not enable) or (count < 2):
            return 0, loss, None
        ret_vect *= (1.0 / count)
        return count, loss, ret_vect

    def step(self, loss : float, batch = None, y = None) -> None:
        "C2min + P2min step" # before setModelParameter() # TODO: cleanup

        # get X (params) and G (grads)
        params, grads = [], [] # list of torch.tensor
        
        if (self.elra_TFO_dim > 0): # normal cycle (faster)
            if (self.__skip_next_grad):
                params_n, _, _ = getattr(self.__optim_instance, self.__optim_step.__name__) (
                    self.param_bak, loss, None )
            else:
                for param in self.__model.parameters():
                    grads.append(param.grad.data.view(-1)) # skip params (reuse them)
                params_n, _, _ = getattr(self.__optim_instance, self.__optim_step.__name__) (
                    self.param_bak, loss, torch.cat(grads)) # , combi_step=self.__next_combi_x
        else: # first cycle
            for param in self.__model.parameters(): # TODO make this more torch like, see e.g. https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
                params.append(param.data.view(-1)) # why we get back params here (only init) ??  C2M+P2M do not need this !?
                grads.append(param.grad.data.view(-1))
            
            # params, grads = torch.cat(params), torch.cat(grads)
            # self.__model.zero_grad() # (still empty first run)

            # Calls our solver with x (params), function value (loss), gradient (grads)
            params_n, _, _ = getattr(self.__optim_instance, self.__optim_step.__name__)(
                torch.cat(params), loss, torch.cat(grads) )
                #lambda theta: dummy_predict(theta, self.__model, y=y, batch=batch, loss=self.__loss),
                #lambda theta: dummy_grad(theta, self.__model, y=y, batch=batch, loss = self.__loss))
            
        self.__skip_next_grad = False # optional
        self.__next_combi_x = False # unused
        del params, grads
        
        if (params_n is not None): # no update step (collect average)
            self.param_bak = params_n
            if (self.elra_TFO_dim < 1):
                print("ParamCount = %d, loss = %.6f, dev=%s" % (len(params_n), loss, str(params_n.device))) # device=CPU
            self.elra_TFO_dim = len(params_n)
            setModelParameter(params_n, self.__model, params_n.device) # set __model.parameters
            
            if (self.CalcParaAvg): # move booster (from LGS to here = faster for ResNet50)
                if (self.ParaAvgCount >= 1):
                    self.ParaAvgTensor += params_n
                else:
                    self.ParaAvgTensor = params_n # no clone()
                self.ParaAvgCount += 1
                
        self.ParaLossSum += loss
        self.ParaLossCount += 1
            

        if False: # TODO: is this really needed ?
            self.state['converged'] = self.__optim_instance.converged
            self.state['o_calls'] = self.__optim_instance.solversteps
            self.state['f_calls'] = self.__optim_instance.fcalls
            self.state['g_calls'] = self.__optim_instance.jaccalls
            
        return

# self.
