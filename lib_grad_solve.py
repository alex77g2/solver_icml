# key subroutines for the application of the gradient descent algo
# for ICML 2024

# from numba import jit, int32, float32
# import cython
# from _optim.lib import c_alpha_fac, c_InitOptimP2M, c_InitOptimC2M, c_UpdateBeta, c_growth_func, c_UpdateErrorDamper, c_CalcErrorTolerance
# import numpy as np
import torch as tt
from math import sqrt, log2, isnan, isinf, log, inf
from statistic_helper import GlobalStatist as gstat


class SelfConstOptim:

    @staticmethod
    def print(*values: object) -> None:
        "(debug)"
        if False:
            print(values)

    def __init__(self, lr: float = 1e-5, cls:int = 0, wd:float = 0.0):
        self.decay_perm: bool = False # 1= dauerdecay, 0= wechseldecay
        self.debug_out: bool = True
        self.__init: bool = True
        self.trackHistory: bool = False
        self.__best_fkt = 99.0e99  # best f(x) this far
        self.__initfkt = 0.0  # initial function value (only p2m)
        self.alpha = lr if (lr > 0.0) else 0.00001 # LR = initial alpha (default=1e-5)
        self.alphaRed = 1.0 # reduces max alpha, if to many retraces
        #self.beginn = 1.0 #3.0 #unused
        self.lastFmean: float = 0.0
        self.lastDev: float = 0.0
        self.Increase: bool = False
        self.upperBs: int = 3 #variable for cascading bs
        self.lowerBs: int = 2 #variable for cascading bs
        self.statlength: int = 768 #statlength/self.collects*100=number of steps taken for average f
        self.lastRed: int = 0 #cycles without better results
        #self.Gold: float = 0.0
        self.sumFvalue: float = 0.0
        self.sumSquareFvalue: float = 0.0
        self.MinOfF: float = 0.0
        self.MinOfF2: float = 0.0
        # self.fixed_beta:bool = True
        self.__xoldnorm: float = 0.0
        self.__noise_avg: float = 0.0
        self.__alpha_noise: float = 0.0
        # self.__beta: float = 0.75 # only C2M
        # self.__mom: tt.Tensor = None # only C2M
        # self.__fkt_avg: float = 0.0  # only C2M
        # self.__yg: tt.Tensor = None # only C2M
        self.__weight_decay: float = 1.0 if (wd <= 0.0) else wd # 1.0=off, 0.9999 .. 0.9998
        self.__weight_decay_backup: float = 1.0 if (wd <= 0.0) else wd
        self.__weight_decay_on = True if (self.__weight_decay < 1.0) else None
        if (not self.__weight_decay_on): self.__weight_decay_backup = None
        # self.__ndim: int = 0 # dimension
        # self.__path = 0.0
        self.coll_inc: int = 0  # new (ResNet50)
        self.coll_cnt: int = 0  # new (ResNet50)
        self.step_cnt: int = 0  # new (ResNet50)
        self.step_cnt2: int = 0  # new (ResNet50)
        self.adjustinit: bool = False  # True only for low-dim !
        self.__last_fkt: float = 0.0
        self.__clipping: float = 10.0 #1.0e10 # or inf for float 32
        # self.__initadjust:int = 0
        if self.trackHistory:
            self.xhist, self.ahist, self.fhist = [], [], []
        # self.FktLimit: float = 1e99 # future: skip-gradient-calc
        # DynamicBatchSize (Nov.2023)
        self.collects: int|None = None  # target=const, <2 off
        self.coll_fxl: list[float] = []  # list of f(x) for combined steps
        # self.coll_ggl: list[float] = []  # test
        self.coll_ggc: float = 0.0
        #self.coll_cos: tt.Tensor = tt.ones(100, dtype=tt.float32, device='cpu')  # [1.0] * 100 # test (RC1)
        #self.coll_contrast: tt.Tensor = tt.ones(100, dtype=tt.float32, device='cpu')
        # self.coll_LastGrd: tt.Tensor = None # test
        self.coll_GradFails: int = 0
        # self.coll_AllSteps: int = 0
        self.coll_grad: tt.Tensor = None
        #self.coll_grd: tt.Tensor = None  # skipping model.zero_grad() could avoid extra Tensor here
        #self.coll_grd2: tt.Tensor = None  # debug/test
        self.RandScaler: tt.Tensor = None # weight_decay + f16
        self.ScalerRuns: int = 0
        self.fcalls: int = 0
        self.jaccalls: int = 0
        self.solversteps: int = 0
        self.signal = [1.0e-8, 10000, False, False]
        self.converged: bool = False
        self.__yg_old: tt.Tensor = tt.tensor([])
        self.__norm_yg_old: float = 0.0
        self.__retrace: bool = False  # (CFFI)
        self.__retrace2: int = 0
        self.x_lastgood: tt.Tensor = None
        # self.__maxgrowth = 25.0 # unused
        # self.__growth = 1
        self.__classes: int = cls # default=0
        self.hlp_log_cls: float = 1.0 if (cls < 1) else 2.35 / log(cls) # log(0), overwritten later
        self.hlp_log_cls1: float = 1.0 if (cls < 1) else log(cls) / log(10.0) # ln(cls)/ln(10), ln(10)=2.30
        self.__growthdamper = 855000.0  # dampens growthrate if to many backsteps (P2M,CFFI)
        self.__errordamper = 500000.0  # dampens the possible worsening of function (P2M,CFFI)
        self.__truebestfkt: float = 0.0  # perhaps same: __best_fkt
        gstat.statist_Init()
        self.history_fx_boost: tt.Tensor = None
        self.history_btl_cnt: int = 0
        # Test
        self.TestGrads = [None] * 4
        self.TestCount = [0] * 4
        self.TestFktx = [0.0] * 4
        self.TestCountAll: int = 0
        self.BatchSizeExt: int = 8  # should match DataLoader
        self.MinCollect: int = 2 # how many micro-batches are always combined
        # Loss History (P2M)
        self.history_fx: tt.Tensor = None
        self.history_pos: int = 0 # if not in retrace
        self.history_sum: float = 0.0
        self.history_ssum: float = 0.0 # squares
        self.history_covar: float = 0.0 # sum x*y for covariance
        self.history_covar2: float = 0.0
        # self.history_min: float = 0.0 - unused
        # self.history_max: float = 0.0 - unused

    # Todo: scaling factor routine that operates safely with magnitude value of gradients

    def LGS_SetClasses(self, classes: int, gpubatchsize: int) -> None:
        "tell solver class-count and external batchsize, helpful for dynamic batchsize and noise estim."
        if (classes >= 1): self.__classes = classes # e.g. 10,100,200
        else: classes = max(0, self.__classes)
        print("LGS_SetClasses(cls=%d, ebs=%d, init=%s)" % (classes, gpubatchsize, str(self.__init)))
        self.hlp_log_cls: float = 2.35 / log(10.0) if (classes < 2) else 2.35 / log(classes)
        self.hlp_log_cls1 = log(classes) / log(10.0) # 1.0 / self.hlp_log_cls
        if (gpubatchsize >= 1): self.BatchSizeExt = gpubatchsize # e.g. 8,24,32, unused!
        return

    def LGS_SetDevice(self, dev: tt.device) -> None:
        "move Tensors to device to free GPU ram during fullbatch (end of epoch), PCIe4x16=32GB/s"
        if (self.__init): return
        from time import time as time_time
        byte: int = 0
        cnt: int = 0
        if (dev is None): dev = tt.device('cpu')
        dt: float = time_time()

        if (self.coll_grad is not None) and (self.coll_grad.device != dev):
            byte += tt.numel(self.coll_grad) * self.coll_grad.element_size()
            self.coll_grad = self.coll_grad.to(dev)
            cnt += 1
        #if (self.coll_grd is not None) and (self.coll_grd.device != dev):
        #    byte += tt.numel(self.coll_grd) * self.coll_grd.element_size()
        #    self.coll_grd = self.coll_grd.to(dev)
        #    cnt += 1
        #if (self.coll_grd2 is not None) and (self.coll_grd2.device != dev):
        #    byte += tt.numel(self.coll_grd2) * self.coll_grd2.element_size()
        #    self.coll_grd2 = self.coll_grd2.to(dev)
        #    cnt += 1
        # if (self.__mom is not None) and (self.__mom.device != dev): # only C2M
        # if (self.__yg is not None) and (self.__yg.device != dev): # only C2M
        if (self.__yg_old is not None) and (self.__yg_old.device != dev):
            byte += tt.numel(self.__yg_old) * self.__yg_old.element_size()
            self.__yg_old = self.__yg_old.to(dev)
            cnt += 1

        if (byte > 0):
            print("LGS: moved %d Tensors to dev=%s, %d KB / %.3f sec" % (cnt, str(dev), byte>>10, time_time() - dt))
        return

    def GetLossLimit(self) -> float:
        "worst loss before retrace (seldom update)"
        return log(self.__classes) if self.__init else self.__initfkt

    def LGS_TellTrainBoostLoss(self, train_loss:float) -> None:
        "get train_loss of boost-params (if avail)"
        if (self.history_fx_boost is None):
            self.history_fx_boost = tt.zeros(10, dtype=tt.float32, device='cpu')
            self.history_fx_boost[0] = train_loss
            self.history_btl_cnt = 1
        else:
            new_list = tt.roll(self.history_fx_boost, 1)
            new_list[0] = train_loss
            self.history_btl_cnt += 1
            if (self.history_btl_cnt >= 10):
                dv = self.history_fx_boost + new_list
                dv = (dv - tt.roll(dv, -1)) * 0.5
                n, a = tt.mean(dv[:3]).item(), tt.mean(dv[3:8]).item()
                r = n/a if (abs(a) > 1e-6) else (0.0 if (n<a) else 1e9)
                print("LGS_TellTrainBoostLoss(%d, %.3g/%.3g = %.3g)" % (self.history_btl_cnt, n, a, r))
            self.history_fx_boost = new_list
        return

    def LGS_CheckNextStep(self) -> tuple[bool, int]:
        "internal: only for MultiGpu SMP/DDP"
        if (self.collects is None): return True, 0
        return (len(self.coll_fxl) + 1) >= self.collects, self.collects

    @staticmethod
    def single_gradient_descent_step(x, alpha: float, grad: tt.Tensor):
        """
        wrapper routine for a gradient-descent step once the alpha value has been obtained. Its
        main purpose lies in generating more compact and adjustable code
        :param x: n-dim position in the function landscape
        :param alpha: externally obtained multiplication factor for step-length
        :param grad: gradient or sufficient gradient approximation obtained previously
        :return: the position in the function landscape is changed according to the gradient-descent step
        """

        # weight decay : 0.0=off, 0.01..0.1
        return tt.add(x, grad, alpha=-alpha)  # - (x * 0.01)

    def InternalsSave(self, fn: str = "state_lgs.txt") -> None:
        if (len(fn) < 2) or (self.__init): return
        f = open(fn, "wt")
        if (f.closed):
            print("lib_grad_solve/InternalsSave(%s)=failed." % fn)
            return
        scol: int = 0 if (self.collects is None) else self.collects
        __fkt_avg: float = 0.0 # only C2M
        f.write('LGS,%.4g,%.4g,%d,%.4g,%.4g,%.4g,%.4g,%.4g,%d\n' % (self.alpha, self.__beta, scol, self.__best_fkt,
               self.__initfkt, self.__noise_avg, self.__alpha_noise, __fkt_avg, self.MinCollect))
        f.close()
        # tt.save(tt.tensor(lst, dtype=tt.float32), fn)
        return

    def InternalsLoad(self, fn: str = "start_lgs.txt", silent:bool=True) -> None:
        if (len(fn) < 2): return
        from os import path

        f = open(fn, "rt") if (path.isfile(fn)) else None
        if (f is None) or (f.closed):
            if (not silent): print("lib_grad_solve/InternalsLoad(%s)=failed." % fn)
            return
        print("lib_grad_solve/InternalsLoad(%s)" % fn)
        # lst = tt.load(fn, weights_only=True).tolist()
        s: str = f.read(4)
        assert("LGS," == s), "wrong header"
        s = f.readline()
        f.close()
        assert(len(s) > 20), "short string"
        lst = s.split(',')
        scol: int = 0
        __fkt_avg: float = 0.0 # only C2M
        self.alpha, self.__beta, scol, self.__best_fkt, \
            self.__initfkt, self.__noise_avg, self.__alpha_noise, __fkt_avg, self.MinCollect = tuple(lst)
        self.collects = None if (scol < 2) else int(scol)
        assert (self.alpha > 0.0), "positive learning-rate"
        self.__init = False
        # if (f < -1.0): # LGS,0.2654,0.75,20,0.005529,5.621,0,0,5.621,0
        #    self.alpha = 0.2654; self.__beta=0.0; self.collects=20; self.__best_fkt=0.005529; self.__initfkt = 5.621
        return

    def UpdateHistoryFx(self, fx: float) -> None:
        "new loss history ring-buffer [256*2]"
        hpos: int = self.history_pos & 511 # %(1<<8) # mod len()
        last_hist: tt.Tensor = self.history_fx[hpos]
        # fxt: tt.Tensor = tt.tensor(fx, device='cpu', dtype=tt.float32)
        self.history_pos += 1
        if (last_hist != tt.tensor(fx, device='cpu', dtype=tt.float32)):
            last_item = last_hist.item()
            self.history_fx[hpos] = fx
            self.history_covar = 0.0
            for i in range(1, 513):
                self.history_covar += (-256.5+i)*self.history_fx[(hpos+i) & 511].item()
            self.history_sum += fx - last_item
            self.history_ssum += fx*fx - last_item*last_item
            #self.history_min = tt.min(self.history_fx).item() \
            #  if abs(last_item - self.history_min) < 1e-6 else min(self.history_min, fx)
            #self.history_max = tt.max(self.history_fx).item() \
            #  if abs(last_item - self.history_max) < 1e-6 else max(self.history_max, fx)
        return

    def FitHistoryFx(self) -> float:
        "fit loss history"
        if (self.history_pos < 512): return -1.0
        import numpy as np
        xvals = -1 * ((self.history_pos - np.arange(512)) & 511) # mod 256*2 (past=positive[step])
        par, res, _,_,_ = np.polyfit(xvals, self.history_fx, 2, full=True) # parabola = array([a,b,c])
        res = np.sqrt(res[0] * (1.0 / 512))
        xs = -np.inf if (par[0] < 1e-9) else par[1] / (-2.0*par[0]) # sign=(-1)^2, (future=positive[step])
        print("- LossParabola, xs=%.1f,r=%.3g, abc[%.3e,%.3e,%.3g]" % (xs,res, par[0],par[1],par[2]))
        return xs
        # plot
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(xvals, self.history_fx, 'b+') # label='loss/steps'
        fp = par[2] + (xvals*par[1]) + (xvals*xvals*par[0])
        plt.plot(xvals, fp, '-g')
        if (xs > -511.0) and (xs < 50.0): plt.plot([xs], [xs*0.5*par[1]+par[2]], 'ro')
        # plt.legend(loc='upper right')
        plt.show()
        return xs

    def FirstFnktVal(self, f: float, dim: int, unused_c2m: bool) -> None:
        """
        save initial f(x)
        :param f:
        :param dim:
        :param solver: True = c2m, False = p2m, unused
        :return:
        """
        self.__best_fkt = f
        self.__last_fkt = f
        self.__truebestfkt = f
        # self.__fkt_avg = f  # only-c2m
        assert(f > 0.0), "loss < 0"
        if (self.__classes >= 1):
            self.__initfkt = max(f, log(self.__classes))  # = curr_fkt # only-p2m
            self.hlp_log_cls1 = log(self.__classes) / log(10)
        else:
            self.__initfkt = max(f, log(2.0)) # min=2
            self.hlp_log_cls = (2.35 / f) if (f > 0.0) else 1.0
            self.hlp_log_cls1 = 1.0 # todo
        print("*fa, 0, %.4e, %.4e" % (self.__initfkt, self.alpha))

        if (self.__weight_decay_on is not None): # < 1.0
            assert (self.__weight_decay <= 1.0) and (self.__weight_decay >= 0.9), "0.9<weight_decay<1.0"
            # if (str(tt.get_default_dtype()).find('float16') > 0) and (self.__weight_decay >= 0.9990):
                # self.RandScaler = self.PrepareFuzzyScaler(self.__weight_decay)
            #    self.__weight_decay = 0.9995 # 0.99951172 = largest number less than one
            print("LGS: weight_decay = %.6f = 1 - %.2e = ON" % (self.__weight_decay, 1.0-self.__weight_decay))

        # self.__xbest = [] if (len(x) > 20) else tt.clone(x)  # xbest used only for low dim statistics
        #self.collects = 8  # 128 // self.BatchSizeExt # 16, under test: 32x8=256 (256 little better than 512)
        #self.MinCollect = 128 if (dim == 23910152) else 2 # ResNet50(24mio)
        #if False: #variant for c2min, distinction between RestNet50 and rest
        #    self.MinCollect = 64 if (dim == 23910152) else 2  # ResNet50(24mio) oder MinCollect = 64
        #    self.collects = 256 if (dim == 23910152) else 8  # 128 // self.BatchSizeExt # 16, under test: 32x8=256 (256 little better than 512)
        #else: # P2M: MC = 24 or 32 ==== DEFAULT (2024)
            # automatic calculation of the collection constant based on the batchsize used by the gpu and the target collector batchsize
        if (dim == 25549352): # ResNet50-1000 ImageNet1k
            self.MinCollect = max(2, int(128.0/self.BatchSizeExt + 0.9))
            self.statlength = 768*4
        elif (dim == 23910152 or dim == 11271432): # ResNet50-200 or ResNet18-200 TinyImageNet
            self.MinCollect = max(2, int(96.0/self.BatchSizeExt + 0.9))
        else:
            self.MinCollect = 2
            #self.MinCollect = max(2,int(96.0/self.BatchSizeExt + 0.9)) if (dim == 23910152) else (max(2,int(128.0/self.BatchSizeExt +0.9)) if (dim == 25549352) else 2)  # ResNet50(24mio) or MinCollect = 24
            #if (dim == 11271432): # ResNet18-200 (TinyImageNet-200)
            #    self.MinCollect = max(2, int(96.0/self.BatchSizeExt + 0.9))
        self.collects = self.MinCollect #4 * self.MinCollect

        self.lowerBs = self.MinCollect
        self.upperBs = int (1.5 * self.MinCollect)
        print("Enable Collect/Averaging, %d (x%d) step." % (self.collects, self.BatchSizeExt), flush=True)
        hlen: int = 1<<9 # len=2^int (256 or 512)
        f11: float = f * 1.1
        self.history_fx = tt.ones(hlen, dtype=tt.float32, device='cpu') * f11
        self.history_pos = 0
        self.lastFmean = f11
        self.lastDev = 0.3*f11
        self.history_sum, self.history_ssum = f11 * hlen, f11 * f11 * hlen
        #self.history_min, self.history_max = f11, f11 #tt.min(self.history_fx).item(), tt.max(self.history_fx).item()
        # self.collects = 1 # disable DBS (dynamic batchsize) here
        self.__init = False
        self.InternalsLoad()
        return

    @staticmethod
    def Contrast(a: float, b: float):
        "(a-b)/(a+b), Interferometric visibility = Michelson-Kontrast"
        s: float = a + b
        return float('nan') if (abs(s) < 1e-12) else abs((a - b) / s)

    @staticmethod
    def LoadSavedX(n: int) -> tt.Tensor:
        "load x vector from disk (unused)"
        assert (n >= 1), "LoadSavedX: empty dimension"
        fn: str = "lastx_" + str(int(n)) + ".pt"
        # assert(path.isfile(fn))
        return tt.load(fn, weights_only=True)

    @staticmethod
    def StdDev(lst: list[float]):
        if (len(lst) <= 1):
            if (len(lst) < 1): return float('nan'), float('nan')
            return float('nan'), float(lst[0])
        if (isinstance(lst, list)):
            lst = tt.tensor(lst, dtype=tt.float32)
        dev, avg = tt.std_mean(lst)
        return dev.item(), avg.item()

    # def _alpha_fac(self, cs: float) -> float:  # ctype=0, only C2M

    def __adjust_initial_alpha(self, x, alpha: float, fkt, grad_fkt) -> float:
        # only used for: run_mathematical_example_problems.py
        """
        this function adjusts the value of alpha in the initial step to ensure stability
        of the iteration and thus avoid catastrophic convergence loss. If needed the
        initial alpha is halved until we have a true descent
        :param x: parameter point
        :param alpha: initial stepwidth
        :param fkt: function which we aim to minimize
        :param grad_fkt: gradient of the function
        :return: re-adjusted alpha
        """
        if (fkt is None):
            print("adjust_initial_alpha needs lambda fkt != None!")
            return self.alpha

        f:  float = fkt(x)
        ft: float = f + 0.2 * abs(f)

        if (isinstance(grad_fkt, tt.Tensor)):
            yg = grad_fkt
        else:
            yg = tt.tensor(grad_fkt)
        alphaset: float = alpha
        self.jaccalls += 1
        while (ft > f):
            xt = SelfConstOptim.single_gradient_descent_step(x, alphaset, yg)
            alphaset *= 0.5
            self.fcalls += 1
            if (alphaset <= 1.0e-20):
                break
            ft = fkt(xt)
        return alphaset

    # def __update_momentum(self, yg: tt.Tensor) -> None: # only C2M
    # def store_optional_best_val(self, f: float, x: tt.Tensor) -> None: # only C2M
    # def UpdateBeta(self, cos: float) -> float:  # C2M, CFFI
    # def UpdateBetaDF(self, f: float) -> float:  # C2M

    def cosine(self, x: tt.Tensor, y: tt.Tensor) -> float:  # unused
        """
        calculates the cosine of two vectors in any dimension via the well-known
        connection to the scalar product applicable in any dimension
        :param x: first vector on input
        :param y: second vector on input
        :return: cosine value, except for very small vectors where the result is forced
        to be zero
        """
        dd: float = tt.dot(x, x).item() * tt.dot(y, y).item()  # unit = norm^4
        if (dd < 1e-38): return 0.0  # to avoid division by zero
        # if(np.isnan(dd) or dd > 1e222):
        #    print("Bad number in cosine function, aborting...", dd); exit(0)
        dt: float = tt.dot(x, y).item()

        if (dd < 1e38): # FLT_MAX=3e38
            return dt / sqrt(dd)

        dd = x.norm().item() * y.norm().item()
        if (dd < 1e38):
            return dt / dd

        print("LibG.cos1!", dt, dd)  # crash(dd=6e25,in 4=toy)
        return -0.5 if (dt < 0.0) else 0.1 # todo

    # def cosine_fast(self, g_new: tt.Tensor, g_old: tt.Tensor) -> float: # only C2Min

    def SwitchWeightDecay(self, x_norm: float) -> None:
        "toggle WD periods (3500 steps)"
        if (self.decay_perm): # 1
            if (self.collects <= 4 or self.step_cnt > 85000): # cooldown or not
                self.__weight_decay = min(1.0, (self.__xoldnorm/x_norm) ** 0.00057) # 0.00057=2/3500, this exponent is needed to decay in 3500 steps an increase over 7000 steps #self.__weight_decay_backup
                #assert(self.__weight_decay <= 1.0), "possible?"
                print('*decay with %.6f = 1 - %.3e' % (self.__weight_decay, 1.0-self.__weight_decay)) #self.__weight_decay_backup)
                self.__weight_decay_on = True if (self.__weight_decay < 1.0) else None
        else: # 0
            if (self.__weight_decay == 1.0): # cooldown or not
                self.__weight_decay = min(1.0, (self.__xoldnorm/x_norm) ** 0.00057) # 0.00057=2/3500, this exponent is needed to decay in 3500 steps an increase over 7000 steps #self.__weight_decay_backup
                #assert(self.__weight_decay <= 1.0), "possible?"
                print('*decay with %.6f = 1 - %.3e' % (self.__weight_decay, 1.0-self.__weight_decay)) #self.__weight_decay_backup)
                self.__weight_decay_on = True if (self.__weight_decay < 1.0) else None
            else:
                # if (self.collects >= 5):
                #    self.__weight_decay = sqrt(self.__weight_decay)
                #    print('*decay with %.6f = 1 - %.3e' % (self.__weight_decay, 1.0 - self.__weight_decay))
                #else:
                self.__weight_decay = 1.0
                self.__weight_decay_on = None
                print('*nodecay')
        return

    def GradientMerge(self, fkt: float, grad_fkt: tt.Tensor, x: tt.Tensor):
        "MicroBatchMerge: average gradients (micro-batch to dynamic mini-batch)"  # ResNet50+, LLM
        # if (self.collects < 2):  # or (not self.__init): # earlier
        #    return fkt, 0.0, grad_fkt  # early exit

        if (grad_fkt is None):
            self.coll_grad = None
            self.coll_fxl = []
            return fkt, 0.0, None # inf = force retrace

        self.coll_cnt += 1
        if not (self.step_cnt & 3) and (self.step_cnt >= 100): # once per 100 moving steps
            #if (self.__weight_decay_backup is not None): # < 1.0):
            #    if (self.decay_perm): # 1
            #        if not (self.step_cnt % 100): #3500):
            #            self.SwitchWeightDecay( x.norm().item() )
            #    else:
            #        if not (self.step_cnt % 3500):
            #            self.SwitchWeightDecay( x.norm().item() )
            # self.TestCollect(-1.1, grad_fkt)
            modo:int = max(1, int(self.statlength/(self.collects*self.BatchSizeExt)))*100 # number of steps used for the average
            if not (self.step_cnt % 100):  # Tune-Here around 0.01
                if (self.step_cnt == 400):
                    self.lastFmean = self.sumFvalue/400.0
                    self.lastDev = sqrt(abs(self.sumSquareFvalue-self.sumFvalue*self.lastFmean)/400)
                if (self.step_cnt2 == modo):  # Tune-Here around 0.01
                    #if (self.coll_ggc<self.Gold):
                    #    self.beginn = 1.0
                    #self.Gold = self.coll_ggc
                    self.step_cnt2 = 0
                    xaver:  float = self.sumFvalue / modo
                    xaver2: float = sqrt(abs(self.sumSquareFvalue-self.sumFvalue*xaver) /modo)
                    self.sumFvalue = self.sumSquareFvalue = 0.0
                    assert(xaver > 0.0), "this optimizer needs loss > 0"
                    quod: float = xaver2 / xaver
                    print('xaver: %.6f ± %.6f, q: %.6f, min: %.6f' % (xaver, xaver2, quod, self.MinOfF))
                    if (xaver > self.lastFmean):
                        self.lastRed += 1
                    else:
                        self.lastRed = 0
                    #if ((xaver > self.test+xaver2*0.1) and (quod < 0.5) and (self.MinOfF > self.MinOfF2)):
                    if (xaver > self.lastFmean+xaver2*0.1) or (quod > 0.5) or (self.lastRed == 3):
                        self.Increase = True
                        self.lastFmean = xaver
                    self.lastFmean = min(xaver, self.lastFmean)
                    self.lastDev = xaver2
                    self.MinOfF2 = min(self.MinOfF, self.MinOfF2)
                    self.MinOfF = xaver
                scol: int = self.collects
                if (self.Increase): #(self.step_cnt == self.cool):
                    scol = int(scol * 1.5)
                    self.Increase = False
                    if (scol > self.upperBs):
                        scol = self.lowerBs
                        self.upperBs = int(self.upperBs * 1.5)
                        self.lowerBs = int(self.lowerBs * 1.5)
                self.collects = max(self.MinCollect, min(scol, 6250)) #max(2, min(self.collects, 4000 // self.BatchSizeExt))  # effective bs<=2000
                self.__retrace2 = 0
                print(
                    "# Increase collects=%d, steps=%d+%d+%d, G=%.3g, a=%.3g." %
                    (self.collects, self.coll_cnt, self.step_cnt, self.coll_inc, self.coll_ggc * 0.01,self.alpha), flush=True)
                self.coll_cnt = 0  # self.step_cnt
                self.coll_inc += 1
                self.step_cnt += 1  # prevent increase loop
                self.step_cnt2 += 1  # prevent increase loop
                self.coll_ggc = 0.0
                self.__retrace = False
                self.InternalsSave()

        if not (fkt < 1e38): # isnan(fkt), float32max = 3e38
            # self.coll_cnt -= 1
            return fkt, None, None  # skip this one (leave here)

        grad_nrm: float = grad_fkt.norm().item()
        if not (grad_nrm < 65e3): # float16 + big-alpha = nan-gradient
            self.coll_GradFails += 1
            print("Warn:GradFail, n=%.3g, cnt=%d, f=%.3g, a=%.3g, SKIP!" %
                 (grad_nrm, self.coll_GradFails, fkt, self.alpha))
            return inf, 0.0, None # force retrace

        hlen: int = len(self.coll_fxl)
        # sel: int = hlen + (hlen >> 1) + (hlen >> 2)  # Python 3.9 has no int.bit_count()
        factor: float = self.MinCollect/(self.collects*self.collects*2.0) #1.0/self.collects
        if not hlen:
            self.coll_grad = tt.mul(grad_fkt, factor)
            #self.coll_grad = grad_fkt.clone()
        else:
            tt.add(self.coll_grad, grad_fkt, alpha=factor, out=self.coll_grad)
            #self.coll_grad += grad_fkt

        self.coll_fxl.append(float(fkt))

        if (1 + hlen < self.collects):
            return fkt, None, None  # continue collecting (leave here)

        #self.coll_grad /= self.collects

        self.coll_grad, rv = None, self.coll_grad # todo: create rv in-place (save memory)

        fc_dev, fkt = self.StdDev(self.coll_fxl)
        self.coll_fxl = []  # reset lists (self.coll_ggl)
        self.step_cnt += 1 # should happen here (do not count retrace steps)
        self.step_cnt2 += 1 # should happen here (do not count retrace steps)
        return fkt, fc_dev, rv

    # def c2min_pv_step(self, x: tt.Tensor, fkt: float, grad_fkt: tt.Tensor, fkt_alpha: float = None, combi_step: bool = False):

    # @jit(nopython=True, nogil=True)
    def growth_func(self) -> float:  # CFFI
        """
        defines the growth function for the parabola-fitting algo
        :return: gives the growth value
        """
        # A1 : float = 1000000.0
        # A2 = 855000
        damp: float = self.__growthdamper
        if (self.__retrace):
            if (damp > 1.0):
                if (damp < (1000000.0 / 16.0)):
                    damp *= 16.0
                else:
                    damp = 855000.0
            else:
                damp += 16.0
        else:
            damp *= (1.0 / 3.9)
        self.__growthdamper = damp
        return 1000000.0 / (1.0 + damp)

    # @jit(nopython=True, nogil=False)
    def __alpha_opt_version(self, alpha: float, f: float, fold: float, yg, yg_old) -> float:
        """
        sets the adaptive stepwidth by fitting a parabola to the function choosing
        the minimum value and extrapolating on the basis of this minimum
        :param alpha:
        :param f:
        :param fold:
        :param yg_old:
        :return:
        """

        Ga: float = self.__norm_yg_old * self.__norm_yg_old # tt.dot(yg_old, yg_old).item()
        g: float = self.growth_func()  # also internal update of the growthdamper
        # g: float = c_growth_func(self.__retrace) # CFFI
        if (self.__retrace):
            # div = f - fold
            d: float = (f - fold) + alpha * Ga
            m: float = (0.5 * alpha * Ga) / d # crash DIV/0
            #if (len(yg) >= 21) and (f < 1.1*self.__initfkt):
            #    m = max(m, 0.01)
            if (m < inf): # isnan(m) or isinf(m), 1e999==inf (fastest)
                alpha *= m
            else:
                alpha *= 1.0 / 100.0
            return alpha
        else:
            S: float = tt.dot(yg, yg_old).item()  # here Skalarprodukt
            h: float = 1.0 - (S / Ga)  # (Ga - S) / Ga
            if (g * h < 1.0):
                alpha *= g
            else:
                if (len(yg) <= 20):
                    alpha /= h
                else:
                    self.__norm_yg_old = yg.norm().item()
                    #hlp: float = self.hlp_log_cls # 2.35 / log(self.__classes) # (2.35/self.__initfkt)
                    #alpha /= h * (1.0 - 1.0*log2(2*self.collects/self.MinCollect) / (1.0 + 0.5 * self.alpha) * 0.14 * max(0.0, min(0.7, (hlp * self.__truebestfkt) ** 0.8)))
                    #alpha /= h * (1.0 - 1.0 * log2(2 * self.collects / self.MinCollect) / (1.0 + 0.5 * self.alpha) * 0.14 * max(0.0, min(0.7, (hlp * self.__truebestfkt) ** 0.7)))
                    # alpha /= h * (1.0 - 0.14 * max(0.0, min(0.7, hlp * (self.__truebestfkt) ** 0.8)))
                    #alpha /= h * (1.0 - 0.15 * self.beginn / (1.0 + 0.5 * self.alpha *self.alpha))
                    alpha /= h * (1.0 - 0.15 / (1.0 + 0.5 * self.alpha * self.alpha))
                    N: float = alpha * self.__norm_yg_old
                    if (self.__clipping < N): # hier evtl auf float32 testen
                        if (N > 1.8): print("Clip: N=%.2g, a=%.2g !" % (N, alpha))
                        alpha *= self.__clipping / N
                        assert(alpha > 1e-30), "clipping nan, zero, etc"
                        return max(min(alpha, 1.0e6), 1.0e-12)
        #return max(min(alpha, 1.0e6), 1.0e-8)  # restricts alpha to 1e-8...1e+6
        if (self.collects > self.MinCollect):
            return max(min(alpha, 1.0e6), 1.0e-2)
        return max(min(alpha, 1.0e6), 0.1/self.__norm_yg_old)  # restricts alpha to 1e-8...1e+6

    def update_best_values(self, x, f: float) -> None:
        if (f < self.__best_fkt):
            self.__best_fkt = f
            if (len(x) <= 20):
                self.__xbest = self.copy_value(x)
        return

    def gen_output(self, x: tt.Tensor) -> None:
        if (self.trackHistory):
            self.xhist.append(x)
            self.ahist.append(self.alpha)
        return

    def copy_value(self, x: tt.Tensor) -> tt.Tensor:
        if (isinstance(x, tt.Tensor)):
            return x.clone()
        else:
            return tt.clone(x)

    def UpdateErrorDamper(self, currfktlesslastfkt: bool) -> None:  # todo: cffi
        error: float = self.__errordamper
        if (currfktlesslastfkt):  # update of the errordamper depending on worsening of result without retrace
            self.__errordamper = error * (1.0 / 1.69)
        else:
            if (error < 1.0):
                self.__errordamper = error + 16.0
            else:
                self.__errordamper = error * 16.0 if (error < 500000.0) else 500000.0 # why not max() ?
        return

    #def CalcErrorTolerance(self, truebestfkt: float) -> float:  # CFFI
    #    # self.__truebestfkt = truebestfkt
    #    return 25.0 + (2500.0 + 71.0 * sqrt(truebestfkt)) / (1.0 + sqrt(self.__errordamper))

    def p2min_step(self, x: tt.Tensor, fkt: float, grad_fkt, combi_step: bool = False):
        """
        performs a single self-consistent gradient descent step updating the global
        storage accordingly for the next step
        :param x: Tensor
        :param fkt: f(x) function-value = e.g. loss
        :param grad_fkt: gradient
        :combi_step: averaged x for combi_step (once per epoch, if benefitial only)
        :return: tuple(x,_,_) | bool
        """

        #if (x is None):  # "decide if gradient is skipped (revert-steps)"
        #    assert (grad_fkt is None)  # test: x=None + grad=None (optional)
        #    return False  # TODO: future (remove this line to activate)
        #    if (self.collects is not None) and (len(self.coll_fxl) < self.collects):
        #        return False  # no skip within collection (gradient sum)
        #    return not (fkt < inf)  # default=False (no skip grad) # unreachable in collect=on

        if (self.collects is not None):  # <2=off
            fkt, fc_dev, grad_fkt = self.GradientMerge(fkt, grad_fkt, x)
            if (fc_dev is None):
                return None, [], False

        # self.step_cnt += 1
        statp2min = []

        # load function values
        curr_fkt: float = fkt
        # assert isinstance(grad_fkt, tt.Tensor), "torch tensor"
        yg = grad_fkt
        # if (grad_fkt is not None):
        #     yg = grad_fkt if (isinstance(grad_fkt, tt.Tensor)) else tt.tensor(grad_fkt)
        # else:
        #     yg = None

        # curr_fkt = fkt(x)
        # self.fcalls += 1 # not needed
        # self.jaccalls += 1 # not needed
        self.solversteps += 1
        tiny_x: bool = len(x) <= 20

        if (self.__init):
            self.sumFvalue = curr_fkt
            self.sumSquareFvalue = curr_fkt * curr_fkt
            self.MinOfF = self.MinOfF2 = curr_fkt
            # self.alpha = 0.00001  # not important
            self.__beta = 0.0 # no moment in p2m, only used in print
            self.FirstFnktVal(curr_fkt, len(x), False)
            self.__xbest = [] if (not tiny_x) else self.copy_value(x)
            self.__yg_old = yg
            self.__xoldnorm = x.norm().item()
            self.__norm_yg_old = yg.norm().item()
            # c_InitOptimP2M(0.0, 0.0); # CFFI
            if (tiny_x):
                xp = self.copy_value(x)
            self.gen_output(x)

            tt.add(x, yg, alpha=-self.alpha, out=x)
            if (tiny_x):
                f = fkt
                self.fhist.append(f)
                self.gen_output(x)
                statp2min = (self.solversteps, xp, tt.tensor([curr_fkt]), yg, tt.tensor(
                    [self.alpha]), x, tt.tensor([f]))
            return x, statp2min, False # self.converged
        else:
            currfktlesslastfkt: bool = True  # is current function smaller than last function
            self.__best_fkt = curr_fkt if (self.__best_fkt > curr_fkt) else self.__best_fkt
            if (tiny_x):
                xp = self.copy_value(x)
                start: bool = True
            else:
                start: bool = (curr_fkt < 1.1 * self.__initfkt)
            # self.CalcErrorTolerance(self.__truebestfkt)
            #errorTolerance: float = 25.0 + (2500.0 + 71.0 * sqrt(self.__truebestfkt)) / (1.0 + sqrt(self.__errordamper))
            # if start and (curr_fkt < errorTolerance * 1.1 * self.__truebestfkt):
            if start and (curr_fkt < self.lastFmean + 5.0 * self.lastDev):
                self.sumFvalue += curr_fkt
                self.sumSquareFvalue += curr_fkt * curr_fkt
                self.MinOfF = min(self.MinOfF, curr_fkt)
                self.x_lastgood = x.clone() # perhaps only for float16
                # simple step based on the adapted alpha
                if (curr_fkt > self.__last_fkt):
                    currfktlesslastfkt = False
                self.__last_fkt = curr_fkt
                self.__retrace = False
                self.alpha = self.__alpha_opt_version(self.alpha, curr_fkt, self.__last_fkt, yg, self.__yg_old)
                self.__yg_old = yg
                if (self.__weight_decay_on is not None): # decays the parameters, optional
                    x -= tt.add(x * (1.0 - self.__weight_decay), yg, alpha=self.alpha)
                    # if (x.element_size() >= 4): # float32
                else:
                    tt.add(x, yg, alpha=-self.alpha, out=x)  # in-place (out=x was crash)
                # self.UpdateHistoryFx(curr_fkt)
            else:
                # retracing step in case the step implied a function value growing too much
                old_retrace: bool = self.__retrace
                self.__retrace = True
                # if True: #(start):
                self.__retrace2 += 1
                #self.alphaRed *= 1.1
                if (self.__last_fkt <= self.__truebestfkt):
                    self.__truebestfkt *= 10.0
                else:
                    self.__truebestfkt *= (self.__last_fkt/self.__truebestfkt) ** 0.125
                #if (self.__retrace2 == 2) and (self.collects is not None):
                #    self.collects += 2
                # retracing to the previous x
                old_alpha: float = self.alpha
                # computation of new alpha
                self.alpha = self.__alpha_opt_version(old_alpha, curr_fkt, self.__last_fkt, yg, self.__yg_old)
                print("Retrace(%d): f=%.3g(%.3g), a=%.3g, gn=%.3g, or=%d" %
                    (self.__retrace2, curr_fkt, self.__last_fkt, old_alpha, self.__yg_old.norm().item(), int(old_retrace)))
                if (self.x_lastgood is not None):
                    axn: float = x.norm().item()
                    # x = self.x_lastgood - tt.add(self.x_lastgood * (1.0 - self.__weight_decay), self.__yg_old, alpha=self.alpha)
                    # tt.add(self.x_lastgood, self.__yg_old, alpha=-self.alpha, out=x)
                    tt.add(self.x_lastgood, self.__yg_old, alpha=-self.alpha, out=x)
                    print("RX = %.6g, %.6g, %.6g" % (self.x_lastgood.norm().item(), x.norm().item(), axn)) # debug (3x norm() = slow!)
                    # self.x_lastgood = None # 2x retrace
                else:
                    if (self.__weight_decay_on is not None) and (not old_retrace):
                        decay_inv: float = 1.0 / self.__weight_decay # >1
                        x += tt.add(x * (decay_inv-1.0), self.__yg_old, alpha=old_alpha*decay_inv-self.alpha)
                    else:
                        tt.add(x, self.__yg_old, alpha=old_alpha-self.alpha, out=x)

                    t1, t2 = tt.tensor(old_alpha), tt.tensor(self.alpha) # old_alpha>self.alpha
                    # x = SelfConstOptim.single_gradient_descent_step(x, -self.alpha, self.__yg_old)
                    if 0.0 == float((t2 - t1) - t2): # <(1e-4 or 1e-8)
                    # 2nd computation of new x with new alpha (in round=0 case, float16)
                        tt.add(x, self.__yg_old, alpha=-self.alpha, out=x) # in-place (combined with last add)
                # x = SelfConstOptim.single_gradient_descent_step(x, self.alpha, self.__yg_old)

            #if (curr_fkt < self.__truebestfkt):  # setting new best function, if actual better
            #    self.__truebestfkt = curr_fkt
            #else:
            #    self.__truebestfkt *= 1.1
            #self.UpdateErrorDamper(currfktlesslastfkt)  # self (CFFI)

            #statistic
            col: int = 0 if (self.collects is None) else self.collects
            gstat.statist_AddNumbers([sqrt(self.__norm_yg_old), self.alpha, self.__weight_decay, self.__growthdamper, self.__errordamper, col])
            self.coll_ggc += self.__norm_yg_old
            # if self.step_cnt <= 25:
            #    print("*fa, %d, %.4e, %.4e" % (self.step_cnt, fkt, self.alpha)) # debug
            # f: compute the function value (for output-purposes only)
            if (not tiny_x):
                if (curr_fkt < self.__best_fkt): self.__best_fkt = curr_fkt # update_best_values()
                return x, statp2min, self.__retrace
            else: # (tiny_x):
                f: float = fkt
                # if (len(x) == 2):  # for SaddlePlots
                #    print("WALK_P2M: %d %.6f  %.6f %.6f" % (self.solversteps, f, x[0], x[1]))
                # self._count+=1 ; self.__calls+=2
                self.fhist.append(f)
                statp2min = (self.solversteps, xp, tt.tensor([curr_fkt]), yg, tt.tensor(
                    [self.alpha]), x, tt.tensor([f]))
                self.update_best_values(x, f)

            self.gen_output(x) # !! trackHistory=False
            # if (curr_fkt == f): # todo: check reason for this
            ##self.signal[2]=True
            # self.converged=True
            if (self.solversteps >= self.signal[1]):
                self.signal[2] = True
                self.converged = False
            if (self.__best_fkt <= self.signal[0]):
                self.signal[2] = True
                self.converged = True
            return x, statp2min, self.__retrace

    def p2min_greedy_solver(self, x, fkt, grad_fkt, flev: float = 1.0e-15, maxit: int = 10000):
        # ("USED:p2min_greedy_solver")
        self.signal[0] = flev
        self.signal[1] = maxit
        converged: bool = True
        print("IN1:", type(grad_fkt))

        # if (not isinstance(x,tt.Tensor)):
        x = tt.tensor(x)
        stats = []

        for i in range(maxit):
            x, finfo, converged = self.p2min_step(x, fkt(x), grad_fkt(x))
            stats.append(finfo)

            if (self.signal[2]):
                break
        x = x.numpy()

        if (converged):
            self.print("p2min_greedy: successful convergence !")
        else:
            self.print("p2min_greedy: solver did not converge in #=", maxit, "steps!")
        if (len(self.__xbest) <= 20):
            self.print("p2min_greedy: obtained parameter results:", self.__xbest)
        self.print("p2min_greedy: obtained optimal functional value:", self.__best_fkt)
        self.print("p2min_greedy: last alpha value :", self.alpha)
        self.print("p2min_greedy: # of actual steps made=", self.solversteps)
        self.print("p2min_greedy: # of function calls=", self.fcalls)
        self.print("p2min_greedy: # of Jacobians called=", self.jaccalls, )
        self.print("p2min_greedy: excess work:", (100 * self.fcalls) / self.solversteps, "%")
        return x, stats

    # def cos2min_greedy_solver(self, .. )

# EoF.
