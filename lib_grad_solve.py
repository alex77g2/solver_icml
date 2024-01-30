# key subroutines for the application of the gradient descent algo
# for ICML 2024

# from numba import jit, int32, float32
# import cython
# from _optim.lib import c_alpha_fac, c_InitOptimP2M, c_InitOptimC2M, c_UpdateBeta, c_growth_func, c_UpdateErrorDamper, c_CalcErrorTolerance
# import numpy as np
import torch as tt
from math import sqrt, log2
from time import time as time_time
from statistic_helper import GlobalStatist as gstat


class SelfConstOptim:

    @staticmethod
    def print(*values: object) -> None:

        if False:
            print(values)

    def __init__(self, lr: float = 1e-6):
        self.debug_out: bool = True
        self.__init: bool = True
        self.trackHistory: bool = False
        self.__best_fkt = 99.0e99  # best f(x) this far
        self.__initfkt = 0.0  # initial function value (only p2m)
        self.alpha = 0.00001  # LR = initial alpha
        # self.fixed_beta:bool = True
        self.__beta: float = 0.75
        self.__noise_avg: float = 0.0
        self.__alpha_noise: float = 0.0
        self.__fkt_avg: float = 0.0  # only-c2m
        # self.__fractional_steps: float = 1.0  # 1e-9..1.0 (for LLM or SGD-noise)
        # self.__weight_decay: float = 0.0 # 0.0=off, 0.01..0.1
        # self.__ndim: int = 0 # dimension
        # self.__progress = []
        # self.__path = 0.0
        self.coll_inc: int = 0  # new (ResNet50)
        self.coll_cnt: int = 0  # new (ResNet50)
        self.step_cnt: int = 0  # new (ResNet50)
        self.adjustinit: bool = False  # True only for low-dim !
        self.__last_fkt: float = 0.0
        # self.__initadjust:int = 0
        self.xhist = []
        self.ahist = []
        self.fhist = []
        # self.FktLimit: float = 1e99 # future: skip-gradient-calc
        # DynamicBatchSize (Nov.2023)
        self.collects: int = 1  # target=const, <2 off
        self.coll_fxl: list[float] = []  # list of f(x) for combined steps
        self.coll_ggl: list[float] = []  # test
        self.coll_ggc: float = 0.0
        self.coll_cos: tt.Tensor = tt.ones(100, dtype=tt.float32, device='cpu')  # [1.0] * 100 # test (RC1)
        # self.coll_LastGrd: tt.Tensor = None # test
        # self.coll_EpoSteps: int = 0
        # self.coll_AllSteps: int = 0
        self.coll_grd: tt.Tensor = None  # skipping model.zero_grad() could avoid extra Tensor here
        self.coll_grd2: tt.Tensor = None  # debug/test
        # self.no_mom = False
        self.fcalls: int = 0
        self.jaccalls: int = 0
        self.solversteps: int = 0
        self.signal = [1.0e-8, 10000, False, False]
        self.converged: bool = False
        self.__yg: tt.Tensor = tt.tensor([])
        self.__yg_old: tt.Tensor = tt.tensor([])
        self.__norm_yg_old: float = 0.0
        self.__retrace: bool = False  # (CFFI)
        # self.__maxgrowth = 25.0 # unused
        # self.__growth = 1
        self.__growthdamper = 855000.0  # dampens growthrate if to many backsteps (P2M,CFFI)
        self.__errordamper = 500000.0  # dampens the possible worsening of function (P2M,CFFI)
        self.__truebestfkt: float = 0.0  # perhaps same: __best_fkt
        gstat.statist_Init()
        # Test
        self.TestGrads = [None] * 4
        self.TestCount = [0] * 4
        self.TestFktx = [0.0] * 4
        self.TestCountAll: int = 0
        self.BatchSizeExt: int = 8  # should match DataLoader
        self.MinCollect: int = 2 # how many micro-batches are always combined
        # self.collects = 32 # 8*32=256

    # Todo: scaling factor routine that operates safely with magnitude value of gradients

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
        f.write('LGS,%.4g,%.4g,%d,%.4g,%.4g,%.4g,%.4g,%.4g,0\n' % (self.alpha, self.__beta, self.collects, self.__best_fkt,
               self.__initfkt, self.__noise_avg, self.__alpha_noise, self.__fkt_avg))
        f.close()
        # tt.save(tt.tensor(lst, dtype=tt.float32), fn)
        return

    def InternalsLoad(self, fn: str = "state_lgs.txt", silent:bool=True) -> None:
        if (len(fn) < 2): return
        f = open(fn, "rt")
        if (f.closed):
            if (not silent): print("lib_grad_solve/InternalsLoad(%s)=failed." % fn)
            return
        print("lib_grad_solve/InternalsLoad(%s)" % fn)
        # lst = tt.load(fn).tolist()
        s: str = f.read(4)
        assert("LGS," == s), "wrong header"
        s = f.readline()
        f.close()
        assert(len(s) > 20), "short string"
        lst = s.split(',')
        self.alpha, self.__beta, self.collects, self.__best_fkt, \
            self.__initfkt, self.__noise_avg, self.__alpha_noise, self.__fkt_avg, _ = tuple(lst)
        assert (self.alpha > 0.0), "positive learning-rate"
        self.__init = False
        return

    def FirstFnktVal(self, f: float, dim: int, solver_c2m: bool) -> None:
        """
        save initial f(x)
        :param f:
        :param dim:
        :param solver: True = c2m, False = p2m
        :return:
        """
        self.__best_fkt = f
        self.__last_fkt = f
        self.__truebestfkt = f
        self.__fkt_avg = f  # only-c2m
        self.__initfkt = f  # = curr_fkt # only-p2m
        # self.__xbest = [] if (len(x) > 20) else tt.clone(x)  # xbest used only for low dim statistics
        self.collects = 8  # 128 // self.BatchSizeExt # 16, under test: 32x8=256 (256 little better than 512)
        #self.MinCollect = 128 if (dim == 23910152) else 2 # ResNet50(24mio)
        if solver_c2m: #variant for c2min, distinction between RestNet50 and rest
            self.MinCollect = 64 if (dim == 23910152) else 2  # ResNet50(24mio) oder MinCollect = 64
            self.collects = 256 if (dim == 23910152) else 8  # 128 // self.BatchSizeExt # 16, under test: 32x8=256 (256 little better than 512)
        else: # MC = 24 or 32 
            self.MinCollect = 12 if (dim == 23910152) else 2  # ResNet50(24mio) or MinCollect = 24
            self.collects = 4*self.MinCollect
        print("Enable Collect/Averaging, %d (x%d) step." % (self.collects, self.BatchSizeExt))
        # self.collects = 1 # disable DBS (dynamic batchsize) here
        self.__init = False
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
        return tt.load(fn)

    @staticmethod
    def StdDev(lst: list[float]):
        if (len(lst) <= 1):
            if (len(lst) < 1): return float('nan'), float('nan')
            return float('nan'), float(lst[0])
        if (isinstance(lst, list)):
            lst = tt.tensor(lst, dtype=tt.float32)
        dev, avg = tt.std_mean(lst)
        return dev.item(), avg.item()

    def _alpha_fac(self, cs: float):  # ctype=0, only C2M
        """
        maps a result of the cosine(s) of vector(s) to a scaling of the gradients. Empirically,
        a linear interpolation for the range: [0.5,1.5], leads to improved results and greater
        stability in the self-consistent change of the gradients-lengths for a plain vanilla
        implementation of cos2min, as deceleration is faster than acceleration. Slightly larger
        factor for positive cosine prevents accumulated deceleration over time if cosine stays
        essentially zero.
        :param cs: cosine value from the scalar product that can be obtained in any dimension
        :return: multiplicative growth factor with which the gradient is multiplied
        """

        t: float = 1.0 + 0.5 * cs if (cs <= 0.0) else 1.0 + 0.6 * cs
        return t

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

        f: float = fkt(x)
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

    def __update_momentum(self, yg) -> None:
        "update momentum (for c2m)"
        # assert((yg-self.__yg_old).norm().item() < 1e-9), "final-test" # ok
        self.__mom = tt.add((self.__mom * 0.8), tt.add(yg, self.__yg_old), alpha=0.2 * 0.5) # TODO
        # self.__mom = tt.add((self.__mom * 0.8), yg, alpha=0.2) # same (but untested)
        # return tt.norm(self.__mom) / tt.norm(yg) # mdg (<1% time)

    def store_optional_best_val(self, f: float, x) -> None:
        "only for low dim math tests"
        self.__last_fkt = f
        if (f < self.__best_fkt):
            self.__best_fkt = f
            assert (len(x) <= 100), "not for high-dim vectors"
            if (len(x) <= 20):
                self.__xbest = tt.clone(x)
        return

    def UpdateBeta(self, cos: float) -> float:  # C2M, CFFI
        self.__alpha_noise = self.__alpha_noise * 0.95 + 0.05 * abs(
            log2(self._alpha_fac(cos)))  # c_abs_log2_afac(cos) # CFFI-todo
        # self.__beta = 0.7
        # beta between 0.59 and 0.79
        if (self.__alpha_noise > 0.2):
            return min(0.79, self.__alpha_noise * 0.1 + 0.75)
        elif (self.__alpha_noise > 0.05):
            return 0.59 + 0.9 * self.__alpha_noise
        else:
            return self.__alpha_noise * 12.9
        # self.__beta = return

    def UpdateBetaDF(self, f: float) -> float:  # C2M
        self.__fkt_avg = 0.9 * self.__fkt_avg + 0.1 * f
        # relative noise level
        noise = min(1.06, (self.__initfkt*(1/2.35))* 1.5 * abs(f / self.__fkt_avg - 1.0)) if (self.__fkt_avg > 0) else 1.06
        # average noise level
        self.__noise_avg = min(max(0.065, 0.9 * self.__noise_avg + 0.1 * noise), 1.06)
        # beta is between 0 and 0.8
        self.__beta = 0.85 - 0.8 * self.__noise_avg
        return self.__beta

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
        if (dd < 1e-80): return 0.0  # to avoid division by zero
        # if(np.isnan(dd) or dd > 1e222):
        #    print("Bad number in cosine function, aborting...", dd); exit(0)
        dt: float = tt.dot(x, y).item()
        if (dd > 1e111):
            print("LibG.cos!", dt, dd, sqrt(dd))  # crash(dd=6e25,in 4=toy)
            return -0.5 if (dt < 0.0) else 0.1
        return dt / sqrt(dd)

    def cosine_fast(self, g_new: tt.Tensor, g_old: tt.Tensor) -> float:  # fewer scalar products
        """
        calculates the cosine of two vectors in any dimension via the well-known
        connection to the scalar product applicable in any dimension
        :param x: first vector on input
        :param y: second vector on input
        :return: cosine value, except for very small vectors where the result is forced
        to be zero
        """
        # float16 tiny(within noise) benefitial (loss+accu), but reducing gradient memory by 50% !!
        # g_new, g_old = g_new.to(dtype=tt.bfloat16), g_old.to(dtype=tt.bfloat16) # test (if float16 is enough, to reduce memory)
        if (self.__norm_yg_old == 0.0):
            self.__norm_yg_old = tt.dot(g_old, g_old).item()  # cache norm (cos)
        norm_new: float = tt.dot(g_new, g_new).item()
        dd: float = norm_new * self.__norm_yg_old  # np.dot(x, x) * np.dot(y, y) # unit = norm
        self.__norm_yg_old = norm_new
        if (dd < 1e-80): return 0.0  # to avoid division by zero
        # if(np.isnan(dd) or dd > 1e222):
        #    print("Bad number in cosine function, aborting...", dd); exit(0)
        dt: float = tt.dot(g_new, g_old).item()
        if (dd > 1e111):
            print("LibG.cos!", dt, dd, sqrt(dd))  # crash(dd=6e25,in 4=toy)
            return -0.5 if (dt < 0.0) else 0.1
        return dt / sqrt(dd)

    def TestCollect2V(self, i1, i2, i3, i4):
        "test: batchsize autodetect"
        a, b = self.TestGrads[i1] + self.TestGrads[i2], self.TestGrads[i3] + self.TestGrads[i4]
        na, nb = a.norm(), b.norm()
        f1, f2 = self.TestFktx[i1] + self.TestFktx[i2], self.TestFktx[i3] + self.TestFktx[i4]
        ppr: float = (a + b).norm() / (na + nb)
        return self.cosine(a, b), ppr, abs(na - nb) / (na + nb), abs(f1 - f2) / (f1 + f2)

    def TestCollect(self, fkt: float, grad_fkt: tt.Tensor):
        "test: batchsize autodetect"
        return  # debug only
        if (fkt > 0.0):
            tc: int = self.TestCountAll
            idx: int = (tc.bit_count() & 1) + 2 * (tc & 1)
            if (self.TestCount[idx] < 1):
                self.TestGrads[idx] = grad_fkt.clone()
            else:
                self.TestGrads[idx] += grad_fkt
            self.TestFktx[idx] += fkt
            self.TestCount[idx] += 1
            self.TestCountAll += 1
        else:
            # for i in range(0, 4): self.TestGrads[i] *= 1.0/self.TestCount
            self.TestCountAll = 0
            self.TestCount = [0] * 4
            self.TestFktx = [0.0] * 4
            return

        tc: int = self.TestCountAll
        if (tc >= 4) and (tc <= 1000) and (0 == (tc % 4)):
            CosLst, PprLst, CstLst, FktLst = [], [], [], []
            cos, ppr, cst, fxr = self.TestCollect2V(0, 1, 2, 3)
            CosLst.append(cos);
            PprLst.append(ppr);
            CstLst.append(cst);
            FktLst.append(fxr)
            cos, ppr, cst, fxr = self.TestCollect2V(0, 2, 1, 3)
            CosLst.append(cos);
            PprLst.append(ppr);
            CstLst.append(cst);
            FktLst.append(fxr)
            cos, ppr, cst, fxr = self.TestCollect2V(0, 3, 1, 2)
            CosLst.append(cos);
            PprLst.append(ppr);
            CstLst.append(cst);
            FktLst.append(fxr)
            cdev, cavg = self.StdDev(CosLst)
            pdev, pavg = self.StdDev(PprLst)
            kdev, kavg = self.StdDev(CstLst)
            fdev, favg = self.StdDev(FktLst)
            mm: int = max(self.TestCount) - min(self.TestCount)
            # print("## e2=%d, c4=%d, mm=%d, CPKF." % (self.coll_inc, tc, mm))
            print("### %d,%d,%d,,%.4f,%.3e,%.4f,%.3e,%.4f,%.3e,%.4f,%.3e" % (
            self.coll_inc, tc, mm, cavg, cdev, pavg, pdev, kavg, kdev, favg, fdev))
        if (tc >= 1000):
            self.TestCollect(-9.9, None)
            exit(0)

        return

    def GradientMerge(self, fkt: float, grad_fkt: tt.Tensor):
        "MicroBatchMerge: average gradients (micro-batch to dynamic mini-batch)"  # ResNet50+, LLM
        if (self.collects < 2):  # or (not self.__init): # optional
            return fkt, 0.0, grad_fkt  # early exit
        hlen: int = len(self.coll_fxl)

        self.coll_cnt += 1
        if (self.step_cnt >= 100): # once per 100 moving steps
            # self.TestCollect(-1.1, grad_fkt)
            if (0 == (self.step_cnt % 100)):  # Tune-Here around 0.01
                s:float = tt.sum(self.coll_cos)#100 * tt.median(self.coll_cos).item() # sum or median()
                s = (1.5 - max(0.0,s))* (1/16.0) if (s<=1.5) else (0.0 if (s<=2.5) else (2.5-min(8.0,s))*(1/16.0))#s = (1.5 - max(0.0,s))/ 16.0 if (s<=1.5) else (0.0 if (s<=2.5) else (2.5-min(5.0,s))/16.0)
                s = (4.0*s+1.0) if (s >= 0.0) else (2.0*s+1.0)
                #s = 4.0*(2.0 - max(0.0,min(4.0,sum(self.coll_cos).item()))) / 16.0 + 1.0
                h:int = 2 if (s > 1.0) else 0
                self.collects = int(self.collects * s) +h
                #self.collects += int(self.collects // 8) + 2 if (
                #            sum(self.coll_cos) < 0.015 * 100 or self.__retrace) else (
                #    1 if (sum(self.coll_cos) < 0.03 * 100) else -int(self.collects * sum(
                #        self.coll_cos) // 48) - 1)  # (16 // self.BatchSizeExt)#int(self.collects // 8) if (sum(self.coll_cos) < 0.1*100) else (16 // self.BatchSizeExt)#self.collects += int(self.collects // 8) if (sum(self.coll_cos) < 0.02*100) else (16 // self.BatchSizeExt)
                self.collects = max(self.MinCollect, min(self.collects, 6250)) #max(2, min(self.collects, 4000 // self.BatchSizeExt))  # effective bs<=2000
                ccd, ccm = self.StdDev(self.coll_cos)  # test
                print("# Increase collects=%d, steps=%d+%d+%d, g=%.3g,a=%.3g, b=%.3g,cs=%.4g+%.3g." %
                      (self.collects, self.coll_cnt, self.step_cnt, self.coll_inc, self.coll_ggc * 0.01, self.alpha,
                       self.__beta, ccm, ccd))
                self.coll_cnt = 0  # self.step_cnt
                self.coll_cos = tt.ones(100, dtype=tt.float32, device='cpu')  # [1.0] * 100
                self.coll_inc += 1
                self.step_cnt += 1  # prevent increase loop
                self.coll_ggc = 0.0
                self.__retrace = False
                self.InternalsSave()

        if (grad_fkt is not None):
            # print("GM:%d, f=%.6f." % (hlen, fkt))
            sel: int = hlen + (hlen >> 1) + (hlen >> 2)  # Python 3.9 has no int.bit_count()
            if (hlen <= 1):
                if (sel & 1):  # parity
                    self.coll_grd = grad_fkt.clone()
                else:
                    self.coll_grd2 = grad_fkt.clone()
            else:
                if (sel & 1):  # parity
                    self.coll_grd += grad_fkt
                else:
                    self.coll_grd2 += grad_fkt
        else:
            self.coll_grd, coll_grd2 = None, None

        # if (-1 == self.coll_inc): # enable tests here
        #     self.TestCollect(fkt, grad_fkt)
        #     return fkt, -9.9, None

        self.coll_fxl.append(float(fkt))
        self.coll_ggl.append(grad_fkt.norm().item())

        if (1 + hlen < self.collects):
            return fkt, -9.9, None  # continue collecting (leave here)

        nrm: float = 1.0 / self.collects
        if (self.coll_grd is not None):
            rv = (self.coll_grd + self.coll_grd2)
            ppr: float = rv.norm().item() / sum(self.coll_ggl)
            ggc: float = self.Contrast(self.coll_grd.norm().item(), self.coll_grd2.norm().item())
            # self.coll_grd *= nrm # average grad
            rv *= nrm
            self.coll_ggc += ggc
            cos = self.cosine(self.coll_grd, self.coll_grd2)
            self.coll_cos[self.step_cnt % 100] = cos
            # self.coll_cos.append(cos) # test
            self.coll_grd, self.coll_grd2 = None, None
        else:
            rv, ggc, ppr, cos = None, -1.0, -1.0, 0.0
            self.coll_grd, self.coll_grd2 = None, None
        # fkt    = sum(self.coll_fxl) * nrm # mean=avg.
        fc_dev, fkt = self.StdDev(self.coll_fxl)
        ggd, ggm = self.StdDev(self.coll_ggl)
        # ccd, ccm = self.StdDev(self.coll_cos)
        # print("CC.stddev=%.4g+%.4g, m/d=%.2f%%." % (ccm, ccd, 100*ccm/ccd), end=" ") # debug
        # print("GG.stddev=%.4g+%.4g, d/m=%.2f%%, ggc=%.3g, ppr=%.3g, cos=%.4f." % (ggm, ggd, 100*ggd/ggm, ggc, ppr, cos)) # debug
        self.coll_fxl = []  # reset list
        self.coll_ggl = []
        self.step_cnt += 1
        return fkt, fc_dev, rv

    def CombiStep(self, fkt: float, grad: tt.Tensor) -> None:
        "special long step (once per epoch)"
        self.__mom = tt.zeros_like(self.__mom)  # later updated with grad
        # print("lgs.CombiStep(f=%.3e)=accepted." % (fkt))
        self.alpha *= 0.1  # todo
        self.__yg *= 0.0  # tt.zeros_like(self.__yg_old)
        self.__norm_yg_old = 0.0
        return

    def c2min_pv_step(self, x: tt.Tensor, fkt: float, grad_fkt: tt.Tensor, fkt_alpha: float = None,
                      combi_step: bool = False):
        """
        performs a single gradient descent step with self-consistent adaption
        of the step length through cos update. This version has no safeguards
        so may be unstable if the function and/or the initial condition is
        pathological especially in terms of potential blow-up. The method is
        designed to ensure local convergence in a bowl-like function landscape
        :param x: independent parameters for the function fkt
        :param fkt: function whose global minimum is to be found
        :param grad_fkt: gradient of the function whose global minimum is to be found
        :combi_step: averaged x for combi_step (once per epoch, if benefitial only)
        :return: updated x-value after the step and statistical function parameter array
        """

        if (x is None):  # "decide if gradient is skipped (revert-steps)"
            return False  # not for C2Min

        if (grad_fkt is not None) and (not isinstance(grad_fkt, tt.Tensor)):
            grad_fkt = tt.tensor(grad_fkt)

        if (combi_step):
            self.CombiStep(fkt, grad_fkt)

        fc_dev: float = -1.0  # <0 = no gradient-average
        if (self.collects > 1):  # <2=off
            fkt, fc_dev, grad_fkt = self.GradientMerge(fkt, grad_fkt)
            if (fc_dev < -1.0):
                return None, [], False  # continue collecting (leave here)
            # if (self.collects > 5):
            #    print("CollectStep:%d, f=%.6f|%.3g, a=%.3e,b=%.3g." % (self.collects, fkt, fc_dev, self.alpha, self.__beta))

        stat_cos2min = []  # statistic only (no effect to algo)
        n: int = len(x)
        fnew = None
        f: float = fkt
        # print(fkt(x),"::",f)
        self.fcalls += 1
        self.jaccalls += 1
        xmem: tt.Tensor = tt.clone(x)

        if (self.__init):  # once (first cycle)
            self.__yg = grad_fkt
            # ++++++++++++++++++++++++++++
            # Gather Function Parameters
            # ++++++++++++++++++++++++++++
            # initialize all global parameters
            self.__mom = tt.clone(self.__yg)  # initialization with gradient for speed-up,
            self.FirstFnktVal(f, len(x), True)
            self.__xbest = [] if (len(x) > 20) else tt.clone(x)  # xbest used only for low dim statistics
            self.__yg_old = tt.clone(self.__yg)
            self.__norm_yg_old = tt.dot(self.__yg_old, self.__yg_old).item()
            # c_InitOptimC2M(0.0); # CFFI
            # ++++++++++++++++++++++++++++
            # adjust alpha initially for stability
            if (self.adjustinit):
                self.alpha = self.__adjust_initial_alpha(
                    x, self.alpha, fkt_alpha, grad_fkt)
            self.solversteps += 1
            self.print(
                "c2min_pv_step[init]:initial alpha adjusted to:", self.alpha)
            if (self.trackHistory):  # False
                self.xhist.append(x)
                self.ahist.append(self.alpha)
                self.fhist.append(f)
            x = SelfConstOptim.single_gradient_descent_step(
                x, self.alpha, self.__yg_old)
            # self._count += 1
            if (n <= 20):
                fnew = fkt
                stat_cos2min.append([self.solversteps, xmem, f,
                                     self.__yg_old, self.alpha, x, fnew])
            if (self.trackHistory):  # False
                if (fnew is None):
                    fnew = fkt
                self.xhist.append(x)
                self.ahist.append(self.alpha)
                self.fhist.append(fnew)
            # self.__calls += 1
            return x, stat_cos2min, False  # self.converged=False(init)
        else:
            # ++++++++++++++++++++++++++++
            # Gather Function Parameters
            self.solversteps += 1
            if (f < 25.0 * self.__truebestfkt):

                # todo: add Cos(G_new, M-G), mdg

                self.__yg_old = tt.clone(self.__yg)
                self.__update_momentum(self.__yg)
                self.__yg = grad_fkt
                # print(fkt(x),"::",f)
                # +++++++++++++++++++++++++++++++++++
                # self-consistent regultation of learning rate alpha
                cos: float = self.cosine_fast(self.__yg, self.__yg_old)
                # self.alpha *= c_alpha_fac(cos) # CFFI
                self.alpha *= self._alpha_fac(cos)  # self/global
                # print("Alpha=%.6f, Beta=%.6f, f=%.6f" % (self.alpha, self.__beta, f))
                # +++++++++++++++++++++++++++++++++++
                # self-consistent regultation of momentum mixing beta
                # self.__beta = c_UpdateBeta(cos) # CFFI
                # self.__beta = self.UpdateBeta(cos) # self
                self.__beta = self.UpdateBetaDF(f)  # new
                # print('f:', f, 'a:', self.alpha, 'b:', self.__beta)
                # +++++++++++++++++++++++++++++++++
                gstat.statist_AddNumbers([sqrt(self.__norm_yg_old), self.alpha, self.__beta])
                # set the inclusion of momentum
                # f1 is coefficient for gradient yg
                f1: float = 1.0 - self.__beta  # 0.3, 0.75 # Freeze beta here !!
                # f2 = 1.0 - f1
                # +++++++++++++++++++++++++++++++++
                # make the actual step
                x = self.single_gradient_descent_step(x, self.alpha,
                                                      tt.add(tt.mul(self.__yg, f1), self.__mom, alpha=1.0 - f1))
                # cgm = self.cosine(yg, self.__mom - self.__yg_old)  # new-test
                # print("+ cos=%.2f, m/g=%.2f,cgm=%.2f, a=%.2e" % (cos, mdg, cgm, self.alpha))
            else:
                # retracing step in case the step implied a function value growing too much
                self.__retrace = True
                # print('retrace')
                # retracing to the previous x
                f1: float = 1.0 - self.__beta
                x = SelfConstOptim.single_gradient_descent_step(x, -self.alpha,
                    tt.add(tt.mul(self.__yg, f1), self.__mom, alpha=1.0 - f1)) #TODO join the x updates to x = x+(alpha_old-alpha_new)G (see paper)
                # computation of new alpha
                self.alpha = self.alpha * (1/3.0)  # *= (1.0 / 10.0) # **self.__fractional_steps
                # computation of new x with new alpha
                x = SelfConstOptim.single_gradient_descent_step(x, self.alpha, tt.add(tt.mul(self.__yg, f1), self.__mom, alpha=1.0 - f1))
            if (f < self.__truebestfkt):  # setting new best function, if actual better
                self.__truebestfkt = f
            else:
                self.__truebestfkt *= 1.1
            # ++++++++++++++++++++++++++++++++++
            # gather statistics
            if (self.trackHistory):
                self.xhist.append(x)
                self.ahist.append(self.alpha)
                self.fhist.append(f)
            self.__last_fkt = f
            if (f < self.__best_fkt):
                self.__best_fkt = f
                if (len(x) <= 20):
                    self.__xbest = tt.clone(x)
            # +++++++++++++++++++++++++++++++++++
            # return statistics
            if (n == 2):  # for SaddlePlots
                fnew = fkt
                print("WALK_C2M: %d %.6f  %.6f %.6f" % (self.solversteps, fnew, x[0], x[1]))
                # if (abs(x[0] + x[1]) > 2.0): exit() # debug
            # stat_cos2min.append([self._count, xmem, f, yg, self.alpha, x, fnew])
            if (n <= 20):  # for Math-Tests
                fnew = fkt
                self.store_optional_best_val(fnew, x)
            else:
                fnew = f  # save time
                xmem = tt.tensor([])  # unused!
            stat_cos2min = (self.solversteps, xmem, tt.tensor([f]), self.__yg, tt.tensor(
                [self.alpha]), x, tt.tensor([fnew]))  # unused in step()
            # self.__calls+=1
            if (fnew > 1e40):  # S. explain: is 1e40 against catastrophic function growth?
                self.signal[2] = True
                self.converged = False
            if (self.solversteps >= self.signal[1]):
                self.signal[2] = True
                self.converged = False
            if (self.__best_fkt <= self.signal[0]):
                self.signal[2] = True
                self.converged = True
            return x, stat_cos2min, self.converged  # (only x used)

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

        Ga: float = tt.dot(yg_old, yg_old).item()
        g: float = self.growth_func()  # also internal update of the growthdamper
        self.__norm_yg_old = Ga  # only for statistic (not algo)
        # g: float = c_growth_func(self.__retrace) # CFFI
        if (self.__retrace):
            # div = f - fold
            m: float = (0.5 * alpha * Ga) / ((f - fold) + alpha * Ga)
            alpha *= m
            # return alpha
        else:
            S: float = tt.dot(yg, yg_old).item()  # das hier ist das skalarprodukt
            h: float = (Ga - S) / Ga
            if (g * h < 1.0):
                alpha *= g
            else:
                if (len(yg) <= 20):
                    alpha /= h
                else:
                    help = (2.35/self.__initfkt)**2.0
                    t = 0.14 if (h<=0) else 0.12
                    alpha /= (h * (1.0 - 0.14 * max(0.0, min(0.7, (help*self.__truebestfkt) ** 0.8))))  # (h*(1.0-0.28*min(0.7,self.__truebest_fkt)))
        return max(min(alpha, 1.0e6), 1.0e-8)  # restricts alpha to 1e-8...1e+6

    def update_best_values(self, x, f: float) -> None:
        if (f < self.__best_fkt):
            self.__best_fkt = f
            if (len(x) <= 20):
                self.__xbest = self.copy_value(x)
        return

    def gen_output(self, x) -> None:
        if (self.trackHistory):
            self.xhist.append(x)
            self.ahist.append(self.alpha)

    def copy_value(self, x) -> tt.Tensor:
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
                if (error < 500000.0):
                    self.__errordamper = error * 16.0
                else:
                    self.__errordamper = 500000.0
        return

    def CalcErrorTolerance(self, truebestfkt: float) -> float:  # CFFI
        # self.__truebestfkt = truebestfkt
        return 25 + (2500.0 + 71.0 * sqrt(truebestfkt)) / (1.0 + sqrt(self.__errordamper))

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

        if (x is None):  # "decide if gradient is skipped (revert-steps)"
            assert (grad_fkt is None)  # test: x=None + grad=None (optional)
            return False  # TODO: future (remove this line to activate)
            if (self.collects > 1) and (len(self.coll_fxl) < self.collects):
                return False  # no skip within collection (gradient sum)
            return (fkt > 1e99)  # default=False (no skip grad) # here f(x)-Bewertung

        if (combi_step):
            self.CombiStep(fkt, grad_fkt)

        if (self.collects > 1):  # <2=off
            fkt, fc_dev, grad_fkt = self.GradientMerge(fkt, grad_fkt)
            if (fc_dev < -1.0):
                return None, [], False

        statp2min = []

        # load function values
        curr_fkt: float = fkt
        assert (grad_fkt is not None)  # TODO: prepare p2min_step() for gradient=None
        if (isinstance(grad_fkt, tt.Tensor)):
            yg = grad_fkt
        else:
            yg = tt.tensor(grad_fkt)
        # curr_fkt = fkt(x)
        self.fcalls += 1
        self.jaccalls += 1
        self.solversteps += 1
        # print("x as loaded:", x)

        if (self.__init):
            self.alpha = 0.00001  # why again?
            self.FirstFnktVal(curr_fkt, len(x), False)
            # if(isinstance(x,tt.Tensor)): self.__xbest = x.clone()
            # else: self.__xbest = x.copy()
            self.__xbest = [] if (len(x) > 20) else self.copy_value(x)
            self.__yg_old = yg
            # c_InitOptimP2M(0.0, 0.0); # CFFI
            if (len(x) <= 20):
                xp = self.copy_value(x)
            self.gen_output(x)

            x = SelfConstOptim.single_gradient_descent_step(x, self.alpha, yg)
            if (len(x) <= 20):
                f = fkt
                self.fhist.append(f)
                self.gen_output(x)
                statp2min = (self.solversteps, xp, tt.tensor([curr_fkt]), yg, tt.tensor(
                    [self.alpha]), x, tt.tensor([f]))
            return x, statp2min, self.converged
        else:
            n: int = len(x)
            self.__best_fkt = curr_fkt if (self.__best_fkt > curr_fkt) else self.__best_fkt
            if (n <= 20):
                xp = self.copy_value(x)
            currfktlesslastfkt: bool = True  # is current function smaller than last function
            errorTolerance: float = self.CalcErrorTolerance(self.__truebestfkt)  # self (CFFI)
            start: bool = True if (n <= 20) else (curr_fkt < 1.1 * self.__initfkt)
            if ((curr_fkt < errorTolerance * 1.1 * self.__truebestfkt) and start):
                # simple step based on the adapted alpha
                if (curr_fkt > self.__last_fkt):
                    currfktlesslastfkt = False
                self.__last_fkt = curr_fkt
                self.__retrace = False
                self.alpha = self.__alpha_opt_version(self.alpha, curr_fkt, self.__last_fkt, yg, self.__yg_old)
                x = SelfConstOptim.single_gradient_descent_step(x, self.alpha, yg)
                self.__yg_old = yg
            else:
                # retracing step in case the step implied a function value growing too much
                self.__retrace = True
                # retracing to the previous x
                x = SelfConstOptim.single_gradient_descent_step(x, -self.alpha, self.__yg_old)
                # computation of new alpha
                self.alpha = self.__alpha_opt_version(self.alpha, curr_fkt, self.__last_fkt, yg, self.__yg_old)
                # computation of new x with new alpha
                x = SelfConstOptim.single_gradient_descent_step(x, self.alpha, self.__yg_old)
            if (curr_fkt < self.__truebestfkt):  # setting new best function, if actual better
                self.__truebestfkt = curr_fkt
            else:
                self.__truebestfkt *= 1.1
            self.UpdateErrorDamper(currfktlesslastfkt)  # self (CFFI)

            gstat.statist_AddNumbers([sqrt(self.__norm_yg_old), self.alpha, self.__growthdamper, self.__errordamper])

            # compute the function value (for output-purposes only)
            f: float = curr_fkt if (n > 20) else fkt
            self.fhist.append(f)
            if (n == 2):  # for SaddlePlots
                print("WALK_P2M: %d %.6f  %.6f %.6f" % (self.solversteps, f, x[0], x[1]))
            # self._count+=1 ; self.__calls+=2
            self.gen_output(x)
            self.update_best_values(x, f)
            if (n <= 20):
                statp2min = (self.solversteps, xp, tt.tensor([curr_fkt]), yg, tt.tensor(
                    [self.alpha]), x, tt.tensor([f]))
            # if (curr_fkt == f): # todo: check reason for this
            ##self.signal[2]=True
            # self.converged=True
            if (self.solversteps >= self.signal[1]):
                self.signal[2] = True
                self.converged = False
            if (self.__best_fkt <= self.signal[0]):
                self.signal[2] = True
                self.converged = True
            return x, statp2min, self.converged

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

    def cos2min_greedy_solver(self, x, fkt, grad_fkt, flev: float = 1.0e-15, maxit: int = 10000, plain_vanilla=True,
                              fixed_beta=True, adjustinit=False):
        # ("USED:cos2min_greedy_solver")
        """
        minimization routine which finds the parameters that minimze a given function fkt given on input
        based on the self-consistent adaptive gradient descent approach. This approach is based on similarity
        of consecutive gradients growing or diminishing the step length according to this similarity
        :param x: initial parameter guess
        :param fkt: function whose global minimum is to be found
        :param grad_fkt: gradient of the function whose global minimum is to be found
        :param flev: target function value to be acquired
        :param maxit: maximum iterations to be tried on the function
        :param plain_vanilla: argument indicating whether the plain-vanilla method without safeguards is to be used
        :param fixed_beta: if plain vanilla is to be used defines whether a fixed relative proportion of moment is used
        :param adjustinit: logical determining whether alpha is initially reset
        :return: minimal parameter values for the given parameter flev or best obtained values in case of failure
        """

        # self.fixed_beta = fixed_beta
        self.adjustinit = adjustinit
        self.signal[0] = flev
        self.signal[1] = maxit
        converged: bool = True

        # nessarry after rework of basic functions to use numpy operands on arrays
        x = tt.tensor(x)

        stats = []

        for i in range(maxit):
            if (plain_vanilla):  # True
                x, finfo, converged = self.c2min_pv_step(x, fkt(x).item(), grad_fkt(x), fkt)
            else:
                x, finfo, converged = self.c2min_greedy_step(x, fkt, grad_fkt)  # broken

            stats.append(finfo)
            if (self.signal[2]):
                break
        # self.print() is disabled!
        if (converged):
            self.print("cos2min_greedy: successful convergence !")
        else:
            self.print("cos2min_greedy: solver did not converge in #=", maxit, "steps!")
        if (len(self.__xbest) <= 20):
            self.print("cos2min_greedy: obtained parameter results:", self.__xbest)
        self.print("cos2min_greedy: obtained optimal functional value:", self.__best_fkt)
        self.print("cos2min_greedy: last alpha value :", self.alpha)
        self.print("cos2min_greedy: # of actual steps made=", self.solversteps)
        self.print("cos2min_greedy: # of function calls=", self.fcalls)
        self.print("cos2min_greedy: # of evaluated jacobians=", self.jaccalls)
        self.print("cos2min_greedy: excess work:", (100 * self.fcalls) / self.solversteps, "%")
        return x, stats

# EoF.
