# Statistic-Helper (2023)

# import math
# import torch
# import statistic_helper as gstat

class GlobalString: # StaticClass:
    # Class-level (static) attributes
    attr_str: str = ""
    count: int = 0
    # attribute2 = "Value 2"

    # Class-level (static) method to change attributes
    @staticmethod
    def add_string(s : str) -> None:
        if (0 == GlobalString.count):
            GlobalString.attr_str = s
        else:
            GlobalString.attr_str += "," + s
        GlobalString.count += 1
        # GlobalString.attribute1 = new_value1

    # Class-level (static) method to print attributes
    #@classmethod (cls)
    @staticmethod
    def get_strings() -> str:
        # print(f"attribute1: {cls.attribute1}")
        GlobalString.count = 0
        return GlobalString.attr_str
        
def StatListStr(lst: list[float]) -> str:
    n: int = len(lst)
    if (n < 2):
        return ("(len=%d<2!)" % n)

    prv: float = lst[0]
    sum_ad: float = 0.0
    for i in range(1, n):
        val : float = lst[i]
        sum_ad += abs(val - prv)
        prv = val
    val = -9.9 if (0.0==sum_ad) else (lst[-1]-lst[0]) /sum_ad
    # print(n,min(lst),sum(lst)/n,max(lst),val); exit(0)
    return ("(:%.3g<%.4g<%.3g:%.3g)" % (min(lst),sum(lst)/n,max(lst),val))
       
class GlobalStatist: # StaticClass:
    num_elem: int = 0
    count: int = 0 # entries
    lists = []
    ppr_vectP = []
    ppr_vect0 = []
    ppr_away:float = 0.0
    ppr_step : int = 0
    
    @staticmethod
    def statist_Init():
        GlobalStatist.count = 0
        GlobalStatist.lists = []
        GlobalStatist.ppr_vectP = []
        GlobalStatist.ppr_vect0 = []
        GlobalStatist.ppr_away = 0.0
        GlobalStatist.ppr_step = 0
    
    @staticmethod
    def statist_AddNumbers(lst: list[float]) -> None:
        if (len(lst) < 1) or (len(lst) > 99): return
        if (GlobalStatist.num_elem < 1) or (GlobalStatist.count < 1):
            GlobalStatist.num_elem = len(lst)
        if (GlobalStatist.num_elem != len(lst)):
            print("GlobalStatist/MisMatch: %d<>%d !" % (GlobalStatist.num_elem, len(lst)))
            return
        if len(GlobalStatist.lists) != len(lst):
            GlobalStatist.lists = []
            for e in lst:
                GlobalStatist.lists.append( [e] )
            GlobalStatist.count = 1
            return
        for i in range(len(lst)):
            GlobalStatist.lists[i].append(lst[i])
        GlobalStatist.count += 1
        return

    @staticmethod
    def statist_UpdateVector(vec): # expencive (2x vector memory)
        if (vec is None) or (len(vec) < 2):
            return
        if (len(vec) != len(GlobalStatist.ppr_vect)) or (GlobalStatist.ppr_step == 0):
            GlobalStatist.ppr_away = 0.0
            GlobalStatist.ppr_step = 0
            GlobalStatist.ppr_vectP = vec
            GlobalStatist.ppr_vect0 = vec
            return
        if (GlobalStatist.ppr_step == 0):
            GlobalStatist.ppr_step += 1
            GlobalStatist.ppr_vectP = vec
            return
        GlobalStatist.ppr_away += torch.dist(GlobalStatist.ppr_vectP, vec)
        GlobalStatist.ppr_vectP = vec # ToDo: calc
        # vn:float = torch.norm(GlobalStatist.ppr_vect0 - GlobalStatist.ppr_vectP)
        # return vn / GlobalStatist.ppr_away # ppr
        return

    @staticmethod
    def statist_GetStr() -> str:
        if (GlobalStatist.count < 2) or (GlobalStatist.num_elem < 1): # len(GlobalStatist.lists)
            return ("LowData(%d,%d)!" % (GlobalStatist.count, GlobalStatist.num_elem))
        s: str = StatListStr( GlobalStatist.lists[0] )
        for i in range(1, len(GlobalStatist.lists)):
            s += "," + StatListStr( GlobalStatist.lists[i] )
        GlobalStatist.count = 0 # clean/reset
        GlobalStatist.lists = []
        return s

# Accessing and printing initial attribute values
# StaticClass.print_attributes()

# Using the static method to change attribute values
# StaticClass.add_attributes("New Value 1", "New Value 2")

# Accessing and printing updated attribute values
# StaticClass.print_attributes() 
