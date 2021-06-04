
import numpy as np
import random
import torch
import torch.nn as nn

def init_random_seed(rand_seed):
    # torch.backends.cudnn.deterministic = True
    # numpy random seed
    np.random.seed(rand_seed)
    # if we are using random
    random.seed(rand_seed)
    # torch random seed
    torch.manual_seed(rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rand_seed)

def gettime():
    import datetime
    import time
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

def pretty_print(ddict):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ddict)