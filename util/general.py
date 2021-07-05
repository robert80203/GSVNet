
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

def load_model(model, model_file, is_restore=False):
    import time
    t_start = time.time()

    device = torch.device('cpu')
    state_dict = torch.load(model_file, map_location=device)
    t_ioend = time.time()

    model.load_state_dict(state_dict, strict = False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        print('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        print('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    print(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))
    return model

def gettime():
    import datetime
    import time
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

def pretty_print(ddict):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ddict)