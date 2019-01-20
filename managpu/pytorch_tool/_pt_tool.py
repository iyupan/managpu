# -*- coding: utf-8 -*-

import torch

def get_device(index=0, is_gpu=True):
    assert isinstance(index, int), "index must be int, but this is %s" % type(index)
    assert isinstance(is_gpu, bool), "index must be bool, but this is %s" % type(is_gpu)
    if is_gpu:
        if torch.cuda.is_available():
            gpu_num = torch._C._cuda_getDeviceCount()
            assert index < gpu_num, "There are %d gpus. The index %d is too large!" % (gpu_num, index)
            print("Obtain cuda successfully!")
            device_type = "cuda:%d" % index
        else:
            print("Obtain cuda unsuccessfully!")
            device_type = "cpu"
    else:
        device_type = "cpu"
    print("Device Type: ", device_type)
    return torch.device(device_type)
