# -*- coding: utf-8 -*-

# Author: Perry
# @Create Time: 2019/6/4 14:01

__all__ = ["GpuManager"]

from managpu._choice._gpu_utils import GPULinux
import platform


class GpuManager(object):
    def __init__(self, visible_gpus: list=None):
        # Type check
        assert isinstance(visible_gpus, list) or visible_gpus is None, "The visible_gpus should be a list or None"
        self.__SUPPORT_OS = ["Linux"]

        self.os_name = platform.system()
        if self.os_name == "Linux":
            self.gpu = GPULinux(visible_gpus)

    def set_by_memory(self, top_k, util_limit=None):
        if self.os_name in self.__SUPPORT_OS:
            best_gpus = self.gpu.set_by_memory(top_k, util_limit)
        else:
            best_gpus = None
            print("All the gpus will be used, because for the system of %s is not supported!" % self.os_name)
        return best_gpus

    def set_specified_gpu(self, gpus: list):
        if self.os_name in self.__SUPPORT_OS:
            self.gpu.set_specified_gpu(gpus)
        else:
            print("All the gpus will be used, because for the system of %s is not supported!" % self.os_name)
