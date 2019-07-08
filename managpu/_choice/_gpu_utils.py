# -*- coding: utf-8 -*-

# Author: Perry
# @Create Time: 2019/6/4 14:01

import os
import abc


class GPUState(object):
    def __init__(self, free: int, util: int, index: int):
        self.gpu_index = index
        self.freemem = free
        self.gpu_util = util


class GPUMachine(object):
    def __init__(self, gpu_num: int):
        self.gpu_num = gpu_num
        self.gpu_states = []

    def add_gpu_state(self, gpu_state: GPUState):
        self.gpu_states.append(gpu_state)


class GPUInfo(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_gpu_machine(self) -> GPUMachine:
        pass


class GPUControl(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_specified_gpu(self, gpu_indexes: list):
        pass


class GPUBase(GPUInfo, GPUControl):
    def __init__(self, visible_gpu_indexes: list = None):
        self.gpu_machine = self.get_gpu_machine()

        if visible_gpu_indexes == None:
            self.visible_gpu_indexes = [i for i in range(self.gpu_machine.gpu_num)]
        else:
            # Type check
            assert isinstance(visible_gpu_indexes, list), "The type of visible_gpu_indexes is not list."
            for v in visible_gpu_indexes:
                assert isinstance(v, int), "The index of gpus is not an integer."

            # Range check
            max_usr_gpu_index = max(visible_gpu_indexes)
            assert max_usr_gpu_index < self.gpu_machine.gpu_num, "The index setted as %d is out of range %d" % \
                                                     (max_usr_gpu_index, self.gpu_machine.gpu_num)

            # Overlap check
            index_length = len(visible_gpu_indexes)
            if index_length > len(set(visible_gpu_indexes)):
                raise ValueError("There are the same index in the visible tuple!")
            self.visible_gpu_indexes = visible_gpu_indexes

        self.visible_gpu_num = len(self.visible_gpu_indexes)

    def set_by_memory(self, top_k, util_limit=None):
        """

        :param top_k: memory size of Top 3
        :param util_limit: m in [0, 100] means m%
        :return:
        """
        to_set_indexes = []
        sorted_visible_gpus = self.__sort_visible_by_memory()
        assert top_k <= self.visible_gpu_num, "Setted num %d is out of range %d." % (top_k, self.visible_gpu_num)
        if util_limit is None:
            print("No GPU Util Limit!")
            for i in range(top_k):
                to_set_indexes.append(sorted_visible_gpus[i].gpu_index)
        else:
            assert isinstance(util_limit, int), "Parameter util_limit should be a int."
            assert 0 <= util_limit and util_limit <= 100, "Parameter util_limit should be set in [0, 100]."
            print("GPU Util limits to %d%%" % util_limit)
            chosen_gpu_amount = 0
            for i in range(self.visible_gpu_num):
                if sorted_visible_gpus[i].gpu_util <= util_limit:
                    to_set_indexes.append(sorted_visible_gpus[i].gpu_index)
                    chosen_gpu_amount += 1
                    if chosen_gpu_amount == top_k:
                        break

        print("Sorted by memory:")
        for one_gpu in sorted_visible_gpus:
            # print("GPU Index: %d" % one_gpu.gpu_index + "\t" + "GPU FreeMemory: %d MB" % one_gpu.freemem + "\t" +
            #       "GPU Util: %d%%" % one_gpu.gpu_util)
            print("    {0:<18} {1:<30} {2:<16}".format("GPU Index: %d" % one_gpu.gpu_index, "GPU FreeMemory: %d MB" \
                                                   % one_gpu.freemem, "GPU Util: %d%%" % one_gpu.gpu_util))
        # "GPU Index: %d" % one_gpu.gpu_index, "GPU FreeMemory: %d MB" % one_gpu.freemem, "GPU Util: %d%%" % one_gpu.gpu_util
        print("Qualified GPU Index is:", to_set_indexes)

        assert len(to_set_indexes) == top_k, "The gpu of gpu_util <= %d is %d, not equal top_k %d." \
                                             % (util_limit, len(to_set_indexes), top_k)
        self.set_specified_gpu(to_set_indexes)
        return to_set_indexes

    def __sort_visible_by_memory(self):
        sorted_all_gpus = self.__sort_all_by_memory()
        def is_visible_gpu(gpu):
            return gpu.gpu_index in self.visible_gpu_indexes
        sorted_visible_gpus = list(filter(is_visible_gpu, sorted_all_gpus))
        assert len(sorted_visible_gpus) == self.visible_gpu_num, "Sort Error."
        return sorted_visible_gpus

    def __sort_all_by_memory(self) -> list:
        gpu_sorted = list(sorted(self.gpu_machine.gpu_states, key=lambda x: x.__dict__["freemem"], reverse=True))
        return gpu_sorted


class GPULinux(GPUBase):
    def __init__(self, visible_gpu_indexes: list=None):
        super(GPULinux, self).__init__(visible_gpu_indexes)

    def get_gpu_machine(self) -> GPUMachine:
        from pynvml.smi import nvidia_smi
        nvsmi = nvidia_smi.getInstance()
        gpu_info = nvsmi.DeviceQuery('index, utilization.gpu, memory.free, count')
        gpu_machine = GPUMachine(gpu_info["count"])
        for one_gpu in gpu_info["gpu"]:
            gpu_machine.add_gpu_state(
                GPUState(free=one_gpu["fb_memory_usage"]["free"],
                         util=one_gpu["utilization"]["gpu_util"],
                         index=int(one_gpu["minor_number"])
                         )
            )
        return gpu_machine

    def set_specified_gpu(self, gpu_indexes: list):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_indexes))
