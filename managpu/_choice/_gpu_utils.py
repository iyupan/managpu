# -*- coding: utf-8 -*-

import os
import abc


class Gpu_Base(metaclass=abc.ABCMeta):
    def __init__(self, visible_gpu_indexes: list=None):
        self.gpu_num = self.get_gpu_amout()

        if visible_gpu_indexes == None:
            self.visible_gpu_indexes = [i for i in range(self.gpu_num)]
        else:
            # Type check
            assert isinstance(visible_gpu_indexes, list), "The type of visible_gpu_indexes is not list."
            for v in visible_gpu_indexes:
                assert isinstance(v, int), "The index of gpus is not an integer."

            # Range check
            max_usr_gpu_index = max(visible_gpu_indexes)
            assert max_usr_gpu_index < self.gpu_num, "The index setted as %d is out of range %d" % \
                                                     (max_usr_gpu_index, self.gpu_num)

            # Overlap check
            index_length = len(visible_gpu_indexes)
            if index_length > len(set(visible_gpu_indexes)):
                raise ValueError("There are the same index in the visible tuple!")
            self.visible_gpu_indexes = visible_gpu_indexes

        self.visible_gpu_num = len(self.visible_gpu_indexes)

    def sort_visible_by_memory(self):
        sorted_all_gpus = self.sort_all_by_memory()
        sorted_visible_index = []
        for v in sorted_all_gpus:
            if v in self.visible_gpu_indexes:
                sorted_visible_index.append(v)
        assert len(sorted_visible_index) == self.visible_gpu_num, "Sort Error."
        return sorted_visible_index

    def set_by_memory(self, top_k):
        to_set_indexes = []
        sorted_visible_gpus = self.sort_visible_by_memory()
        assert top_k <= self.visible_gpu_num, "Setted num %d is out of range %d." % (top_k, self.visible_gpu_num)
        for i in range(top_k):
            to_set_indexes.append(sorted_visible_gpus[i])
        self.set_specified_gpu(to_set_indexes)
        return to_set_indexes

    """
        Which needed rewrite are as follows.
    """
    @abc.abstractmethod
    def get_gpu_amout(self):
        pass

    @abc.abstractmethod
    def sort_all_by_memory(self):
        pass

    @abc.abstractmethod
    def set_specified_gpu(self, gpu_indexes: list):
        pass


class Linux_Gpu(Gpu_Base):
    def __init__(self, visible_gpu_indexes: list=None):
        super(Linux_Gpu, self).__init__(visible_gpu_indexes=visible_gpu_indexes)

    def get_gpu_amout(self):
        CMD_get_gpu_num = 'nvidia-smi -L | wc -l'
        gpu_num = int(os.popen(CMD_get_gpu_num).read())
        return gpu_num

    def sort_all_by_memory(self):
        CMD1 = 'nvidia-smi| grep MiB | grep -v Default | cut -c 4-8'
        CMD3 = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'

        # first choose the free gpus
        gpu_usage = set(map(lambda x: int(x), os.popen(CMD1).read().split()))
        free_gpus = set(range(self.gpu_num)) - gpu_usage

        # then choose the most memory free gpus
        gpu_free_mem = list(map(lambda x: int(x), os.popen(CMD3).read().split()))
        gpu_sorted = list(sorted(range(self.gpu_num), key=lambda x: gpu_free_mem[x], reverse=True))[
                     len(free_gpus):]
        sorted_index = list(free_gpus) + list(gpu_sorted)
        return sorted_index

    def set_specified_gpu(self, gpu_indexes: list):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_indexes))
