# -*- coding: utf-8 -*-
# print(1)
# from pynvml.smi import nvidia_smi
# print(2)
# nvsmi = nvidia_smi.getInstance()
# print(3)
# a = nvsmi.DeviceQuery("utilization.memory, utilization.gpu, memory.free, memory.total")
# print(4)

from managpu import GpuManager

if __name__ == '__main__':
    my_gpu = GpuManager()
    free_gpu = my_gpu.set_by_memory(3)
    print(free_gpu)