# -*- coding: utf-8 -*-

from managpu import GpuManager

if __name__ == '__main__':
    my_gpu = GpuManager()
    free_gpu = my_gpu.set_by_memory(3)
    print(free_gpu)