# -*- coding: utf-8 -*-

# Author: Perry
# @Create Time: 2019/6/4 14:56

from managpu import GpuManager


if __name__ == '__main__':
    my_gpu = GpuManager()
    my_gpu.set_by_memory(1, 5)
