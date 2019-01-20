# -*- coding: utf-8 -*-

def set_tensorflow_config(fraction: float=None, is_auto_increase: bool=True):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = is_auto_increase
    if fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = fraction
    return config