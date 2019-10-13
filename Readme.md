[![Pypi Version](https://img.shields.io/pypi/v/managpu?color=green)](https://pypi.org/project/managpu/)
[![License](https://img.shields.io/pypi/l/managpu)](https://pypi.org/project/managpu/)
# managpu

It is used to choose gpu to run for AI.

## How to install it?
 - Setup by hand:
 
 ```python
python setup.py build
python setup.py install
```
 - Setup by `pip`:
 
```bash
pip install managpu
```

## How to choose gpus?
 - Import the package:
 
 ```python
from managpu import GpuManager
```
 - Create an entry:
 
 ```python
my_gpu = GpuManager(visible_gpus)
```
 - Then choose gpus:
 
 ```python
res = my_gpu.set_by_memory(top_k, limited_gpu_util)
```
