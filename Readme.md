[![Pypi Version](https://img.shields.io/pypi/v/managpu?color=green)](https://pypi.org/project/managpu/)

# managpu

It is used to choose gpu to run for AI.

## How to install it?
 - Download it, and then run
 
 ```python
python setup.py build
python setup.py install
```
 - Use `pip` by running command
 
```bash
pip install managpu
```

## How to choose gpus?
 - Import the package
 
 ```python
from managpu as GpuManager
```
 - Create an entry by
 
 ```python
my_gpu = GpuManager(visible_gpus)
```
 - To choose gpus, run
 
 ```python
res = my_gpu.set_by_memory(top_k, limited_gpu_util)
```
