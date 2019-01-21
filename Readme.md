# managpu
Newest version: 1.4.5

    It is used to choose gpu to run for AI.

## How to install it?
 - Download it, and then run
 ```python
python setup.py build
python setup.py install
```
 - Use pip by running code
```bash
pip install managpu
```

## How to choose gpus?
 - Import the package
 ```python
import managpu as mg
```
 - Create an entry by
 ```python
my_gpu = mg.GpuManager(visible_gpus)
```
 - Then run
 ```python
res = my_gpu.set_by_memory(top_k)
```
to choose your gpus

## How to set framework?
>For example, with a Keras or Tensorflow framework.
 - Import the package
 ```python
from managpu.tensorflow_tool import set_tensorflow_config
```
 - Then in your codes, you can write
 ```python
config = set_tensorflow_config(fraction, is_auto_increase)
```
to set the fraction and auto_increase for the framework.
>As for pytorch
 - Import the package
 ```python
from managpu.pytorch_tool import get_device
```
 - Write as follows to choose your device
 ```python
device = get_device(index, is_gpu)
```