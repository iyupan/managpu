# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = "FishBone"
__version__ = "1.0.0"

setup(name='managpu',
      version=__version__,
      description='managpu: for choosing gpus',
      author=__author__,
      maintainer=__author__,
      url='https://github.com/DogfishBone/GpuManager',
      packages=find_packages(),
      py_modules=[],
      long_description="Make setting gpus more easy.",
      license="Public",
      platforms=["any"],
)
