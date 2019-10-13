# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = "Perry"
__version__ = "1.2.2"

setup(name='managpu',
      version=__version__,
      description='managpu: for choosing gpus',
      author=__author__,
      maintainer=__author__,
      url='https://github.com/DogfishBone/managpu',
      packages=find_packages(),
      py_modules=[],
      long_description="Make setting gpus more easy.",
      license="GPLv3",
      platforms=["any"],
      install_requires = ["pynvml>=8.0.1"]
)
