# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
      name='mcse',
      version='0.0',
      packages=['mcse',
                'mcse/core',
                'mcse/io',
                'mcse/molecules',
                'mcse/crystals',
                'mcse/plot',
                'mcse/libmpi',
                'mcse/workflow',
                'mcse/dimers',
                ],
      install_requires=[
        'scipy>=1.5.0',
        'spglib>=1.9.9.44'
        'matplotlib', 
        'sklearn', 
        'pandas',
        'pymatgen', 
        'pymongo',
        'torch', 
        'vtk',
	"ase @ https://gitlab.com/ase/ase.git", ## Always use newest version of ASE
        "pycifrw"],
      #data_files=[]
      )
