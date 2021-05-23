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
                ],
      install_requires=[
        'numpy>=1.20.1', 
        'scipy>=1.5.0',
        'spglib>=1.9.9.44'
        'mpi4py', 'h5py', 'matplotlib', 'sklearn', 
        'pandas','pymatgen', 'pymongo',
        'torch', 'numba',
        'vtk',
	"ase @ https://gitlab.com/ase/tarball/master", ## Always use newest version of ASE
        "pycifrw"],
      #data_files=[]
      )
