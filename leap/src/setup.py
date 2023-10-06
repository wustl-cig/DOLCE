################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
################################################################################

import sysconfig
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11


setup(
    name='LEAP',
    author='Hyojin Kim, Kyle Champley',
    author_email='hkim@llnl.gov',
    description='Livermore AI/ML projector for computed tomography',
    python_requires='>=3.6',
    py_modules = ['LEAP_torch'],
    ext_modules=[
        CUDAExtension(
            name='LEAP',
		    sources=['main_projector.cpp', 'projectors_cpu.cpp', 'projectors.cu', 'projectors_SF.cu', 'parameters.cpp'],
            extra_compile_args={'cxx': ['-g', '-D__USE_GPU', '-I%s' % pybind11.get_include()], 
                                'nvcc': ['-g', '-D__USE_GPU']}
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

