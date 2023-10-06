# LivermorE AI Projector for Computed Tomography (LEAP)
This is a source copy from the official LivermorE AI Projector for Computed Tomography (LEAP), A differentiable forward and backward projectors for AI/ML-driven computed tomography applications. For most updated version, https://github.com/LLNL/LEAP/tree/main.

The current version of LEAP is tested and fully compatible with [DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction, ICCV 2023](https://github.com/wustl-cig/DOLCE).

## Installation
To install LEAP package, use pip command: 

$ pip install .    

It is strongly recommended to run "pip uninstall leapct" if you have installed the previous version.  

## CPP-CUDA Library

This is c++ with CUDA library that can be used in both C++ and python with the provided wrapper class. To compile the code using cmake, use cmake prefix such that  

$ cmake -DCMAKE_CXX_COMPILER=/usr/local/gcc63/bin/g++63   -DCMAKE_CUDA_HOST_COMPILER=/usr/local/gcc63/bin/g++63   -DCMAKE_CUDA_ARCHITECTURES=75 ..  
$ cmake --build . --config Release  

Note that cuda compiler (nvcc) does not support gcc higher than 7.5.   

## Python-binding
In addition to our provided python library using pybind11, you can make a separate ctype python library using setup_ctype.py. Rename it to setup.py, and then run:  

$ python setup.py install  

Note that this binding option provides cpu-to-gpu copy option only, i.e., numpy array data as input and output (f, g) and they will be moved to GPU memory internally  


## Source code list
* src/CMakeLists.txt: CMake for GPU ctype projector  
* src/main_projector_ctype.cpp, .h: main code for ctype binding   
* src/main_projector.cpp: main code for pybind11  
* src/parameters.h .cpp: projector parameters used in main_projector and projectors  
* src/projectors_cpu.cpp: CPU projector (forward and backproject) for multiple scanner geometry types   
* src/projectors.cu: GPU projector (forward and backproject) for multiple scanner geometry types  
* src/leapctype.py: python wrapper class for standard ctype package  
* src/leaptorch.py: python wrapper class for pytorch nn.module package  
* setup.py: setup.py for torch projector  
* setup_ctype.py: setup.py for ctype projector  


## Resource
Information about python-c++ binding: https://realpython.com/python-bindings-overview/  
https://pytorch.org/tutorials/advanced/cpp_extension.html  


## Authors
Hyojin Kim (hkim@llnl.gov)  
Kyle Champley (champley1@llnl.gov)  


## Other Contributors
Jiaming Liu (jiaming.liu@wustl.edu) for reconstruction sample code  


## Citing LEAP

Please cite the following paper if you need to reference LEAP in your publications:  
Hyojin Kim and Kyle Champley, Differentiable Forward Projector for X-ray Computed Tomography, Arxiv, 2023


## License
LEAP is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.  
See [LICENSE](LICENSE) for more details.  
SPDX-License-Identifier: MIT  
LLNL-CODE-848657  
