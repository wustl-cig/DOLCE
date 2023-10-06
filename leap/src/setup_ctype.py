################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
################################################################################

from setuptools import setup, Extension, Command
import setuptools.command.develop
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.build
import subprocess
import sys
import os


class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)


setup(name = 'LEAP',
    version = '0.1',
    description='',
    cmdclass = {'install': install},
    packages=['LEAP'],
    package_dir={'LEAP': 'LEAP_cuda'},
    package_data={
        'LEAP': [
            'main_projector.so',
            'main_projector.py',
            'main_projector.h',
        ]
    },
    include_package_data=True
);
