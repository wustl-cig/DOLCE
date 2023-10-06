################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# DOLCE project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# DOLCE: A Model-Based Probabilistic Diffusion Framework for Limited-Angle CT Reconstruction
################################################################################

from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided_diffusion"],
    install_requires=["tqdm"],
)
