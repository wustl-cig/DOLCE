#!/bin/bash
source ~/.bashrc
conda env create -f environment.yml
conda activate dolce
cd ./leap/src
pip uninstall leapct
pip install .
cd ..
cd ..