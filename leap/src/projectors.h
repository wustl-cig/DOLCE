////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_H
#define __PROJECTORS_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_cone(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_cone(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_parallel(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_parallel(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_modular(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_modular(float*, float*&, parameters*, bool cpu_to_gpu);

// Utility Functions for pushing/pulling data to/from CPU/GPU
float* copyProjectionDataToGPU(float* g, parameters* params, int whichGPU);
bool pullProjectionDataFromGPU(float* g, parameters* params, float* dev_g, int whichGPU);
float* copyVolumeDataToGPU(float* f, parameters* params, int whichGPU);
bool pullVolumeDataFromGPU(float* f, parameters* params, float* dev_f, int whichGPU);

/* Utility Functions for anti-symmetric projections
float* splitVolume(float*, parameters* params, bool rightHalf = true);
float* splitProjection(float*, parameters* params, bool rightHalf = true);
bool mergeSplitVolume(float*, float*, parameters* params, bool rightHalf = true);
bool mergeSplitProjection(float*, float*, parameters* params, bool rightHalf = true);
//*/

#endif
