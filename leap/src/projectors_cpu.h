////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_CPU_H
#define __PROJECTORS_CPU_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool CPUproject_cone(float*, float*, parameters*);
bool CPUbackproject_cone(float*, float*, parameters*);

bool CPUproject_parallel(float*, float*, parameters*);
bool CPUbackproject_parallel(float*, float*, parameters*);

bool CPUproject_modular(float*, float*, parameters*);
bool CPUbackproject_modular(float*, float*, parameters*);

float projectLine(float* f, parameters* params, float* pos, float* traj);

inline float tex3D(float* f, int, int, int, parameters* params);

bool CPUproject_AbelCone(float*, float*, parameters*);
bool CPUbackproject_AbelCone(float*, float*, parameters*);

bool CPUproject_AbelParallel(float*, float*, parameters*);
bool CPUbackproject_AbelParallel(float*, float*, parameters*);

bool CPUproject_SF_parallel(float*, float*, parameters*);
bool CPUbackproject_SF_parallel(float*, float*, parameters*);

bool CPUproject_SF_cone(float*, float*, parameters*);
bool CPUbackproject_SF_cone(float*, float*, parameters*);

bool CPUproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi);
bool CPUbackproject_SF_cone_kernel(float* aProj, float* xSlice, parameters* params, int ix, int iphi);

bool applyPolarWeight(float* g, parameters* params);
bool applyInversePolarWeight(float* g, parameters* params);

#endif
