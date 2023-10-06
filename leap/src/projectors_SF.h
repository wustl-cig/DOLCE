////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////

#ifndef __PROJECTORS_SF_H
#define __PROJECTORS_SF_H

#ifdef WIN32
#pragma once
#endif

#include "parameters.h"

bool project_SF_cone(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_SF_cone(float*, float*&, parameters*, bool cpu_to_gpu);

bool project_SF_parallel(float*&, float*, parameters*, bool cpu_to_gpu);
bool backproject_SF_parallel(float*, float*&, parameters*, bool cpu_to_gpu);

#endif
