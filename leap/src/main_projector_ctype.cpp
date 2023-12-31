////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////

#include "main_projector_ctype.h"
#include "parameters.h"
#include "projectors.h"
#include "projectors_SF.h"
#include "projectors_cpu.h"
#include <stdlib.h>
#include <stdio.h>

//#ifdef DEFINE_STATIC_UI
parameters params;

bool printParameters()
{
	params.printAll();
	return true;
}

bool reset()
{
	params.clearAll();
	params.setDefaults(1);
	return true;
}

bool project(float* g, float* f, bool cpu_to_gpu)
{
	if (params.allDefined() == false || g == NULL || f == NULL)
	{
		printf("ERROR: project: invalid parameters or invalid input arrays!\n");
		return false;
	}
	else if (params.whichGPU >= 0)
	{
		if (params.geometry == parameters::CONE)
        {
            if (params.useSF())
                return project_SF_cone(g, f, &params, cpu_to_gpu);
            else
                return project_cone(g, f, &params, cpu_to_gpu);
        }
		else if (params.geometry == parameters::PARALLEL)
        {
            if (params.useSF())
                return project_SF_parallel(g, f, &params, cpu_to_gpu);
            else
                return project_parallel(g, f, &params, cpu_to_gpu);
        }
		else
			return project_modular(g, f, &params, cpu_to_gpu);
	}
	else
	{
		if (params.geometry == parameters::CONE)
		{
			if (params.useSF())
				return CPUproject_SF_cone(g, f, &params);
			else
				return CPUproject_cone(g, f, &params);
		}
		else if (params.geometry == parameters::PARALLEL)
		{
			if (params.useSF())
				return CPUproject_SF_parallel(g, f, &params);
			else
				return CPUproject_parallel(g, f, &params);
		}
		else
			return CPUproject_modular(g, f, &params);
	}
}

bool backproject(float* g, float* f, bool cpu_to_gpu)
{
	if (params.allDefined() == false || g == NULL || f == NULL)
		return false;
	else if (params.whichGPU >= 0)
	{
		if (params.geometry == parameters::CONE)
        {
            if (params.useSF())
                return backproject_SF_cone(g, f, &params, cpu_to_gpu);
            else
                return backproject_cone(g, f, &params, cpu_to_gpu);
        }
		else if (params.geometry == parameters::PARALLEL)
        {
            if (params.useSF())
                return backproject_SF_parallel(g, f, &params, cpu_to_gpu);
            else
                return backproject_parallel(g, f, &params, cpu_to_gpu);
        }
		else
			return backproject_modular(g, f, &params, cpu_to_gpu);
	}
	else
	{
		if (params.geometry == parameters::CONE)
		{
			if (params.useSF())
				return CPUbackproject_SF_cone(g, f, &params);
			else
				return CPUbackproject_cone(g, f, &params);
		}
		else if (params.geometry == parameters::PARALLEL)
		{
			if (params.useSF())
				return CPUbackproject_SF_parallel(g, f, &params);
			else
				return CPUbackproject_parallel(g, f, &params);
		}
		else
			return CPUbackproject_modular(g, f, &params);
	}
}

bool setConeBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd)
{
	params.geometry = parameters::CONE;
	params.detectorType = parameters::FLAT;
	params.sod = sod;
	params.sdd = sdd;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = centerCol;
	params.centerRow = centerRow;
	params.setAngles(phis, numAngles);
	return params.geometryDefined();
}

bool setParallelBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis)
{
	params.geometry = parameters::PARALLEL;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.centerCol = centerCol;
	params.centerRow = centerRow;
	params.setAngles(phis, numAngles);
	return params.geometryDefined();
}

bool setModularBeamParams(int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in)
{
	params.geometry = parameters::MODULAR;
	params.pixelWidth = pixelWidth;
	params.pixelHeight = pixelHeight;
	params.numCols = numCols;
	params.numRows = numRows;
	params.numAngles = numAngles;
	params.setSourcesAndModules(sourcePositions_in, moduleCenters_in, rowVectors_in, colVectors_in, numAngles);
	return params.geometryDefined();
}

bool setVolumeParams(int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	params.numX = numX;
	params.numY = numY;
	params.numZ = numZ;
	params.voxelWidth = voxelWidth;
	params.voxelHeight = voxelHeight;
	params.offsetX = offsetX;
	params.offsetY = offsetY;
	params.offsetZ = offsetZ;
	return params.volumeDefined();
}

bool setGPU(int whichGPU)
{
	params.whichGPU = whichGPU;
	return true;
}

bool setProjector(int which)
{
    if (which == parameters::SEPARABLE_FOOTPRINT)
        params.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        params.whichProjector = 0;
    return true;
}

bool set_axisOfSymmetry(float axisOfSymmetry)
{
	params.axisOfSymmetry = axisOfSymmetry;
	return true;
}

bool set_rFOV(float rFOV_in)
{
	params.rFOVspecified = rFOV_in;
	return true;
}

bool projectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::CONE;
	tempParams.detectorType = parameters::FLAT;
	tempParams.sod = sod;
	tempParams.sdd = sdd;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return project_SF_cone(g, f, &tempParams, cpu_to_gpu);
        else
            return project_cone(g, f, &tempParams, cpu_to_gpu);
    }
}

bool backprojectConeBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::CONE;
	tempParams.detectorType = parameters::FLAT;
	tempParams.sod = sod;
	tempParams.sdd = sdd;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return backproject_SF_cone(g, f, &tempParams, cpu_to_gpu);
        else
            return backproject_cone(g, f, &tempParams, cpu_to_gpu);
    }
}

bool projectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::PARALLEL;
	tempParams.detectorType = parameters::FLAT;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return project_SF_parallel(g, f, &tempParams, cpu_to_gpu);
        else
            return project_parallel(g, f, &tempParams, cpu_to_gpu);
    }
}

bool backprojectParallelBeam(float* g, float* f, bool cpu_to_gpu, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, float* phis, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
	parameters tempParams;
	tempParams.geometry = parameters::PARALLEL;
	tempParams.detectorType = parameters::FLAT;
	tempParams.pixelWidth = pixelWidth;
	tempParams.pixelHeight = pixelHeight;
	tempParams.numCols = numCols;
	tempParams.numRows = numRows;
	tempParams.numAngles = numAngles;
	tempParams.centerCol = centerCol;
	tempParams.centerRow = centerRow;
	tempParams.setAngles(phis, numAngles);

	tempParams.numX = numX;
	tempParams.numY = numY;
	tempParams.numZ = numZ;
	tempParams.voxelWidth = voxelWidth;
	tempParams.voxelHeight = voxelHeight;
	tempParams.offsetX = offsetX;
	tempParams.offsetY = offsetY;
	tempParams.offsetZ = offsetZ;

	if (tempParams.allDefined() == false || g == NULL || f == NULL)
		return false;
	else
    {
        if (params.useSF())
            return backproject_SF_parallel(g, f, &tempParams, cpu_to_gpu);
        else
            return backproject_parallel(g, f, &tempParams, cpu_to_gpu);
    }
}


/*
// Scanner Parameters
int geometry;
int detectorType;
float sod, sdd;
float pixelWidth, pixelHeight, angularRange;
int numCols, numRows, numAngles;
float centerCol, centerRow;
float* phis;

// Volume Parameters
int volumeDimensionOrder;
int numX, numY, numZ;
float voxelWidth, voxelHeight;
float offsetX, offsetY, offsetZ;
//*/

