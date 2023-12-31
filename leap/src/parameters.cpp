////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "parameters.h"

using namespace std;

parameters::parameters()
{
	sourcePositions = NULL;
	moduleCenters = NULL;
	rowVectors = NULL;
	colVectors = NULL;
	phis = NULL;
	setDefaults(1);
}

parameters::parameters(int N)
{
	sourcePositions = NULL;
	moduleCenters = NULL;
	rowVectors = NULL;
	colVectors = NULL;
	phis = NULL;
	setDefaults(N);
}

parameters::parameters(const parameters& other)
{
    sourcePositions = NULL;
    moduleCenters = NULL;
    rowVectors = NULL;
    colVectors = NULL;
    phis = NULL;
    setDefaults(1);
    assign(other);
}

parameters::~parameters()
{
    clearAll();
}

parameters& parameters::operator = (const parameters& other)
{
    if (this != &other) {
        this->assign(other);
    }
    return *this;
}

void parameters::assign(const parameters& other)
{
    this->clearAll();
    
    this->whichGPU = other.whichGPU;
    this->whichProjector = other.whichProjector;
    this->geometry = other.geometry;
    this->detectorType = other.detectorType;
    this->sod = other.sod;
    this->sdd = other.sdd;
    this->pixelWidth = other.pixelWidth;
    this->pixelHeight = other.pixelHeight;
    this->angularRange = other.angularRange;
    this->numCols = other.numCols;
    this->numRows = other.numRows;
    this->numAngles = other.numAngles;
    this->centerCol = other.centerCol;
    this->centerRow = other.centerRow;
    this->tau = other.tau;
    this->rFOVspecified = other.rFOVspecified;
    this->axisOfSymmetry = other.axisOfSymmetry;
    this->volumeDimensionOrder = other.volumeDimensionOrder;
    this->numX = other.numX;
    this->numY = other.numY;
    this->numZ = other.numZ;
    this->voxelWidth = other.voxelWidth;
    this->voxelHeight = other.voxelHeight;
    this->offsetX = other.offsetX;
    this->offsetY = other.offsetY;
    this->offsetZ = other.offsetZ;

    if (this->phis != NULL)
        delete [] this->phis;
    this->phis = new float[numAngles];
    for (int i = 0; i < numAngles; i++)
        this->phis[i] = other.phis[i];
    
    this->setSourcesAndModules(other.sourcePositions, other.moduleCenters, \
        other.rowVectors, other.colVectors, other.numAngles);
        
}

void parameters::setDefaults(int N)
{
	whichGPU = 0;
    whichProjector = SEPARABLE_FOOTPRINT;

	geometry = CONE;
	detectorType = FLAT;
	sod = 1100.0;
	sdd = 1400.0;
	numCols = 2048 / N;
	numRows = numCols;
	numAngles = int(ceil(1440.0*float(numCols) / 2048.0));
	pixelWidth = 0.2*2048.0 / float(numCols);
	pixelHeight = pixelWidth;
	angularRange = 360.0;
	centerCol = float(numCols - 1) / 2.0;
	centerRow = float(numCols - 1) / 2.0;
	tau = 0.0;
	rFOVspecified = 0.0;

	axisOfSymmetry = 90.0;

    setAngles(); // added by Hyojin
	setDefaultVolumeParameters();
}

float parameters::T_phi()
{
    if (numAngles <= 1 || phis == NULL)
        return 2.0*PI;
    else
        return phis[1]-phis[0];
}

float parameters::rFOV()
{
	if (rFOVspecified > 0.0)
		return rFOVspecified;
    else if (geometry == MODULAR)
        return 1.0e16;
    else if (geometry == PARALLEL)
    {
        return min(fabs(u_0()), fabs(pixelWidth*float(numCols-1) + u_0()));
    }
    else
    {
        /*
        double alpha_right = lateral(0);
        double alpha_left = lateral(N_lateral-1);
        if (isFlatPanel == true)
        {
            alpha_right = atan(alpha_right);
            alpha_left = atan(alpha_left);
        }
        //return R_tau*sin(min(fabs(alpha_right+atan(tau/R)), fabs(alpha_left+atan(tau/R))));
        double retVal = R_tau*sin(min(fabs(alpha_right-atan(tau/R)), fabs(alpha_left-atan(tau/R))));

        //sid/sdd * c / sqrt(1+(c/sdd)*(c/sdd))
        //return R*u_max()/sqrt(1.0+u_max()*u_max());
        if (theSCT->dxfov.unknown == false && theSCT->dxfov.value > 0.0)
            retVal = min(retVal, 0.5*theSCT->dxfov.value);
        return retVal;
        //*/
        
        float alpha_right = u_0();
        float alpha_left = pixelWidth*float(numCols-1) + u_0();
        alpha_right = atan(alpha_right/sdd);
        alpha_left = atan(alpha_left/sdd);
        return sod*sin(min(fabs(alpha_right),fabs(alpha_left)));
    }
}

bool parameters::useSF()
{
    if (whichProjector == SIDDON || geometry == MODULAR || isSymmetric() == true)
        return false;
    else
    {
        if (geometry == CONE)
        {
            float largestDetectorWidth = sdd/(sod-rFOV())*pixelWidth;
            float smallestDetectorWidth = sdd/(sod+rFOV())*pixelWidth;
            
            float largestDetectorHeight = sdd/(sod-rFOV())*pixelHeight;
            float smallestDetectorHeight = sdd/(sod+rFOV())*pixelHeight;
			if (0.5*largestDetectorWidth <= voxelWidth && voxelWidth <= 2.0*smallestDetectorWidth && 0.5*largestDetectorHeight <= voxelHeight && voxelHeight <= 2.0*smallestDetectorHeight)
			{
				//printf("using SF projector\n");
				return true;
			}
			else
			{
				//printf("using Siddon projector\n");
				return false;
			}
        }
        else //if (geometry == PARALLEL)
        {
            if (0.5*pixelWidth <= voxelWidth && voxelWidth <= 2.0*pixelWidth && 0.5*pixelHeight <= voxelHeight && voxelHeight <= 2.0*pixelHeight)
                return true;
            else
                return false;
        }
    }
}

bool parameters::isSymmetric()
{
	if (numAngles == 1 && fabs(axisOfSymmetry) <= 30.0)
		return true;
	else
		return false;
}

bool parameters::allDefined()
{
	return geometryDefined() & volumeDefined();
}

bool parameters::geometryDefined()
{
	if (geometry != CONE && geometry != PARALLEL && geometry != MODULAR)
		return false;
	if (numCols <= 0 || numRows <= 0 || numAngles <= 0 || pixelWidth <= 0.0 || pixelHeight <= 0.0)
		return false;
	if (geometry == MODULAR)
	{
		if (sourcePositions == NULL || moduleCenters == NULL || rowVectors == NULL || colVectors == NULL)
			return false;
	}
	else if (angularRange == 0.0 && phis == NULL)
		return false;
	if (geometry == CONE)
	{
		if (sod <= 0.0 || sdd <= sod)
			return false;
	}

	return true;
}

bool parameters::volumeDefined()
{
	if (numX <= 0 || numY <= 0 || numZ <= 0 || voxelWidth <= 0.0 || voxelHeight <= 0.0 || volumeDimensionOrder < 0 || volumeDimensionOrder > 1)
		return false;
	else
	{
		if (geometry == PARALLEL)
		{
			if (voxelHeight != pixelHeight)
			{
				voxelHeight = pixelHeight;
				printf("Warning: for parallel-beam data volume voxel height must equal detector pixel height, so forcing voxel height to match pixel height!\n");
			}
			offsetZ = floor(0.5 + offsetZ / voxelHeight) * voxelHeight;
		}
		if (geometry == MODULAR && voxelWidth != voxelHeight)
		{
			voxelHeight = voxelWidth;
			printf("Warning: for modular-beam data volume voxel height must equal voxel width (voxels must be cubic), so forcing voxel height to match voxel width!\n");
		}
		if (isSymmetric())
		{
			if (numX > 1)
			{
				printf("Error: symmetric objects must specify numX = 1!\n");
				return false;
			}
			if (numY % 2 == 1)
			{
				printf("Error: symmetric objects must specify numY as even !\n");
				return false;
			}
			offsetX = 0.0;
			offsetY = 0.0;
		}
		return true;
	}
}

bool parameters::setDefaultVolumeParameters()
{
	if (geometryDefined() == false)
		return false;

	// Volume Parameters
	volumeDimensionOrder = XYZ;
	numX = numCols;
	numY = numX;
	numZ = numRows;
	voxelWidth = sod / sdd * pixelWidth;
	voxelHeight = sod / sdd * pixelHeight;
	offsetX = 0.0;
	offsetY = 0.0;
	offsetZ = 0.0;

	if (isSymmetric())
	{
		numX = 1;
		offsetX = 0.0;

		offsetY = 0.0;
		if (numY % 2 == 1)
			numY += 1;
	}

	return true;
}

void parameters::clearAll()
{
	if (phis != NULL)
		delete[] phis;
	phis = NULL;

	clearModularBeamParameters();
}

bool parameters::clearModularBeamParameters()
{
	if (sourcePositions != NULL)
		delete[] sourcePositions;
	sourcePositions = NULL;
	if (moduleCenters != NULL)
		delete[] moduleCenters;
	moduleCenters = NULL;
	if (rowVectors != NULL)
		delete[] rowVectors;
	rowVectors = NULL;
	if (colVectors != NULL)
		delete[] colVectors;
	colVectors = NULL;
	return true;
}

void parameters::printAll()
{
	printf("\n");

	if (geometry == CONE)
		printf("======== CT Cone-Beam Geometry ========\n");
	else if (geometry == PARALLEL)
		printf("======== CT Parallel-Beam Geometry ========\n");
	else
		printf("======== CT Modular-Beam Geometry ========\n");
	printf("number of angles: %d\n", numAngles);
	printf("number of detector elements: %d x %d\n", numRows, numCols);
	if (phis != NULL && numAngles >= 2)
		printf("angular range: %f degrees\n", 180.0 / PI * (phis[numAngles - 1] - phis[0] + (phis[numAngles - 1] - phis[numAngles - 2])));
	printf("detector pixel size: %f mm x %f mm\n", pixelHeight, pixelWidth);
	printf("center detector pixel: %f, %f\n", centerRow, centerCol);
	if (geometry == CONE)
	{
		printf("sod = %f mm\n", sod);
		printf("sdd = %f mm\n", sdd);
	}
	printf("\n");

	printf("======== CT Volume ========\n");
	printf("number of voxels: %d x %d x %d\n", numX, numY, numZ);
	printf("voxel size: %f mm x %f mm x %f mm\n", voxelWidth, voxelWidth, voxelHeight);
	if (offsetX != 0.0 || offsetY != 0.0 || offsetZ != 0.0)
		printf("volume offset: %f mm, %f mm, %f mm\n", offsetX, offsetY, offsetZ);
	if (isSymmetric())
		printf("axis of symmetry = %f degrees\n", axisOfSymmetry);
	//printf("x_0 = %f, y_0 = %f, z_0 = %f\n", x_0(), y_0(), z_0());

	printf("\n");
}

bool parameters::setToZero(float* data, int N)
{
	if (data != NULL && N > 0)
	{
		for (int i = 0; i < N; i++)
			data[i] = 0.0;
		return true;
	}
	else
		return false;
}

bool parameters::windowFOV(float* f)
{
	if (f == NULL)
		return false;
	else
	{
		float rFOVsq = rFOV()*rFOV();
		for (int ix = 0; ix < numX; ix++)
		{
			float x = ix * voxelWidth + x_0();
			for (int iy = 0; iy < numY; iy++)
			{
				float y = iy * voxelWidth + y_0();
				if (x*x + y * y > rFOVsq)
				{
					float* zLine = &f[ix*numY*numZ + iy*numZ];
					for (int iz = 0; iz < numZ; iz++)
						zLine[iz] = 0.0;
				}
			}
		}
		return true;
	}
}

bool parameters::setAngles()
{
	if (phis != NULL)
		delete[] phis;
	phis = NULL;
	if (numAngles <= 0 || angularRange == 0.0)
		return false;
	else
	{
		phis = new float[numAngles];
		for (int i = 0; i < numAngles; i++)
			phis[i] = float(i)*angularRange*(PI / 180.0) / float(numAngles) - 0.5*PI;
		return true;
	}
}

bool parameters::setAngles(float* phis_new, int numAngles_new)
{
	if (phis != NULL)
		delete[] phis;
	phis = NULL;
	if (phis_new == NULL || numAngles_new <= 0)
		return false;
	else
	{
		numAngles = numAngles_new;
		phis = new float[numAngles];
		for (int i = 0; i < numAngles; i++)
			phis[i] = phis_new[i] * PI / 180.0 - 0.5*PI;
		return true;
	}
}

bool parameters::setSourcesAndModules(float* sourcePositions_in, float* moduleCenters_in, float* rowVectors_in, float* colVectors_in, int numPairs)
{
	clearModularBeamParameters();
	if (sourcePositions_in == NULL || moduleCenters_in == NULL || rowVectors_in == NULL || colVectors_in == NULL || numPairs <= 0)
		return false;
	else
	{
		numAngles = numPairs;
		sourcePositions = new float[3 * numPairs];
		moduleCenters = new float[3 * numPairs];
		rowVectors = new float[3 * numPairs];
		colVectors = new float[3 * numPairs];
		for (int i = 0; i < 3 * numPairs; i++)
		{
			sourcePositions[i] = sourcePositions_in[i];
			moduleCenters[i] = moduleCenters_in[i];
			rowVectors[i] = rowVectors_in[i];
			colVectors[i] = colVectors_in[i];
		}

		return true;
	}
}

float parameters::u_0()
{
	return -centerCol * pixelWidth;
}

float parameters::v_0()
{
	return -centerRow * pixelHeight;
}

float parameters::x_0()
{
	return offsetX - 0.5*float(numX - 1)*voxelWidth;
}

float parameters::y_0()
{
	return offsetY - 0.5*float(numY - 1)*voxelWidth;
}

float parameters::z_0()
{
	//return offsetZ - 0.5*float(numZ - 1)*voxelHeight;
	if (geometry == PARALLEL)
		return offsetZ - centerRow * (pixelHeight / voxelHeight) * voxelHeight;
	else if (geometry == MODULAR)
		return offsetZ - 0.5*float(numZ-1) * voxelHeight;
	else
		return offsetZ - centerRow * ((sod / sdd * pixelHeight) / voxelHeight) * voxelHeight;
}
