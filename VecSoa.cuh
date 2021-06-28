#pragma once
#pragma warning(disable : 4503)
#pragma warning(disable : 4995)

#include <cuda_runtime.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "Vec1Soa.cuh"
#include "common.cuh"

typedef Vec1Soa<bool>			Vec1bSoa;
typedef Vec1Soa<int>			Vec1iSoa;
typedef Vec1Soa<unsigned int>	Vec1uiSoa;
typedef Vec1Soa<float>			Vec1fSoa;
typedef Vec1Soa<double>			Vec1dSoa;

typedef hVec1Soa<bool>			hVec1bSoa;
typedef hVec1Soa<int>			hVec1iSoa;
typedef hVec1Soa<unsigned int>  hVec1uiSoa;
typedef hVec1Soa<float>			hVec1fSoa;
typedef hVec1Soa<double>		hVec1dSoa;

//device_Position
static Vec1dSoa Pos_x;
static Vec1dSoa Pos_y;
static Vec1dSoa Pos_z;

//device_Velocity
static Vec1dSoa Vel_x;
static Vec1dSoa Vel_y;
static Vec1dSoa Vel_z;

//device_Acc
static Vec1dSoa Acc_x;
static Vec1dSoa Acc_y;
static Vec1dSoa Acc_z;

//device_Dens
static Vec1dSoa Dens;

//device_Type
static Vec1iSoa Type;

//device_Press
static Vec1dSoa Press;
static Vec1dSoa minPress;

//device_neighbor
static Vec1uiSoa NeighborIndex;
static Vec1uiSoa NeighborNum;
static Vec1uiSoa lapNeighborIndex;
static Vec1uiSoa lapNeighborNum;

//device_Boundary
static Vec1iSoa Boundary;

static Vec1dSoa csr;
static Vec1iSoa csrIndices;
static Vec1iSoa csrNum;
static Vec1iSoa csrOffsets;


//host_Pos
static hVec1dSoa hPos_x;
static hVec1dSoa hPos_y;
static hVec1dSoa hPos_z;

//host_Vel
static hVec1dSoa hVel_x;
static hVec1dSoa hVel_y;
static hVec1dSoa hVel_z;

//host_Acc
static hVec1dSoa hAcc_x;
static hVec1dSoa hAcc_y;
static hVec1dSoa hAcc_z;

//host_Dens
static hVec1dSoa hDens;

//host_Type
static hVec1iSoa hType;

//host_Press
static hVec1dSoa hPress;
static hVec1dSoa hminPress;

//host_Boundary
static hVec1iSoa hBoundary;

//params
static hVec1dSoa StandardParams;


