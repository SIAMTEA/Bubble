#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "VecSoa.cuh"
#include "Neighbor.cuh"
#include "common.cuh"


class Particle
{
public:

	//device_getPos
	Vec1dSoa& getPositions_x()			{ return Pos_x; }
	Vec1dSoa& getPositions_y()			{ return Pos_y; }
	Vec1dSoa& getPositions_z()			{ return Pos_z; }
	
	//device_getVel
	Vec1dSoa& getVelocity_x()			{ return Vel_x; }
	Vec1dSoa& getVelocity_y()			{ return Vel_y; }
	Vec1dSoa& getVelocity_z()			{ return Vel_z; }

	//device_getAcc
	Vec1dSoa& getAcceleration_x()		{ return Acc_x; }
	Vec1dSoa& getAcceleration_y()		{ return Acc_y; }
	Vec1dSoa& getAcceleration_z()		{ return Acc_z; }

	//device_getPressure
	Vec1dSoa& getPressure()				{ return Press; }
	Vec1dSoa& getminPressure()			{ return minPress; }

	//device_getDensity
	Vec1dSoa& getDensity()				{ return Dens; }

	//device_getParticleType
	Vec1iSoa& getParticleType()			{ return Type; }

	//device_getBoundary
	Vec1iSoa& getBoundary()				{ return Boundary; }

	//device_getNeighbor
	Vec1uiSoa& getNeighborIndex()		{ return NeighborIndex; }
	Vec1uiSoa& getNeighborNum()			{ return NeighborNum; }
	Vec1uiSoa& getlapNeighborIndex()	{ return lapNeighborIndex; }
	Vec1uiSoa& getlapNeighborNum()		{ return lapNeighborNum; }

	//device_getCSR
	Vec1dSoa& getCSRMatrix()			{ return csr; }
	Vec1iSoa& getCSRIndices()			{ return csrIndices; }
	Vec1iSoa& getCSRNum()				{ return csrNum; }
	Vec1iSoa& getCSROffsets()			{ return csrOffsets; }

	//host_getPos
	hVec1dSoa& gethPositions_x()		{ return hPos_x; }
	hVec1dSoa& gethPositions_y()		{ return hPos_y; }
	hVec1dSoa& gethPositions_z()		{ return hPos_z; }

	//host_getVel
	hVec1dSoa& gethVelocity_x()			{ return hVel_x; }
	hVec1dSoa& gethVelocity_y()			{ return hVel_y; }
	hVec1dSoa& gethVelocity_z()			{ return hVel_z; }

	//host_getAcc
	hVec1dSoa& gethAcceleration_x()		{ return hAcc_x; }
	hVec1dSoa& gethAcceleration_y()		{ return hAcc_y; }
	hVec1dSoa& gethAcceleration_z()		{ return hAcc_z; }

	//host_getPressure
	hVec1dSoa& gethPressure()			{ return hPress; }
	hVec1dSoa& gethminPressure()		{ return hminPress; }

	//host_getDensity
	hVec1dSoa& gethDensity()			{ return hDens; }

	//host_getType
	hVec1iSoa& gethParticleType()		{ return hType; }

	//host_getBoundary
	hVec1iSoa& gethBoundary()			{ return hBoundary; }

	unsigned int getSize()				{ return hPos_x.size(); }

	hVec1dSoa& getStandardParams()		{ return StandardParams; }

  

private:

    double _r_e;
    double _lap_r_e;
    double _fpd;
    int _fluid;
    int _wall;
    int _dwall;
	int _air;  //6/17
    int _viscos_fluid;
    int _max_neighbor;


public:

	Particle(const double re, const double lap_re, const double fpd,
		     const int fluid, const int wall, const int dwall, const int air, const int max_neighbor);  //air 6/17

    virtual ~Particle(void);

	void InitParticlePosition(hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z);


	void InitParticleType_and_FirstVel(hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z,
                                       hVec1dSoa& hvel_x, hVec1dSoa& hvel_y, hVec1dSoa& hvel_z,
                                       hVec1iSoa& Type, const int totalParticle);


	void UpdataResize(
			Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
			Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
			Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
			Vec1dSoa& press, Vec1dSoa& minpress,
			Vec1dSoa& dens, Vec1iSoa& type,
			Vec1iSoa& boundary,
			Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
			Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
			hVec1dSoa& hpos_x, hVec1dSoa& hpos_y, hVec1dSoa& hpos_z,
			hVec1dSoa& hvel_x, hVec1dSoa& hvel_y, hVec1dSoa& hvel_z,
			hVec1dSoa& hacc_x, hVec1dSoa& hacc_y, hVec1dSoa& hacc_z,
			hVec1dSoa& hpress, hVec1dSoa& hminpress,
			hVec1dSoa& hdens, hVec1iSoa& htype,
			hVec1iSoa& hboundary,
			unsigned int totalParticle);

	void calcInitStandardParams(hVec1dSoa& Stan);

	void PotentialCoef(hVec1dSoa& Stan);

	void Convert(
			Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
			Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
			Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
			Vec1dSoa& press, Vec1dSoa& minpress,
			Vec1dSoa& dens, Vec1iSoa& type,
			Vec1iSoa& boundary,
			hVec1dSoa& hpos_x, hVec1dSoa& hpos_y, hVec1dSoa& hpos_z,
			hVec1dSoa& hvel_x, hVec1dSoa& hvel_y, hVec1dSoa& hvel_z,
			hVec1dSoa& hacc_x, hVec1dSoa& hacc_y, hVec1dSoa& hacc_z,
			hVec1dSoa& hpress, hVec1dSoa& hminpress,
			hVec1dSoa& hdens, hVec1iSoa& htype,
			hVec1iSoa& hboundary, 
			unsigned int totalParticle);


};