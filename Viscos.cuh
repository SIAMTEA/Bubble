#pragma once

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "VecSoa.cuh"
#include "common.cuh"

namespace viscos
{

	__constant__ double _r_e;
	__constant__ double _lap_r_e;
	__constant__ double _density;
	__constant__ double _air_density;  //6/21
	__constant__ double _dt;

	__constant__ double3 _gf;

	__constant__ double _kinematic_viscosity_coef;
	__constant__ double _air_kinematic_viscosity_coef;  //6/21

	__constant__ double _viscosity_coef;
	__constant__ double _air_viscosity_coef;  //6/21

	__constant__ int _fluid;
	__constant__ int _wall;
	__constant__ int _dwall;
	__constant__ int _air;  //6/17
	__constant__ int _ghost;

	__constant__ int _max_neighbor;
}

class Viscos
{

private:

	double _r_e;
	double _lap_r_e;
	double _density;
	double _air_density;  //6/21
	double _dt;

	int _fluid;
	int _wall;
	int _dwall;
	int _air;  //6/17
	int _ghost;

	double3 _gf;
	double _gy;

	int _dim;

	double _kinematic_viscosity_coef;
	double _air_kinematic_viscosity_coef;  //6/21

	double _viscosity_coef;
	double _air_viscosity_coef;  //6/21

	int _max_neighbor;


public:

	Viscos(const double re, const double lap_re, const double density, const double air_density, const double d_t,  //air_density 6/21
		   const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/17
		   const double gy, const double kvc, const double a_kvc,  //a_kvc 6/21
		   const double vc, const double a_vc, const int dim,  //a_vc 6/21
		   const int max_neighbor);

	virtual ~Viscos(void);

	void calcViscos(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
		Vec1iSoa& type, 
		Vec1uiSoa& lapNeighborIndex, Vec1uiSoa& lapNeighborNum,
		hVec1dSoa& StandardParams, unsigned int totalParticle);

	/*void calcImplicitViscosCSR(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
		Vec1iSoa& type,
		Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
		Vec1iSoa& csrIndices, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
		hVec1dSoa& StandardParams, unsigned int totalParticle);*/
	

};