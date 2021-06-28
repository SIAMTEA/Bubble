#pragma once

#include "VecSoa.cuh"

namespace gradient
{

	__constant__ double _r_e;
	__constant__ double _r_min;
	__constant__ double _density;
	__constant__ double _air_density;  //6/21
	__constant__ double _dt;

	__constant__ int _fluid;
	__constant__ int _wall;
	__constant__ int _dwall;
	__constant__ int _air;  //air
	__constant__ int _ghost;

	__constant__ int _max_neighbor;

}

class PressureGradient
{

private:

	double _r_e;
	double _r_min;
	double _density;
	double _air_density;  //6/21
	double _dt;
	int _fluid;
	int _wall;
	int _dwall;
	int _air;  //6/21
	int _ghost;

	int _max_neighbor;


public:

	PressureGradient(const double re, const double r_min, const double density, const double air_density, const double d_t,  //air_density 6/21
					 const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/21
					 const int max_neighbor);

	virtual ~PressureGradient(void);

	void calcPressureGradient(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
							  Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
							  Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
							  Vec1iSoa& type, Vec1dSoa& press, Vec1dSoa& minpress,
							  Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
							  hVec1dSoa& StandardParams, unsigned int totalParticle);

	void HighOrdercalcPressureGradient(
		Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
		Vec1iSoa& type, Vec1dSoa& press, Vec1dSoa& minpress,
		Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
		hVec1dSoa& StandardParams, unsigned int totalParticle);


};