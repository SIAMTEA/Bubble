#pragma once

#include "VecSoa.cuh"

namespace collision
{

	__constant__ double _r_e;
	__constant__ double _coll_limit;
	__constant__ double _density;
	__constant__ double _air_density;  //6/21
	__constant__ double _dt;

	__constant__ int _fluid;
	__constant__ int _wall;
	__constant__ int _dwall;
	__constant__ int _air;  //6/21
	__constant__ int _ghost;

	__constant__ int _max_neighbor;

	__constant__ double _col_rate;

}

class Collision
{

private:

	double _r_e;
	double _coll_limit;
	double _density;
	double _air_density;  //6/21
	double _dt;

	int _fluid;
	int _wall;
	int _dwall;
	int _air;  //6/21
	int _ghost;

	int _max_neighbor;
	double _col_rate;

public:

	Collision(const double re, const double coll_limit, const double density, const double air_density, const double d_t,  //air_density 6/21
		      const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/21
			  const int max_neighbor, const double col_rate);

	virtual ~Collision(void);

	void calcCollisionTerm(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
		Vec1iSoa& type,
		Vec1uiSoa& NeighborIndex, Vec1uiSoa& NeighborNum,
		unsigned int totalParticle);

	void HighOrdercalcCollisionTerm(
		Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
		Vec1iSoa& type,
		Vec1uiSoa& NeighborIndex, Vec1uiSoa& NeighborNum,
		unsigned int totalParticle);

};