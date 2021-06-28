#pragma once

#include "VecSoa.cuh"
#include "common.cuh"

namespace density
{

	__constant__ double _r_e;
	__constant__ int _max_neighbor;
	__constant__ int _ghost;

}

class ParticleDensity
{

private:

	double _r_e;
	int _max_neighbor;
	int _ghost;


public:

	ParticleDensity(const double re, const int max_neighbor, const int ghost);

	virtual ~ParticleDensity(void);

	void calcParticleDensity(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1iSoa& type, Vec1dSoa& Dens,
		Vec1uiSoa& NeighborIndex, Vec1uiSoa& NeighborNum,
		unsigned int totalParticle);

};