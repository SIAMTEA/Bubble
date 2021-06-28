#pragma once

#include "VecSoa.cuh"

class Neighbor
{
private:

	double _r_e;
	double _lap_r_e;
	double _cell;

	int _blocksize;

	int _three_dim_block;

	int _max_neighbor;
	int _max_particle;

public:

	Neighbor(const double re, const double lap_re, const int blocksize, const int max_neighbor, const int max_particle);

	virtual ~Neighbor(void);


	void calcNeighborStorageCSR(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1iSoa& type, Vec1dSoa& dens,
		Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
		Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
		Vec1dSoa& csr, Vec1iSoa& csrIndices, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
		unsigned int totalParticle);

};