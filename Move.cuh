#pragma once

#include "VecSoa.cuh"

namespace move
{

	__constant__ double _r_min;
	__constant__ double _dt;

	__constant__ int _fluid;
	__constant__ int _air;  //6/21
	__constant__ int _viscos_fluid;


}

class Move
{

private:

	double _r_min;
	double _dt;

	int _fluid;
	int _air;  //6/21

public:

	Move(const double r_min, const double d_t, const int fluid, const int air);  //air 6/21

	virtual ~Move(void);

	void UpdataParams(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
					  Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
					  Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
					  Vec1iSoa& type,
					  unsigned int totalParticle);

	void ModifyParams(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
					  Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
					  Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
					  Vec1iSoa& type,
					  unsigned int totalParticle);

	void HighOrderUpdataParams(
		Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
		Vec1iSoa& type,
		unsigned int totalParticle);

	void HighOrderModifyParams(
		Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
		Vec1iSoa& type,
		unsigned int totalParticle);

};