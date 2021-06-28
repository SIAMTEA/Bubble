#pragma once

#include "VecSoa.cuh"
#include "common.cuh"

namespace csr_press
{

	__constant__ double _r_e;
	__constant__ double _lap_r_e;
	__constant__ double _r_min;
	__constant__ double _density;
	__constant__ double _air_density;  //6/21
	__constant__ double _dt;

	__constant__ int _fluid;
	__constant__ int _wall;
	__constant__ int _dwall;
	__constant__ int _air;  //6/21
	__constant__ int _ghost;

	__constant__ int _externalparticle;
	__constant__ int _innerparticle;
	__constant__ int _surfaceparticle;

	__constant__ double _surface_coef;

	__constant__ int _dim;

	__constant__ double _pi;

	__constant__ int _max_neighbor;

	__constant__ double _dirichle;
	__constant__ double _relax_coe;
	__constant__ double _compress;

	__constant__ int _max_iter;
	__constant__ double _error;

}

class PressureTerm
{
private:

	double _r_e;
	double _lap_r_e;
	double _r_min;
	double _density;
	double _air_density;  //6/21
	double _dt;
	int _fluid;
	int _wall;
	int _dwall;
	int _air;  //6/21
	int _ghost;
	int _externalparticle;
	int _innerparticle;
	int _surfaceparticle;
	int _dim;
	int _max_neighbor;
	double _dirichle;
	double _relax_coe;
	double _compress;
	int _max_iter;
	double _error;

	cusparseHandle_t _cusparse;
	cusparseMatDescr_t _matDescr;

public:

	PressureTerm(const double re, const double lap_re, const double r_min, const double density, const double air_density, const double d_t,  //air_density 6/21
				 const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/21
				 const int externalparticle, const int innerparticle, const int surfaceparticle, const int dim, const int max_neighbor,
				 const double dirichle, const double relax_coe, const double compress, const int max_iter, const double error,
				 cusparseHandle_t cusparse, cusparseMatDescr_t matDescr);

	virtual ~PressureTerm(void);


	void calcCSRPressureTerm(
		Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1iSoa& type,
		Vec1dSoa& press, Vec1dSoa& minpress, Vec1dSoa& dens,
		Vec1iSoa& boundary,
		Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
		Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
		Vec1dSoa& csr, Vec1iSoa& csrIndices, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
		hVec1dSoa& StandardParams, unsigned int totalParticle);

	void HighOrdercalcCSRPressureTerm(
		Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
		Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
		Vec1iSoa& type,
		Vec1dSoa& press, Vec1dSoa& minpress, Vec1dSoa& dens,
		Vec1iSoa& boundary,
		Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
		Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
		Vec1dSoa& csr, Vec1iSoa& csrIndices, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
		hVec1dSoa& StandardParams, unsigned int totalParticle);

};