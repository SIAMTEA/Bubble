#pragma once

#include "VecSoa.cuh"
#include "common.cuh"


namespace potential
{


    __constant__ double _lap_r_e;
    __constant__ double _r_min;
	__constant__ double _density;
    __constant__ double _dt;

    __constant__ int _fluid;
    __constant__ int _wall;
    __constant__ int _dwall;
	__constant__ int _air;  //6/21
    __constant__ int _ghost;


    __constant__ double _surface_coef;

    __constant__ int _max_neighbor;



}

class Potential{
private:


    double _lap_r_e;
    double _r_min;
    double _density;
    double _dt;

	int _fluid;
    int _wall;
    int _dwall;
	int _air;  //6/21
    int _ghost;

    double _surface_coef;

    int _max_neighbor;


public:

    Potential(const double lap_re, const double r_min, const double density, const double d_t,
              const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/21
              const double surface_coef, const int max_neighbor);

    virtual ~Potential(void);


    void calcPotential(
            Vec1dSoa &pos_x, Vec1dSoa &pos_y, Vec1dSoa &pos_z,
            Vec1dSoa &vel_x, Vec1dSoa &vel_y, Vec1dSoa &vel_z,
            Vec1dSoa &acc_x, Vec1dSoa &acc_y, Vec1dSoa &acc_z,
            Vec1iSoa &type,
            Vec1uiSoa &lapneighborIndex, Vec1uiSoa &lapneighborNum,
            hVec1dSoa &StandardParams, unsigned int totalParticle);

};

