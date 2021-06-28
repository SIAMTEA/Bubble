
#pragma once

#include "VecSoa.cuh"
#include "common.cuh"

namespace surfaceTens
{

    __constant__ double _r_e;
    __constant__ double _lap_r_e;
    __constant__ double _r_min;
    __constant__ double _density_f;
    __constant__ double _density_v;
    __constant__ double _dt;

    __constant__ int _fluid;
    __constant__ int _viscos_fluid;
    __constant__ int _wall;
    __constant__ int _dwall;
    __constant__ int _ghost;

    __constant__ int _externalparticle;
    __constant__ int _innerparticle;
    __constant__ int _surfaceparticle;

    __constant__ double _surface_coef;

    __constant__ double _pi;

    __constant__ int _max_neighbor;

    __constant__ double _dirichle;

    __constant__ int _catheter_on;
    __constant__ int _catheter_out;

}

class SurfaceTension
{

private:

    double _r_e;
    double _lap_r_e;
    double _r_min;
    double _density_f;
    double _density_v;
    double _dt;

    int _fluid;
    int _viscos_fluid;
    int _wall;
    int _dwall;
    int _ghost;
    int _externalparticle;
    int _innerparticle;
    int _surfaceparticle;

    int _max_neighbor;
    double _surface_coef;
    double _pi;
    double _dirichle;

    int _catheter_on;
    int _catheter_out;

public:

    SurfaceTension(const double re, const double lap_re, const double r_min, const double density_f, const double density_v, const double d_t,
                   const int fluid, const int wall, const int dwall, const int viscos_fluid, const int ghost,
                   const int externalparticle, const int innerparticle, const int surfaceparticle,
                   const int cat_on, const int cat_out,
                   const double surface_coef, const double pi, const double diriche, const int max_neighbor);

    virtual ~SurfaceTension(void);


    /*void calcSurfaceTension(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
                            Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
                            Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
                            Vec1iSoa& type, Vec1iSoa& boundary,
                            Vec1iSoa& wett_boundary_particle,
                            Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
                            Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
                            Vec1uiSoa& id,
                            hVec1iSoa& wett,
                            hVec1iSoa& hstboundary,
                            hVec1dSoa& StandardParams, unsigned int totalParticle);*/

    void calcSurfaceTension(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
                            Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
                            Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
                            Vec1iSoa& type, Vec1iSoa& boundary,
                            Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
                            Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
                            Vec1uiSoa& id,
                            hVec1iSoa& hstboundary,
                            hVec1dSoa& StandardParams, unsigned int totalParticle);



};