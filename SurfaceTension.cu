//

#include <cuda_runtime.h>
#include "VecSoa.cuh"
#include "SurfaceTension.cuh"
#include "common.cuh"
#include "Some.cuh"

namespace surfaceTens
{

    __device__  double d_weight(double distance)
    {
        int i = 0;
        double weight_ij = 0.0;
        //double re = r_e;

        if(distance >= _r_e){
            weight_ij = 0.0;
        }else{

            //weight_ij = (_r_e / distance) - 1.0;
            weight_ij = pow((distance / _r_e) - 1.0, 2.0);

        }

        return weight_ij;
    }

    __device__  double d_lap_weight(double distance)
    {
        int i = 0;
        double weight_ij = 0.0;
        //double re = r_e;

        if(distance >= _lap_r_e){
            weight_ij = 0.0;
        }else{

            //weight_ij = (_lap_r_e / distance) - 1.0;
            weight_ij = pow((distance / _lap_r_e) - 1.0, 2.0);

        }

        return weight_ij;
    }

    __device__ int d_SurfaceWeightStep1(double distance)
    {
        int i = 0;
        int weight_ij = 0;

        if(distance >= _lap_r_e){

            weight_ij = 0.0;

        }else{

            weight_ij = 1;

        }

        return weight_ij;

    }

    __device__  int d_SurfaceWeightStep2(double distance,  int ni, int nj)
    {

        int i = 0;
        int weight_ij = 0;

        if((distance < _lap_r_e) && ( nj > ni)){

            weight_ij = 1;

        }else{

            weight_ij = 0;

        }
        return weight_ij;

    }

    __global__ void surfaceParticleDensity(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, double* pre_dens,
            unsigned int* neighborIndex, unsigned int* neighborNum,
            unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < totalParticle){

            double pre_n = 0.0;
            if(type[threadId] == _fluid){

                const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                for(int i = 0; i < neighborNum[threadId]; i++){
                    const int j = neighborIndex[threadId * _max_neighbor + i];

                    if(j == threadId) continue;
                    if(type[j] == _ghost) continue;
                    if(type[j] == _viscos_fluid) continue;

                    double3 Position;

                    const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                    Position.x = Position_j.x - Position_i.x;
                    Position.y = Position_j.y - Position_i.y;
                    Position.z = Position_j.z - Position_i.z;

                    double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                    double distance = sqrt(pre_distance);

                    if(distance < _r_e) {

                        double w = d_weight(distance);

                        pre_n += w;

                    }
                }

                pre_dens[threadId] = pre_n;

            }else if(type[threadId] == _viscos_fluid) {

                const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                for (int i = 0; i < neighborNum[threadId]; i++) {
                    const int j = neighborIndex[threadId * _max_neighbor + i];

                    if (j == threadId) continue;
                    if (type[j] == _ghost) continue;
                    if (type[j] == _fluid) continue;

                    double3 Position;

                    const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                    Position.x = Position_j.x - Position_i.x;
                    Position.y = Position_j.y - Position_i.y;
                    Position.z = Position_j.z - Position_i.z;

                    double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                    double distance = sqrt(pre_distance);

                    if (distance < _r_e) {

                        double w = d_weight(distance);

                        pre_n += w;

                    }
                }

                pre_dens[threadId] = pre_n;

            }
        }
    }

    __global__ void setSurfaceTensionBoundaryCondition(
            int* type, int*boundary, double* pre_dens,
            double n_0, unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < totalParticle) {
            if(type[threadId] == _ghost || type[threadId] == _dwall) {

                boundary[threadId] = _externalparticle;

            }else if((pre_dens[threadId] < _dirichle * n_0) && pre_dens[threadId] != 0.0){

                boundary[threadId] = _surfaceparticle;

            }else{

                boundary[threadId] = _innerparticle;

            }
        }
    }

    __global__ void surfaceDensityStep1(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, int* step1, int* boundary,
            unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
            unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < totalParticle){

            int pre_dens = 0;

            if(type[threadId] == _fluid) {
                //if (boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _viscos_fluid) continue;

                        double3 Position;

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            int w = d_SurfaceWeightStep1(distance);

                            pre_dens += w;

                        }
                    }

                step1[threadId] = pre_dens;

                // }
            }else if(type[threadId] == _viscos_fluid){
                //if (boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _fluid) continue;

                        double3 Position;

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            int w = d_SurfaceWeightStep1(distance);

                            pre_dens += w;

                        }
                    }

                step1[threadId] = pre_dens;

                // }
            }
        }

    }

    __global__ void surfaceDensityStep2(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, int* step1, int* step2, int* boundary,
            unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
            unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < totalParticle){

            int pre_dens = 0;

            if(type[threadId] == _fluid) {
                if (boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    const int ni = step1[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _viscos_fluid) continue;

                        double3 Position;
                        const int nj = step1[j];

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            int w = d_SurfaceWeightStep2(distance, ni, nj);

                            pre_dens += w;

                        }
                    }

                    step2[threadId] = pre_dens;

                }
            }else if(type[threadId] == _viscos_fluid){
                if (boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    const int ni = step1[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _fluid) continue;

                        double3 Position;

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        const int nj = step1[j];

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            int w = d_SurfaceWeightStep2(distance, ni,nj);

                            pre_dens += w;

                        }
                    }
                }

                step2[threadId] = pre_dens;

            }
        }

    }

    __global__ void calcTheta(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, int* step2, int* boundary,
            double* curvature, double* theta,
            unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
            int n_0st_lap, unsigned int totalParticle) {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if (threadId < totalParticle) {

            double rad = 0.0;

            if (type[threadId] == _fluid || type[threadId] == _viscos_fluid) {
                if (boundary[threadId] == _surfaceparticle) {

                    //rad = 1.0 - ((double) step2[threadId] / (double) n_0st_lap);

                    //theta[threadId] = acos(rad);
                    //theta[threadId] = 2.0 * rad / _lap_r_e;

                    rad = ((double)step2[threadId]*3.141519) / ((double)n_0st_lap * 2.0f);

                    theta[threadId] = 2.0f*cos(rad) / _lap_r_e;

                }

            }
        }
    }

    __global__ void calcSmoothingCurva(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, int* step2, int* boundary,
            double* curvature, double* theta,
            unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
            int n_0st_lap, double lap_n_0, unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        const double coef = 1.0 / lap_n_0;

        //smoothing
        if (threadId < totalParticle){
            if (type[threadId] == _fluid){
                if (boundary[threadId] == _surfaceparticle){

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double a = 0.0;
                    double b = 0.0;
                    double s_curva = 0.0;

                    const double curva_i = curvature[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _viscos_fluid) continue;

                        double3 Position;

                        double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        double curva_j = curvature[j];

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            a += curva_j * w;

                        }

                    }

                    s_curva = a * coef;

                    curvature[threadId] = (s_curva + curva_i) / 2.0;
                }

            }
            else if (type[threadId] == _viscos_fluid){
                if (boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double a = 0.0;
                    double b = 0.0;
                    double s_curva = 0.0;

                    const double curva_i = curvature[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _fluid) continue;

                        double3 Position;

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        double curva_j = curvature[j];

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            a += curva_j * w;
                            b += w;

                        }

                    }

                    s_curva = a * coef;

                    curvature[threadId] = (s_curva + curva_i) / 2.0;

                }
            }
        }
    }


    __global__ void calcSmoothingTheta(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, int* step2, int* boundary,
            double* curvature, double* theta,
            unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
            int n_0st_lap, unsigned int totalParticle) {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        //smoothing
        if(threadId < totalParticle){
            if(type[threadId] == _fluid){
                if(boundary[threadId] == _surfaceparticle){

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double a = 0.0;
                    double b = 0.0;
                    double s_theta = 0.0;

                    const double theta_i = theta[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _viscos_fluid) continue;

                        double3 Position;

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        const double theta_j = theta[j];

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            a += theta_j * w;
                            b += w;

                        }

                    }

                    s_theta = (theta_i + (a / b)) / 2.0;

                    curvature[threadId] = (2.0 * cos(s_theta)) / _lap_r_e;
                }

            }else if(type[threadId] == _viscos_fluid){
                if(boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double a = 0.0;
                    double b = 0.0;
                    double s_theta = 0.0;

                    const double theta_i = theta[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _fluid) continue;

                        double3 Position;

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        const double theta_j = theta[j];

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            a += theta_j * w;
                            b += w;

                        }

                    }

                    s_theta = (theta_i + (a / b)) / 2.0;

                    curvature[threadId] = (2.0 * cos(s_theta)) / _lap_r_e;

                }
            }
        }
    }

    __global__ void calcUnitVector(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, int* boundary, int* step1,
            double* unit_x, double* unit_y, double* unit_z,
            unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
            double lap_n_0, unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        double coef = 3.0 / lap_n_0;

        //double coef = 3.0f / 75.0f;

        if(threadId < totalParticle){
            if(type[threadId] == _fluid){
                if(boundary[threadId] == _surfaceparticle){

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double3 Unit = make_double3(0.0, 0.0, 0.0);

                    const double ni = (double)step1[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _viscos_fluid) continue;

                        double3 Position;
                        const double nj = (double)step1[j];

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            Unit.x += ((nj - ni) * Position.x * w) / pre_distance;
                            Unit.y += ((nj - ni) * Position.y * w) / pre_distance;
                            Unit.z += ((nj - ni) * Position.z * w) / pre_distance;


                        }
                    }

                    Unit.x *= coef;
                    Unit.y *= coef;
                    Unit.z *= coef;

                    double UnitCoef = sqrt(pow(Unit.x, 2) + pow(Unit.y, 2) + pow(Unit.z, 2));

                    if (UnitCoef != 0.0){

                        unit_x[threadId] = Unit.x / UnitCoef;
                        unit_y[threadId] = Unit.y / UnitCoef;
                        unit_z[threadId] = Unit.z / UnitCoef;


                    }
                }

            }else if(type[threadId] == _viscos_fluid) {
                if(boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double3 Unit = make_double3(0.0, 0.0, 0.0);

                    const double ni = (double) step1[threadId];

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _fluid) continue;

                        double3 Position;
                        const double nj = (double) step1[j];

                        const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            Unit.x += ((nj - ni) * Position.x * w) / pre_distance;
                            Unit.y += ((nj - ni) * Position.y * w) / pre_distance;
                            Unit.z += ((nj - ni) * Position.z * w) / pre_distance;

                        }
                    }

                    Unit.x *= coef;
                    Unit.y *= coef;
                    Unit.z *= coef;

                    double UnitCoef = sqrt(pow(Unit.x, 2) + pow(Unit.y, 2) + pow(Unit.z, 2));

                    if (UnitCoef != 0.0){

                        unit_x[threadId] = Unit.x / UnitCoef;
                        unit_y[threadId] = Unit.y / UnitCoef;
                        unit_z[threadId] = Unit.z / UnitCoef;


                    }
                }
            }
        }

    }

    __global__ void calcSmoothingUnit(
            double* pos_x, double* pos_y, double* pos_z,
            int* type, int* boundary, int* step1,
            double* unit_x, double* unit_y, double* unit_z,
            unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
            double lap_n_0, unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        const double coef = 1.0 / lap_n_0;

        if (threadId < totalParticle){
            if (type[threadId] == _fluid) {
                if (boundary[threadId] == _surfaceparticle) {

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double3 Unit_i = make_double3(unit_x[threadId], unit_y[threadId], unit_z[threadId]);

                    double3 preUnit = make_double3(0.0, 0.0, 0.0);

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _viscos_fluid) continue;

                        double3 Position;

                        double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        double3 Unit_j = make_double3(unit_x[j], unit_y[j], unit_z[j]);

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            preUnit.x += Unit_j.x * w;
                            preUnit.y += Unit_j.y * w;
                            preUnit.z += Unit_j.z * w;

                        }
                    }

                    preUnit.x *= coef;
                    preUnit.y *= coef;
                    preUnit.z *= coef;

                    __syncthreads();

                    unit_x[threadId] = (Unit_i.x + preUnit.x) / 2.0;
                    unit_y[threadId] = (Unit_i.y + preUnit.y) / 2.0;
                    unit_z[threadId] = (Unit_i.z + preUnit.z) / 2.0;

                }
            }else if(type[threadId] == _viscos_fluid){
                if(boundary[threadId] == _surfaceparticle){

                    const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

                    double3 Unit_i = make_double3(unit_x[threadId], unit_y[threadId], unit_z[threadId]);

                    double3 preUnit = make_double3(0.0, 0.0, 0.0);

                    for (int i = 0; i < lapneighborNum[threadId]; i++) {
                        const int j = lapneighborIndex[threadId * _max_neighbor + i];

                        if (j == threadId) continue;
                        if (type[j] == _ghost) continue;
                        if (type[j] == _fluid) continue;

                        double3 Position;

                        double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

                        double3 Unit_j = make_double3(unit_x[j], unit_y[j], unit_z[j]);

                        Position.x = Position_j.x - Position_i.x;
                        Position.y = Position_j.y - Position_i.y;
                        Position.z = Position_j.z - Position_i.z;

                        double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
                        double distance = sqrt(pre_distance);

                        if (distance < _lap_r_e) {

                            double w = d_lap_weight(distance);

                            preUnit.x += Unit_j.x * w;
                            preUnit.y += Unit_j.y * w;
                            preUnit.z += Unit_j.z * w;

                        }
                    }

                    preUnit.x *= coef;
                    preUnit.y *= coef;
                    preUnit.z *= coef;

                    __syncthreads();

                    unit_x[threadId] = (Unit_i.x + preUnit.x) / 2.0;
                    unit_y[threadId] = (Unit_i.y + preUnit.y) / 2.0;
                    unit_z[threadId] = (Unit_i.z + preUnit.z) / 2.0;

                }
            }
        }
    }

    //contact_angle

    __global__ void surfaceResult(
            double* acc_x, double* acc_y, double* acc_z,
            int* type, int* boundary, double* curvature,
            double* unit_x, double* unit_y, double* unit_z,
            unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < totalParticle) {

            double fd = 0.0;
            double st = 0.0;

            if (type[threadId] == _fluid || type[threadId] == _viscos_fluid) {
                if (boundary[threadId] == _surfaceparticle) {

                    if (type[threadId] == _fluid) {

                        fd = _density_f;

                    }else if(type[threadId] == _viscos_fluid){

                        fd = _density_v;

                    }



                    acc_x[threadId] += _surface_coef / fd * curvature[threadId] * unit_x[threadId] / (1.55 * _r_min);
                    acc_y[threadId] += _surface_coef / fd * curvature[threadId] * unit_y[threadId] / (1.55 * _r_min);
                    acc_z[threadId] += _surface_coef / fd * curvature[threadId] * unit_z[threadId] / (1.55 * _r_min);
                }
            }
        }
    }

    __global__ void modifySurface(
            double* vel_x, double* vel_y, double* vel_z,
            double* acc_x, double* acc_y, double* acc_z,
            int* type, int* boundary, unsigned int* id,
            unsigned int totalParticle)
    {

        const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < totalParticle){
            if(/*type[threadId] == _fluid ||*/type[threadId] == _viscos_fluid){    //水にも表面張力を掛けると水槽内の水が変形するのでかけない
                //if(id[threadId] == _catheter_out){    //常にかけるよう試す(20180704)
                    if(boundary[threadId] == _surfaceparticle){

                        vel_x[threadId] += acc_x[threadId] * _dt;
                        vel_y[threadId] += acc_y[threadId] * _dt;
                        vel_z[threadId] += acc_z[threadId] * _dt;

                    }
                //}

            }

            acc_x[threadId] = acc_y[threadId] = acc_z[threadId] = 0.0;

        }
    }

};

SurfaceTension::SurfaceTension(const double re, const double lap_re, const double r_min, const double density_f, const double density_v, const double d_t,
                               const int fluid, const int wall, const int dwall, const int viscos_fluid, const int ghost,
                               const int externalparticle, const int innerparticle, const int surfaceparticle,
                               const int cat_on, const int cat_out,
                               const double surface_coef, const double pi, const double dirichle, const int max_neighbor)
:_r_e(re), _lap_r_e(lap_re), _r_min(r_min), _density_f(density_f), _density_v(density_v), _dt(d_t),
 _fluid(fluid), _wall(wall), _dwall(dwall), _viscos_fluid(viscos_fluid), _ghost(ghost),
 _externalparticle(externalparticle), _innerparticle(innerparticle), _surfaceparticle(surfaceparticle),
 _catheter_on(cat_on), _catheter_out(cat_out),
 _surface_coef(surface_coef), _pi(pi), _dirichle(dirichle), _max_neighbor(max_neighbor)
{

    (cudaMemcpyToSymbol(surfaceTens::_r_e, &_r_e, sizeof(double)));
    (cudaMemcpyToSymbol(surfaceTens::_lap_r_e, &_lap_r_e, sizeof(double)));
    (cudaMemcpyToSymbol(surfaceTens::_r_min, &_r_min, sizeof(double)));
    (cudaMemcpyToSymbol(surfaceTens::_density_f, &_density_f, sizeof(double)));
    (cudaMemcpyToSymbol(surfaceTens::_density_v, &_density_v, sizeof(double)));
    (cudaMemcpyToSymbol(surfaceTens::_dt, &_dt, sizeof(double)));

    (cudaMemcpyToSymbol(surfaceTens::_fluid, &_fluid, sizeof(int)));
    (cudaMemcpyToSymbol(surfaceTens::_viscos_fluid, &_viscos_fluid, sizeof(int)));
    (cudaMemcpyToSymbol(surfaceTens::_wall, &_wall, sizeof(int)));
    (cudaMemcpyToSymbol(surfaceTens::_dwall, &_dwall, sizeof(int)));
    (cudaMemcpyToSymbol(surfaceTens::_ghost, &_ghost, sizeof(int)));

    (cudaMemcpyToSymbol(surfaceTens::_externalparticle, &_externalparticle, sizeof(int)));
    (cudaMemcpyToSymbol(surfaceTens::_innerparticle, &_innerparticle, sizeof(int)));
    (cudaMemcpyToSymbol(surfaceTens::_surfaceparticle, &_surfaceparticle, sizeof(int)));

    (cudaMemcpyToSymbol(surfaceTens::_surface_coef, &_surface_coef, sizeof(double)));

    (cudaMemcpyToSymbol(surfaceTens::_pi, &_pi, sizeof(double)));

    (cudaMemcpyToSymbol(surfaceTens::_max_neighbor, &_max_neighbor, sizeof(int)));

    (cudaMemcpyToSymbol(surfaceTens::_dirichle, &_dirichle, sizeof(double)));

}

SurfaceTension::~SurfaceTension(void)
{

}

void SurfaceTension::calcSurfaceTension(
        Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
        Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
        Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
        Vec1iSoa& type, Vec1iSoa& boundary,
        Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
        Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
        Vec1uiSoa& id,
        hVec1iSoa& hstboundary,
        hVec1dSoa& StandardParams, unsigned int totalParticle)
{

    int numThreadPerBlock = 256;
    int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

    double n0 = StandardParams[0];
    double lap_n0 = StandardParams[1];
    double lam = StandardParams[2];
    int boundary_st = StandardParams[3];
    int boundary_st_lap = StandardParams[4];
    int n_0st = StandardParams[5];
    int n_0st_lap = StandardParams[6];


    //���Ԍv���p
    cudaEvent_t start, stop;
    float elapseTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //���Ԍv���p

    Vec1dSoa pre_dens;
    Vec1dSoa theta;
    Vec1dSoa curvature;
    Vec1dSoa unit_x;
    Vec1dSoa unit_y;
    Vec1dSoa unit_z;
    Vec1iSoa step1;
    Vec1iSoa step2;


    pre_dens.resize(totalParticle, 0.0);
    theta.resize(totalParticle, 0.0);
    curvature.resize(totalParticle, 0.0);
    unit_x.resize(totalParticle, 0.0);
    unit_y.resize(totalParticle, 0.0);
    unit_z.resize(totalParticle, 0.0);
    step1.resize(totalParticle, 0);
    step2.resize(totalParticle, 0);

    boundary.assign(totalParticle, 0.0);

    hstboundary.resize(totalParticle, 0);

    surfaceTens::surfaceParticleDensity<<<numBlock, numThreadPerBlock>>>(
            pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
            type._rawPointer, pre_dens._rawPointer,
            neighborIndex._rawPointer, neighborNum._rawPointer,
            totalParticle);

    CHECK(cudaDeviceSynchronize());

    surfaceTens::setSurfaceTensionBoundaryCondition<<<numBlock, numThreadPerBlock>>>(
            type._rawPointer, boundary._rawPointer, pre_dens._rawPointer, n0, totalParticle);

    CHECK(cudaDeviceSynchronize());

    surfaceTens::surfaceDensityStep1<<<numBlock, numThreadPerBlock>>>(
            pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
            type._rawPointer, step1._rawPointer, boundary._rawPointer,
            lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
            totalParticle);

    CHECK(cudaDeviceSynchronize());

    surfaceTens::surfaceDensityStep2<<<numBlock, numThreadPerBlock>>>(
            pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
            type._rawPointer, step1._rawPointer, step2._rawPointer,
            boundary._rawPointer, lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
            totalParticle);

    CHECK(cudaDeviceSynchronize());

    surfaceTens::calcTheta<<<numBlock, numThreadPerBlock>>>(
            pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
            type._rawPointer, step2._rawPointer, boundary._rawPointer,
            curvature._rawPointer, theta._rawPointer,
            lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
            n_0st_lap, totalParticle);

    CHECK(cudaDeviceSynchronize());

    thrust::copy(theta.begin(), theta.end(), curvature.begin());

    /*surfaceTens::calcSmoothingCurva<<<numBlock, numThreadPerBlock>>>(
            pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
            type._rawPointer, step2._rawPointer, boundary._rawPointer,
            curvature._rawPointer, theta._rawPointer,
            lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
            n_0st_lap, lap_n0, totalParticle);

    CHECK(cudaDeviceSynchronize());*/

    surfaceTens::calcUnitVector<<<numBlock, numThreadPerBlock>>>(
            pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
            type._rawPointer, boundary._rawPointer, step1._rawPointer,
            unit_x._rawPointer, unit_y._rawPointer, unit_z._rawPointer,
            lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
            lap_n0, totalParticle);

    CHECK(cudaDeviceSynchronize());

    /*surfaceTens::calcSmoothingUnit<<<numBlock, numThreadPerBlock>>>(
            pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
            type._rawPointer, boundary._rawPointer, step1._rawPointer,
            unit_x._rawPointer, unit_y._rawPointer, unit_z._rawPointer,
            lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
            lap_n0, totalParticle);

    CHECK(cudaDeviceSynchronize());*/

    surfaceTens::surfaceResult<<<numBlock, numThreadPerBlock>>>(
            acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
            type._rawPointer, boundary._rawPointer, curvature._rawPointer,
            unit_x._rawPointer, unit_y._rawPointer, unit_z._rawPointer,
            totalParticle);

    CHECK(cudaDeviceSynchronize());

    surfaceTens::modifySurface<<<numBlock, numThreadPerBlock>>>(
            vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
            acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
            type._rawPointer, boundary._rawPointer, id._rawPointer,
            totalParticle);

    CHECK(cudaDeviceSynchronize());

    thrust::copy(boundary.begin(), boundary.end(), hstboundary.begin());

    pre_dens.clear();
    theta.clear();
    curvature.clear();
    unit_x.clear();
    unit_y.clear();
    unit_z.clear();
    step1.clear();
    step2.clear();

}


