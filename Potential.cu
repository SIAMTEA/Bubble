
#include <cuda_runtime.h>

#include "VecSoa.cuh"
#include "Potential.cuh"
#include "common.cuh"
#include "Some.cuh"


namespace potential
{


	__device__  double d_lap_weight(double distance)
	{
		int i = 0;
		double weight_ij = 0.0;
		//double re = r_e;

		if (distance >= _lap_r_e){
			weight_ij = 0.0;
		}
		else{

			//weight_ij = (_lap_r_e / distance) - 1.0;
			weight_ij = pow((distance / _lap_r_e) - 1.0, 2.0);

		}

		return weight_ij;
	}


	__global__ void calcPotentialForce(
		double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
		double Coef_ff, double Coef_fs, unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		double r_min3 = pow(_r_min, 3);

		double Coef;
		double fd;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21


				const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

				double3 acc = make_double3(0.0, 0.0, 0.0);

				//for (int j = 0; j < totalParticle; j++){
				for (int i = 0; i < lapneighborNum[threadId]; i++) {
					const int j = lapneighborIndex[threadId * _max_neighbor + i];
					if (j == threadId) continue;
					//if (type[j] == _fluid) continue;

					double3 Position;

					const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

					Position.x = Position_j.x - Position_i.x;
					Position.y = Position_j.y - Position_i.y;
					Position.z = Position_j.z - Position_i.z;

					double pre_distance = pow(Position.x, 2) + pow(Position.y, 2) + pow(Position.z, 2);
					double distance = sqrt(pre_distance);

					if (distance < _lap_r_e) {


						if (type[j] == _wall) {

							Coef = Coef_fs;

						}
						else{

							Coef = Coef_fs;

						}

						acc.x += Coef * (distance - _r_min) * (distance - _lap_r_e) * (Position.x / distance);
						acc.y += Coef * (distance - _r_min) * (distance - _lap_r_e) * (Position.y / distance);
						acc.z += Coef * (distance - _r_min) * (distance - _lap_r_e) * (Position.z / distance);

					}


				}

				__syncthreads();

				
				fd = _density;
				
				acc_x[threadId] += acc.x / (fd * r_min3);
				acc_y[threadId] += acc.y / (fd * r_min3);
				acc_z[threadId] += acc.z / (fd * r_min3);

			}
		}
	}

}

Potential::Potential(const double lap_re, const double r_min, const double density, const double d_t,
        const int fluid, const int wall, const int dwall, const int air, const int ghost,  //6/21
        const double surface_coef, const int max_neighbor)
        :_lap_r_e(lap_re), _r_min(r_min), _density(density), _dt(d_t),
         _fluid(fluid), _wall(wall), _dwall(dwall), _air(air), _ghost(ghost),  //air 6/21
         _surface_coef(surface_coef), _max_neighbor(max_neighbor)
{


    (cudaMemcpyToSymbol(potential::_lap_r_e, &_lap_r_e, sizeof(double)));
    (cudaMemcpyToSymbol(potential::_r_min, &_r_min, sizeof(double)));
    (cudaMemcpyToSymbol(potential::_density, &_density, sizeof(double)));

    (cudaMemcpyToSymbol(potential::_fluid, &_fluid, sizeof(int)));
    (cudaMemcpyToSymbol(potential::_wall, &_wall, sizeof(int)));
    (cudaMemcpyToSymbol(potential::_dwall, &_dwall, sizeof(int)));
	(cudaMemcpyToSymbol(potential::_air, &_air, sizeof(int)));  //6/21
    (cudaMemcpyToSymbol(potential::_ghost, &_ghost, sizeof(int)));

    (cudaMemcpyToSymbol(potential::_dt, &_dt, sizeof(double)));

    (cudaMemcpyToSymbol(potential::_surface_coef, &_surface_coef, sizeof(double)));

    (cudaMemcpyToSymbol(potential::_max_neighbor, &_max_neighbor, sizeof(int)));

}

Potential::~Potential(void)
{


}

void Potential::calcPotential(
	Vec1dSoa &pos_x, Vec1dSoa &pos_y, Vec1dSoa &pos_z,
	Vec1dSoa &vel_x, Vec1dSoa &vel_y, Vec1dSoa &vel_z,
	Vec1dSoa &acc_x, Vec1dSoa &acc_y, Vec1dSoa &acc_z,
	Vec1iSoa &type,
	Vec1uiSoa &lapneighborIndex, Vec1uiSoa &lapneighborNum,
	hVec1dSoa & StandardParams, unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	double Coef_ff = StandardParams[7];
	double Coef_fs = StandardParams[8];
	double Coef_fs2 = StandardParams[8];


	potential::calcPotentialForce << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
		Coef_ff, Coef_fs, totalParticle);

	CHECK(cudaDeviceSynchronize());

	/*potential::modify << <numBlock, numThreadPerBlock >> >(
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer, id._rawPointer, wett_boundary._rawPointer,
		totalParticle);

	 CHECK(cudaDeviceSynchronize());*/

	 
}