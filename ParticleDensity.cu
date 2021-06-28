#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <math.h>

#include "VecSoa.cuh"
#include "common.cuh"
#include "Some.cuh"
#include "ParticleDensity.cuh"


namespace density
{
	__device__ double weight(double distance, double re)
	{

		double weight_ij = 0.0;
		//double re = r_e;

		if (distance >= re){
			weight_ij = 0.0;
		}

		else{

			weight_ij = (re / distance) - 1.0;
			//weight_ij = pow((1.0 - (distance / re)), 2);
			//weight_ij = pow((distance / _r_e) - 1.0, 2.0);


		}

		return weight_ij;
	}

	__global__ void d_calcParticleDensity(double* pos_x, double* pos_y, double* pos_z,
		int* type, double* dens,
		unsigned int* neighbourIndex, unsigned int* neighbourNum,
		unsigned int totalParticle)
	{

		unsigned int i;
		unsigned int j;
		double x_ij;
		double y_ij;
		double z_ij;
		int ix, iy, iz;
		int jx, jy, jz;
		double distance;
		double pre_distance;
		double w;


		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		//�v�Z����
		if (threadId < totalParticle){
			if (type[threadId] != _ghost){
				double pre_n = 0.0;

				const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);
				//for(j = 0; j < totalParticle; j++){
				for (int i = 0; i < neighbourNum[threadId]; i++){
					const int j = neighbourIndex[threadId * _max_neighbor + i];

					if (j == threadId) continue;

					double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

					x_ij = Position_j.x - Position_i.x;
					y_ij = Position_j.y - Position_i.y;
					z_ij = Position_j.z - Position_i.z;

					pre_distance = (x_ij*x_ij) + (y_ij*y_ij) + (z_ij*z_ij);
					distance = sqrt(pre_distance);

					if (distance == 0.0) printf("density::indexNumber = %d\n", threadId);

					
					w = weight(distance, _r_e);

					pre_n += w;


					
				}

				dens[threadId] = pre_n;

			}
		}
	}
};

ParticleDensity::ParticleDensity(const double re, const int max_neighbor, const int ghost)
		:_r_e(re), _max_neighbor(max_neighbor), _ghost(ghost)
{

	(cudaMemcpyToSymbol(density::_r_e, &_r_e, sizeof(double)));

	(cudaMemcpyToSymbol(density::_ghost, &_ghost, sizeof(int)));

	(cudaMemcpyToSymbol(density::_max_neighbor, &_max_neighbor, sizeof(int)));

}

ParticleDensity::~ParticleDensity(void)
{

}


void ParticleDensity::calcParticleDensity(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1iSoa& type, Vec1dSoa& dens,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
	unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;
	
	density::d_calcParticleDensity << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		type._rawPointer, dens._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		totalParticle);
		
	CHECK(cudaDeviceSynchronize());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("ParticleDensity_error : %s\n", cudaGetErrorString(err));
	}

}