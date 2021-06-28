#include <cuda_runtime.h>

#include "Move.cuh"
#include "VecSoa.cuh"
#include "common.cuh"
#include "Some.cuh"
#include "Particle.cuh"


namespace move
{

    __global__ void UpdataParticle(double* pos_x, double* pos_y, double* pos_z,
                                   double* vel_x, double* vel_y, double* vel_z,
                                   double* acc_x, double* acc_y, double* acc_z,
                                   int* type, 
                                   unsigned int totalParticle)
    {

        int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21



				vel_x[threadId] += acc_x[threadId] * _dt;
				vel_y[threadId] += acc_y[threadId] * _dt;
				vel_z[threadId] += acc_z[threadId] * _dt;

				pos_x[threadId] += vel_x[threadId] * _dt;
				pos_y[threadId] += vel_y[threadId] * _dt;
				pos_z[threadId] += vel_z[threadId] * _dt;
			}

			acc_x[threadId] = acc_y[threadId] = acc_z[threadId] = 0.0;

		}
    }

	__global__ void HighOrderUpdataParticle(
		double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		unsigned int totalParticle)
	{

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21

				vel_x[threadId] += acc_x[threadId] * _dt;
				vel_y[threadId] += acc_y[threadId] * _dt;
				vel_z[threadId] += acc_z[threadId] * _dt;

			}

			acc_x[threadId] = acc_y[threadId] = acc_z[threadId] = 0.0;

		}
	}

	__global__ void ModifyParticle(
		double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		unsigned int totalParticle)
	{

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;


		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21

				vel_x[threadId] += acc_x[threadId] * _dt;
				vel_y[threadId] += acc_y[threadId] * _dt;
				vel_z[threadId] += acc_z[threadId] * _dt;

				__syncthreads();

				pos_x[threadId] += acc_x[threadId] * _dt * _dt;
				pos_y[threadId] += acc_y[threadId] * _dt * _dt;
				pos_z[threadId] += acc_z[threadId] * _dt * _dt;

			}

			acc_x[threadId] = acc_y[threadId] = acc_z[threadId] = 0.0;

		}
	}

    __global__ void HighOrderModifyParticle(
		double* pos_x, double* pos_y, double* pos_z,
        double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
        unsigned int totalParticle)
    {

        int threadId = blockIdx.x * blockDim.x + threadIdx.x;


		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21

				vel_x[threadId] += acc_x[threadId] * _dt;
				vel_y[threadId] += acc_y[threadId] * _dt;
				vel_z[threadId] += acc_z[threadId] * _dt;

				__syncthreads();

				pos_x[threadId] += vel_x[threadId] * _dt;
				pos_y[threadId] += vel_y[threadId] * _dt;
				pos_z[threadId] += vel_z[threadId] * _dt;

			}

			acc_x[threadId] = acc_y[threadId] = acc_z[threadId] = 0.0;

		}
    }

	
};

Move::Move(const double r_min, const double d_t, const int fluid, const int air)  //air 6/21
:_r_min(r_min), _dt(d_t), _fluid(fluid), _air(air)  //air 6/21
{

    (cudaMemcpyToSymbol(move::_r_min, &_r_min, sizeof(double)));
    (cudaMemcpyToSymbol(move::_dt, &_dt, sizeof(double)));
   
    (cudaMemcpyToSymbol(move::_fluid, &_fluid, sizeof(int)));
	(cudaMemcpyToSymbol(move::_air, &_air, sizeof(int)));
   
}

Move::~Move(void)
{

}

//通常のMPS法
void Move::UpdataParams(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type,
	unsigned int totalParticle)
{

    int numThreadPerBlock = 256;
    int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

    move::UpdataParticle << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer, 
		totalParticle);

    
	CHECK(cudaDeviceSynchronize());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("UpdataParams_error : %s\n", cudaGetErrorString(err));
	}

}

//高精度MPS法
void Move::HighOrderUpdataParams(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type,
	unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	move::HighOrderUpdataParticle << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		totalParticle);


	CHECK(cudaDeviceSynchronize());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("UpdataParams_error : %s\n", cudaGetErrorString(err));
	}

}


//通常のMPS法
void Move::ModifyParams(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type,
	unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	//���Ԍv���p

	//cudaEventRecord(start, 0);

	move::ModifyParticle << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("UpdataParams_error : %s\n", cudaGetErrorString(err));
	}
}

//高精度MPS法

void Move::HighOrderModifyParams(
	Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type,
	unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	//���Ԍv���p

	//cudaEventRecord(start, 0);

	move::HighOrderModifyParticle << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("UpdataParams_error : %s\n", cudaGetErrorString(err));
	}
}