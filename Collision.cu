#include <cuda_runtime.h>

#include "VecSoa.cuh"
#include "common.cuh"
#include "Some.cuh"
#include "Collision.cuh"

namespace collision
{

	__device__  double d_weight(double distance)
	{
		int i = 0;
		double weight_ij = 0.0;
		//double re = r_e;

		if (distance >= _r_e){

			weight_ij = 0.0;

		}
		else{

			weight_ij = (_r_e / distance) - 1.0;
			//weight_ij = pow((1.0 - (distance / re)), 2);
			//weight_ij = pow((distance / _r_e) - 1.0, 2.0);

		}

		return weight_ij;
	}

	__global__ void calcCollision(
			double* pos_x, double* pos_y, double* pos_z,
			double* vel_x, double* vel_y, double* vel_z,
			double* acc_x, double* acc_y, double* acc_z,
			int* type,
			double* afterVelocity_x,
			double* afterVelocity_y,
			double* afterVelocity_z,
			unsigned int* neighborIndex, unsigned int* neighborNum,
			unsigned int totalParticle)
	{

		int j = 0;					//�Ώۗ��q�ԍ��p�ϐ�
		double x_ij;
		double y_ij;
		double z_ij;
		int ix, iy, iz;
		int jx, jy, jz;
		int jBacket;
		double pre_distance;
		double distance;
		double w;
		double limit2 = pow(_coll_limit, 2);
		double ForceDt;
		double mi, mj;

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21
				
				//変更　6/21
				if (type[threadId] == _fluid){

					mi = _density;

				}
				else if (type[threadId] == _air){

					mi = _air_density;

				}

				//mi = _density;

				const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);
				const double3 Velocity_i = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);
				double3 Vector = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);

				//for (j = 0; j < totalParticle; j++){
				for (int l = 0; l < neighborNum[threadId]; l++){
					const int j = neighborIndex[threadId * _max_neighbor + l];

					if (j == threadId || type[j] == _ghost) continue;

					double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);
					double3 Velocity_j = make_double3(vel_x[j], vel_y[j], vel_z[j]);

					x_ij = Position_j.x - Position_i.x;
					y_ij = Position_j.y - Position_i.y;
					z_ij = Position_j.z - Position_i.z;

					pre_distance = (x_ij*x_ij) + (y_ij*y_ij) + (z_ij*z_ij);

					if (pre_distance < limit2){

						distance = sqrt(pre_distance);

						if (distance == 0.0) printf("calcCollision::indexNumber = %d\n", threadId);

						ForceDt = (Velocity_i.x - Velocity_j.x)*(x_ij / distance) + (Velocity_i.y - Velocity_j.y)*(y_ij / distance) + (Velocity_i.z - Velocity_j.z)*(z_ij / distance);

						if (ForceDt > 0.0){

							//変更　6/21
							if (type[threadId] == _fluid){

								mj = _density;

							}
							else if (type[threadId] == _air){

								mj = _air_density;

							}

							//mj = _density;

							ForceDt *= _col_rate * mi * mj / (mi + mj);


							Vector.x -= (ForceDt / mi)*(x_ij / distance);
							Vector.y -= (ForceDt / mi)*(y_ij / distance);
							Vector.z -= (ForceDt / mi)*(z_ij / distance);

							//printf("Collision\n");
						}
					}
				}

				__syncthreads();

				afterVelocity_x[threadId] = Vector.x;
				afterVelocity_y[threadId] = Vector.y;
				afterVelocity_z[threadId] = Vector.z;


			}
		}

		__syncthreads();

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21

				pos_x[threadId] += (afterVelocity_x[threadId] - vel_x[threadId])*_dt;
				pos_y[threadId] += (afterVelocity_y[threadId] - vel_y[threadId])*_dt;
				pos_z[threadId] += (afterVelocity_z[threadId] - vel_z[threadId])*_dt;

				vel_x[threadId] = afterVelocity_x[threadId];
				vel_y[threadId] = afterVelocity_y[threadId];
				vel_z[threadId] = afterVelocity_z[threadId];

			}
		}
	}

	__global__ void HighOrdercalcCollision(
		double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		double* afterVelocity_x,
		double* afterVelocity_y,
		double* afterVelocity_z,
		unsigned int* neighborIndex, unsigned int* neighborNum,
		unsigned int totalParticle)
	{

		int j = 0;					//�Ώۗ��q�ԍ��p�ϐ�
		double x_ij;
		double y_ij;
		double z_ij;
		int ix, iy, iz;
		int jx, jy, jz;
		int jBacket;
		double pre_distance;
		double distance;
		double w;
		double limit2 = pow(_coll_limit, 2);
		double ForceDt;
		double mi, mj;

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21
				//printf("%d\n", threadId);
				
				//変更　6/21
				if (type[threadId] == _fluid){

					mi = _density;

				}
				else if (type[threadId] == _air){

					mi = _air_density;

				}

				//mi = _density;

				const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);
				const double3 Velocity_i = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);
				double3 Vector = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);

				//for (j = 0; j < totalParticle; j++){
				for (int l = 0; l < neighborNum[threadId]; l++){
					const int j = neighborIndex[threadId * _max_neighbor + l];

					if (j == threadId || type[j] == _ghost) continue;

					double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);
					double3 Velocity_j = make_double3(vel_x[j], vel_y[j], vel_z[j]);

					x_ij = Position_j.x - Position_i.x;
					y_ij = Position_j.y - Position_i.y;
					z_ij = Position_j.z - Position_i.z;

					pre_distance = (x_ij*x_ij) + (y_ij*y_ij) + (z_ij*z_ij);

					if (pre_distance < limit2){

						distance = sqrt(pre_distance);

						if (distance == 0.0) printf("calcCollision::indexNumber = %d\n", threadId);

						ForceDt = (Velocity_i.x - Velocity_j.x)*(x_ij / distance) + (Velocity_i.y - Velocity_j.y)*(y_ij / distance) + (Velocity_i.z - Velocity_j.z)*(z_ij / distance);

						if (ForceDt > 0.0){

							//変更　6/21
							if (type[threadId] == _fluid){

								mj = _density;

							}
							else if (type[threadId] == _air){

								mj = _air_density;

							}

							//mj = _density;

							ForceDt *= _col_rate * mi * mj / (mi + mj);


							Vector.x -= (ForceDt / mi)*(x_ij / distance);
							Vector.y -= (ForceDt / mi)*(y_ij / distance);
							Vector.z -= (ForceDt / mi)*(z_ij / distance);

							//printf("Collision\n");
						}
					}
				}

				__syncthreads();

				afterVelocity_x[threadId] = Vector.x;
				afterVelocity_y[threadId] = Vector.y;
				afterVelocity_z[threadId] = Vector.z;


			}
		}

		//__syncthreads();  //変更 6/9

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21

				vel_x[threadId] = afterVelocity_x[threadId];
				vel_y[threadId] = afterVelocity_y[threadId];
				vel_z[threadId] = afterVelocity_z[threadId];

			}
		}
	}
};

Collision::Collision(const double re, const double coll_limit, const double density, const double air_density, const double d_t,  //air_density 6/21
	                const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/21
					const int max_neighbor, const double col_rate)
:_r_e(re), _coll_limit(coll_limit), _density(density), _air_density(air_density), _dt(d_t),
 _fluid(fluid), _wall(wall), _dwall(dwall), _air(air), _ghost(ghost),  //air 6/21
 _max_neighbor(max_neighbor), _col_rate(col_rate)
{

	
	(cudaMemcpyToSymbol(collision::_r_e, &_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(collision::_coll_limit, &_coll_limit, sizeof(double)));
	(cudaMemcpyToSymbol(collision::_density, &_density, sizeof(double)));
	(cudaMemcpyToSymbol(collision::_air_density, &_air_density, sizeof(double)));  //6/21
	(cudaMemcpyToSymbol(collision::_dt, &_dt, sizeof(double)));

	(cudaMemcpyToSymbol(collision::_fluid, &_fluid, sizeof(int)));
	(cudaMemcpyToSymbol(collision::_wall, &_wall, sizeof(int)));
	(cudaMemcpyToSymbol(collision::_dwall, &_dwall, sizeof(int)));
	(cudaMemcpyToSymbol(collision::_air, &_air, sizeof(int)));  //6/21
	(cudaMemcpyToSymbol(collision::_ghost, &_ghost, sizeof(int)));

	(cudaMemcpyToSymbol(collision::_max_neighbor, &_max_neighbor, sizeof(int)));
	(cudaMemcpyToSymbol(collision::_col_rate, &_col_rate, sizeof(double)));

}

Collision::~Collision(void)
{

}

void Collision::calcCollisionTerm(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
	unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	//���Ԍv���p
	cudaEvent_t start, stop;
	float elapseTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Vec1dSoa afterVelocity_x;
	Vec1dSoa afterVelocity_y;
	Vec1dSoa afterVelocity_z;

	afterVelocity_x.resize(totalParticle, 0.0);
	afterVelocity_y.resize(totalParticle, 0.0);
	afterVelocity_z.resize(totalParticle, 0.0);

	
	collision::calcCollision << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		afterVelocity_x._rawPointer,
		afterVelocity_y._rawPointer,
		afterVelocity_z._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		totalParticle);
		
	CHECK(cudaDeviceSynchronize());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Collision_error : %s\n", cudaGetErrorString(err));
	}

	afterVelocity_x.clear();
	afterVelocity_y.clear();
	afterVelocity_z.clear();
	
}

void Collision::HighOrdercalcCollisionTerm(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
	unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	//���Ԍv���p
	cudaEvent_t start, stop;
	float elapseTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Vec1dSoa afterVelocity_x;
	Vec1dSoa afterVelocity_y;
	Vec1dSoa afterVelocity_z;

	afterVelocity_x.resize(totalParticle, 0.0);
	afterVelocity_y.resize(totalParticle, 0.0);
	afterVelocity_z.resize(totalParticle, 0.0);


	collision::HighOrdercalcCollision << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		afterVelocity_x._rawPointer,
		afterVelocity_y._rawPointer,
		afterVelocity_z._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Collision_error : %s\n", cudaGetErrorString(err));
	}

	afterVelocity_x.clear();
	afterVelocity_y.clear();
	afterVelocity_z.clear();

}

