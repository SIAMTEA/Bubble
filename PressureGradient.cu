#include <cuda_runtime.h>
#include <cuda.h>

#include "common.cuh"
#include "VecSoa.cuh"
#include "PressureGradient.cuh"


namespace gradient
{

	__device__ double d_weight(double distance)
	{
		int i = 0;
		double weight_ij = 0.0;

		if (distance >= _r_e){
			weight_ij = 0.0;
		}
		else{

			weight_ij = (_r_e / distance) - 1.0;
			//weight_ij = pow((1.0 - (distance / re)), 2);*/
			//weight_ij = pow((distance / _r_e) - 1.0, 2.0);
		}

		return weight_ij;
	}

	__global__ void calcGradient(double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		double* press, double* minpress, int* type,
		unsigned int* neighbourIndex, unsigned int* neighbourNum,
		double n_0, double another_Coe2,
		unsigned int totalParticle)
	{

		int j = 0;
		double x_ij = 0;
		double y_ij = 0;
		double z_ij = 0;
		int ix, iy, iz;
		int jx, jy, jz;
		int jBacket;
		double pre_distance;
		double distance;
		double w;
		double preAcc = 0.0;
		double Acceleration_x;
		double Acceleration_y;
		double Acceleration_z;

		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		double fd = 0.0;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21

				//変更　6/21
				if (type[threadId] == _fluid){

					fd = _density;

				}
				else if (type[threadId] == _air){

					fd = _air_density;

				}

				//fd = _density;

				double3 Acceleration = make_double3(0.0, 0.0, 0.0);
				double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);
				const double minPressure = minpress[threadId];

				//for (int j = 0; j < totalParticle; j++){
				for (int i = 0; i < neighbourNum[threadId]; i++){
					const int j = neighbourIndex[threadId * _max_neighbor + i];
					if (j == threadId) continue;
					if (type[j] == _ghost) continue;
					if (type[j] == _dwall) continue;

					double3 Position;
					const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);
					const double Pressure = press[j];

					Position.x = Position_j.x - Position_i.x;
					Position.y = Position_j.y - Position_i.y;
					Position.z = Position_j.z - Position_i.z;

					const double3 posi = make_double3(Position.x, Position.y, Position.z);
					pre_distance = (posi.x*posi.x) + (posi.y*posi.y) + (posi.z*posi.z);
					const double distance = sqrt(pre_distance);

					if (distance == 0.0) printf("calcGradient::indexNumber = %d\n", threadId);
					if (distance < _r_e){

						const double w = d_weight(distance);

						Acceleration.x += (Pressure - minPressure) * posi.x * w / (distance*distance);
						Acceleration.y += (Pressure - minPressure) * posi.y * w / (distance*distance);
						Acceleration.z += (Pressure - minPressure) * posi.z * w / (distance*distance);


					}
				}

				__syncthreads();

				Acceleration.x *= (another_Coe2 / fd);
				Acceleration.y *= (another_Coe2 / fd);
				Acceleration.z *= (another_Coe2 / fd);


				acc_x[threadId] = (-1.0)*Acceleration.x;
				acc_y[threadId] = (-1.0)*Acceleration.y;
				acc_z[threadId] = (-1.0)*Acceleration.z;

			}
		}
	}


	__global__ void calcNeighborParticle(
		double* pos_x, double* pos_y, double* pos_z,
		int* type, unsigned int* counter,
		unsigned int* neighbourIndex, unsigned int* neighbourNum,
		unsigned int totalParticle)
	{

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] != _ghost){

				unsigned int count = 0;

				const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

				for (int i = 0; i < neighbourNum[threadId]; i++){
					const int j = neighbourIndex[threadId * _max_neighbor + i];

					if (j == threadId) continue;

					double3 Position_ij;

					const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

					Position_ij.x = Position_j.x - Position_i.x;
					Position_ij.y = Position_j.y - Position_i.y;
					Position_ij.z = Position_j.z - Position_i.z;

					double pre_distance = pow(Position_ij.x, 2) + pow(Position_ij.y, 2) + pow(Position_ij.z, 2);

					double distance = sqrt(pre_distance);

					if (distance < _r_e){

						count++;

					}

				}

				counter[threadId] = count;

			}
		}

	}

	__global__ void calcMGradient(double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		double* press, double* minpress, int* type,
		unsigned int* counter,
		unsigned int* neighbourIndex, unsigned int* neighbourNum,
		double n_0, double another_Coe,
		unsigned int totalParticle)
	{

		int j = 0;
		double x_ij = 0;
		double y_ij = 0;
		double z_ij = 0;
		int ix, iy, iz;
		int jx, jy, jz;
		int jBacket;
		double pre_distance;
		double distance;
		double w;
		double preAcc = 0.0;
		double Acceleration_x;
		double Acceleration_y;
		double Acceleration_z;

		
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		double fd = 0.0;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //6/21
				if (counter[threadId] >= 3){

					double3 Acceleration = make_double3(0.0, 0.0, 0.0);

					double3 at_x = make_double3(0.0, 0.0, 0.0);
					double3 at_y = make_double3(0.0, 0.0, 0.0);
					double3 at_z = make_double3(0.0, 0.0, 0.0);
					double3 t_x = make_double3(0.0, 0.0, 0.0);
					double3 t_y = make_double3(0.0, 0.0, 0.0);
					double3 t_z = make_double3(0.0, 0.0, 0.0);
					double3 tensor = make_double3(0.0, 0.0, 0.0);

					double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);
					const double minPressure = minpress[threadId];
					
					//変更　6/21
					if (type[threadId] == _fluid){

						fd = _density;

					}
					else if (type[threadId] == _air){

						fd = _air_density;

					}

					//fd = _density;

					//for (int j = 0; j < totalParticle; j++){
					for (int i = 0; i < neighbourNum[threadId]; i++){
						const int j = neighbourIndex[threadId * _max_neighbor + i];
						if (j == threadId) continue;
						if (type[j] == _ghost) continue;
						if (type[j] == _dwall) continue;

						double3 Position;
						const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);
						const double Pressure = press[j];

						Position.x = Position_j.x - Position_i.x;
						Position.y = Position_j.y - Position_i.y;
						Position.z = Position_j.z - Position_i.z;

						const double3 posi = make_double3(Position.x, Position.y, Position.z);
						pre_distance = (posi.x*posi.x) + (posi.y*posi.y) + (posi.z*posi.z);
						const double distance = sqrt(pre_distance);

						if (distance == 0.0) printf("calcGradient::indexNumber = %d\n", threadId);
						if (distance < _r_e){

							const double w = d_weight(distance);

							tensor.x = posi.x / (distance)* w;
							tensor.y = posi.y / (distance)* w;
							tensor.z = posi.z / (distance)* w;



							t_x.x += tensor.x * (posi.x / distance);
							t_x.y += tensor.x * (posi.y / distance);
							t_x.z += tensor.x * (posi.z / distance);
							t_y.x += tensor.y * (posi.x / distance);
							t_y.y += tensor.y * (posi.y / distance);
							t_y.z += tensor.y * (posi.z / distance);
							t_z.x += tensor.z * (posi.x / distance);
							t_z.y += tensor.z * (posi.y / distance);
							t_z.z += tensor.z * (posi.z / distance);


							Acceleration.x += (Pressure - minPressure) * posi.x * w / (distance*distance);
							Acceleration.y += (Pressure - minPressure) * posi.y * w / (distance*distance);
							Acceleration.z += (Pressure - minPressure) * posi.z * w / (distance*distance);


						}
					}

					__syncthreads();

					t_x.x *= another_Coe;
					t_x.y *= another_Coe;
					t_x.z *= another_Coe;
					t_y.x *= another_Coe;
					t_y.y *= another_Coe;
					t_y.z *= another_Coe;
					t_z.x *= another_Coe;
					t_z.y *= another_Coe;
					t_z.z *= another_Coe;


					const double Inverse_Coe = 1 / ((t_x.x*t_y.y*t_z.z) + (t_x.y*t_y.z*t_z.x) + (t_x.z*t_y.x*t_z.y) - (t_x.z*t_y.y*t_z.x) - (t_x.y*t_y.x*t_z.z) - (t_x.x*t_y.z*t_z.y));
					if (Inverse_Coe != 0){

						at_x.x = (t_y.y*t_z.z - t_y.z*t_z.y) * Inverse_Coe;
						at_x.y = (t_x.z*t_z.y - t_x.y*t_z.z) * Inverse_Coe;
						at_x.z = (t_x.y*t_y.z - t_x.z*t_y.y) * Inverse_Coe;
						at_y.x = (t_y.z*t_z.x - t_y.x*t_z.z) * Inverse_Coe;
						at_y.y = (t_x.x*t_z.z - t_x.z*t_z.x) * Inverse_Coe;
						at_y.z = (t_x.z*t_y.x - t_x.x*t_y.z) * Inverse_Coe;
						at_z.x = (t_y.x*t_z.y - t_y.y*t_z.x) * Inverse_Coe;
						at_z.y = (t_x.y*t_z.x - t_x.x*t_z.y) * Inverse_Coe;
						at_z.z = (t_x.x*t_y.y - t_x.y*t_y.x) * Inverse_Coe;
					}


					Acceleration.x *= another_Coe;
					Acceleration.y *= another_Coe;
					Acceleration.z *= another_Coe;



					acc_x[threadId] = (-1.0)*(at_x.x*Acceleration.x + at_x.y*Acceleration.y + at_x.z*Acceleration.z) / fd;
					acc_y[threadId] = (-1.0)*(at_y.x*Acceleration.x + at_y.y*Acceleration.y + at_y.z*Acceleration.z) / fd;
					acc_z[threadId] = (-1.0)*(at_z.x*Acceleration.x + at_z.y*Acceleration.y + at_z.z*Acceleration.z) / fd;

				}
			}
		}
	}
};

PressureGradient::PressureGradient(const double re, const double r_min, const double density, const double air_density, const double d_t,  //air_density 6/21
								   const int fluid, const int wall, const int dwall, const int air, const int ghost, const int max_neighbor)  //air 6/21
:_r_e(re), _r_min(r_min), _density(density), _air_density(air_density), _dt(d_t),  //air_density 6/21
 _fluid(fluid), _wall(wall), _dwall(dwall), _air(air), _ghost(ghost),  //air 6/21
 _max_neighbor(max_neighbor)
{

	(cudaMemcpyToSymbol(gradient::_r_e, &_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(gradient::_r_min, &_r_min, sizeof(double)));
	(cudaMemcpyToSymbol(gradient::_density, &_density, sizeof(double)));
	(cudaMemcpyToSymbol(gradient::_air_density, &_air_density, sizeof(double)));  //6/21 
	(cudaMemcpyToSymbol(gradient::_dt, &_dt, sizeof(double)));

	(cudaMemcpyToSymbol(gradient::_fluid, &_fluid, sizeof(int)));
	(cudaMemcpyToSymbol(gradient::_wall, &_wall, sizeof(int)));
	(cudaMemcpyToSymbol(gradient::_dwall, &_dwall, sizeof(int)));
	(cudaMemcpyToSymbol(gradient::_air, &_air, sizeof(int)));  //6/21
	(cudaMemcpyToSymbol(gradient::_ghost, &_ghost, sizeof(int)));

	(cudaMemcpyToSymbol(gradient::_max_neighbor, &_max_neighbor, sizeof(int)));

}

PressureGradient::~PressureGradient(void)
{

}

void PressureGradient::calcPressureGradient(
	Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type, Vec1dSoa& press, Vec1dSoa& minpress,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum, 
	hVec1dSoa& StandardParams, unsigned int totalParticle)
{

	

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	//cudaEventRecord(start, 0);

	double n0 = StandardParams[0];

	double another_Coe = DIM / n0;

	Vec1uiSoa counter(totalParticle);

	counter.assign(totalParticle, 0);

	gradient::calcGradient << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		press._rawPointer, minpress._rawPointer, type._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		n0, another_Coe, totalParticle);

	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Gradient_error : %s\n", cudaGetErrorString(err));
	}

	counter.clear();

}

void PressureGradient::HighOrdercalcPressureGradient(
	Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type, Vec1dSoa& press, Vec1dSoa& minpress,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
	hVec1dSoa& StandardParams, unsigned int totalParticle)
{



	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	//cudaEventRecord(start, 0);

	double n0 = StandardParams[0];

	double another_Coe = 1.0 / n0;

	Vec1uiSoa counter(totalParticle);

	counter.assign(totalParticle, 0);

	gradient::calcNeighborParticle << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		type._rawPointer, counter._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		totalParticle);

	cudaDeviceSynchronize();

	gradient::calcMGradient << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		press._rawPointer, minpress._rawPointer, type._rawPointer,
		counter._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		n0, another_Coe, totalParticle);

	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Gradient_error : %s\n", cudaGetErrorString(err));
	}

	counter.clear();

}
