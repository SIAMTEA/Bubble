#include <cuda_runtime.h>

//#include <cublas_v2.h>
//#include <cusparse_v2.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

#include "VecSoa.cuh"
#include "Viscos.cuh"
#include "common.cuh"
#include "Some.cuh"

template <typename T1, typename T2>
struct AXPBY
{

	T1 alpha;
	T2 beta;

	AXPBY(T1 _alpha, T2 _beta)
		: alpha(_alpha), beta(_beta){}

	template <typename Tuple>
	__host__ __device__
		void operator()(Tuple t)
	{

			thrust::get<2>(t) = alpha * thrust::get<0>(t) +beta * thrust::get<1>(t);

		}

};

namespace viscos
{

	__device__ double weight(double distance)
	{

		double weight_ij = 0.0;

		if (distance >= _r_e){

			weight_ij = 0.0;

		}

		else{

			weight_ij = (_r_e / distance) - 1.0;
			//weight_ij = pow((1.0 - (distance / re)), 2);
			//weight_ij = pow((distance / _r_e) - 1.0, 2);


		}


		return weight_ij;
	}


	__device__  double d_lap_weight(double distance)
	{

		int i = 0;
		double weight_ij = 0.0;

		if (distance >= _lap_r_e){

			weight_ij = 0.0;

		}
		else{

			weight_ij = (_lap_r_e / distance) - 1.0;
			//weight_ij = pow((1.0 - (distance / re)), 2);
			//weight_ij = pow((distance / _lap_r_e) - 1.0, 2.0);

		}

		return weight_ij;
	}

	//Explicit

	__global__ void SetGravityTerm(double* acc_x, double* acc_y, double* acc_z,
		int* type,
		unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;


		//�d�͍�
		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //air 6/17

				acc_x[threadId] += _gf.x;
				acc_y[threadId] += _gf.y;
				acc_z[threadId] += _gf.z;


			}
			else{

				acc_x[threadId] = 0.0f;
				acc_y[threadId] = 0.0f;
				acc_z[threadId] = 0.0f;
			}
		}

	}

	__global__ void calcViscosTerm(double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		unsigned int* lapneighborIndex, unsigned int* lapneighborNum,
		double Coe_ViscosTerm, unsigned int totalParticle)
	{

		int j = 0;					//�Ώۗ��q�ԍ��p�ϐ�
		double x_ij;
		double y_ij;
		double z_ij;


		double pre_distance;
		double distance;
		double w;

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;


		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //air 6/17

				double Acceleration_x = 0.0;
				double Acceleration_y = 0.0;
				double Acceleration_z = 0.0;

				const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);
				const double3 Velocity_i = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);

				//変更 6/21
				if (type[threadId] == _fluid){

					Coe_ViscosTerm *= _kinematic_viscosity_coef;

				}
				else if (type[threadId] == _air){
					Coe_ViscosTerm *= _air_kinematic_viscosity_coef;
				}


				//Coe_ViscosTerm *= _kinematic_viscosity_coef;



				//__syncthreads;
				//for(j = 0; j < totalParticle; j++){
				for (int i = 0; i < lapneighborNum[threadId]; i++){
					const int j = lapneighborIndex[threadId * _max_neighbor + i];

					if (j == threadId || type[j] == _ghost) continue;

					double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);
					double3 Velocity_j = make_double3(vel_x[j], vel_y[j], vel_z[j]);

					x_ij = Position_j.x - Position_i.x;
					y_ij = Position_j.y - Position_i.y;
					z_ij = Position_j.z - Position_i.z;

					pre_distance = (x_ij*x_ij) + (y_ij*y_ij) + (z_ij*z_ij);

					distance = sqrt(pre_distance);

					if (distance < _lap_r_e){

						w = d_lap_weight(distance);

						//non-slip����
						Acceleration_x += (Velocity_j.x - Velocity_i.x) * w;
						Acceleration_y += (Velocity_j.y - Velocity_i.y) * w;
						Acceleration_z += (Velocity_j.z - Velocity_i.z) * w;

						//free-slip����
						/*if(type[j] == _fluid){

						Acceleration_x += (d_Vel_x[j] - Velocity_i.x) * w;
						Acceleration_y += (d_Vel_y[j] - Velocity_i.y) * w;
						Acceleration_z += (d_Vel_z[j] - Velocity_i.z) * w;

						}else if(type[j] == _wall; || type[j] == _dwall){

						Acceleration_x += (Velocity_i.x - Velocity_i.x) * w;
						Acceleration_y += (Velocity_i.y - Velocity_i.y) * w;
						Acceleration_z += (Velocity_i.z - Velocity_i.z) * w;


						}*/


					}
				}

				Acceleration_x = Acceleration_x * Coe_ViscosTerm;
				Acceleration_y = Acceleration_y * Coe_ViscosTerm;
				Acceleration_z = Acceleration_z * Coe_ViscosTerm;

				__syncthreads();

				acc_x[threadId] += Acceleration_x;
				acc_y[threadId] += Acceleration_y;
				acc_z[threadId] += Acceleration_z;

			}
		}
	}

	//Implicit





	__global__ void SetExternalForce(
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){

				double3 acc = make_double3(acc_x[threadId], acc_y[threadId], acc_z[threadId]);

				acc_x[threadId] += _gf.x;
				acc_y[threadId] += _gf.y;
				acc_z[threadId] += _gf.z;


			}
			else{

				acc_x[threadId] = 0.0f;
				acc_y[threadId] = 0.0f;
				acc_z[threadId] = 0.0f;

			}
		}
	}

	__global__ void SetX(
		double* vel_x, double* vel_y, double* vel_z,
		double* x,
		unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){

			double3 vel = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);

			x[threadId * 3] = vel.x;
			x[threadId * 3 + 1] = vel.y;
			x[threadId * 3 + 2] = vel.z;

		}
	}

	__global__ void SetRightSide(double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		double* bx,
		unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){

			double3 b = make_double3(0.0f, 0.0f, 0.0f);

			double3 vel = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);
			double3 acc = make_double3(acc_x[threadId], acc_y[threadId], acc_z[threadId]);

			//kinematic_viscosの場合
			b.x =  -vel.x / _dt - acc.x;
			b.y =  -vel.y / _dt - acc.y;
			b.z =  -vel.z / _dt - acc.z;

			//viscosの場合
			/*b.x = _density * -vel.x / _dt - acc.x;
			b.y = _density * -vel.y / _dt - acc.y;
			b.z = _density * -vel.z / _dt - acc.z;*/

			__syncthreads();

			bx[threadId * 3] = b.x;
			bx[threadId * 3 + 1] = b.y;
			bx[threadId * 3 + 2] = b.z;

		}


	}

	__global__ void SetOffsets(
		int* viscosNum,
		int* csrNum,
		unsigned int totalParticle)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){

			viscosNum[threadId * 3] = csrNum[threadId];
			viscosNum[threadId * 3 + 1] = csrNum[threadId];
			viscosNum[threadId * 3 + 2] = csrNum[threadId];

		}


	}

	__global__ void SetMatrix3
		(double* pos_x, double* pos_y, double* pos_z,
		int* type,
		double* as,
		int* csrIndices_viscos,
		int* csrIndices,
		int* csrNum,
		int* csrOffsets,
		int* csrOffsets_viscos,
		double d_Coe_ViscosTerm,
		unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		bool flag = true;

		if (threadId < totalParticle){

			const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

			double a_ii = 0.0;
			int ownNeighbor = 0;
			int iNeighbor = 0;
			double fd = 0.0;

			int type_i = type[threadId];

			int offset = csrOffsets[threadId];

			int offset1 = csrOffsets_viscos[threadId * 3];
			int offset2 = csrOffsets_viscos[threadId * 3 + 1];
			int offset3 = csrOffsets_viscos[threadId * 3 + 2];

			if (type[threadId] == _fluid){  //air 6/17


				d_Coe_ViscosTerm *= _kinematic_viscosity_coef;
				//fd = _density;


			}//変更　6/21
			else if (type[threadId] == _air){

				d_Coe_ViscosTerm *= _air_kinematic_viscosity_coef;
				//fd = _air_density;

			}else{

				d_Coe_ViscosTerm *= 0.0;

			}


			for (iNeighbor = 0; iNeighbor < csrNum[threadId]; iNeighbor++){
				int columnIndex = csrIndices[offset + iNeighbor];

				flag = true;

				if (columnIndex == threadId){

					ownNeighbor = iNeighbor;
					flag = false;

				}

				if (flag == true){

					double3 Position;

					const double3 Position_j = make_double3(pos_x[columnIndex], pos_y[columnIndex], pos_z[columnIndex]);

					Position.x = Position_j.x - Position_i.x;
					Position.y = Position_j.y - Position_i.y;
					Position.z = Position_j.z - Position_i.z;

					double pre_distance = (Position.x*Position.x) + (Position.y*Position.y) + (Position.z*Position.z);
					const double distance = sqrt(pre_distance);

					if (distance == 0.0) printf("SetMatrix::indexNumber = %d\n", threadId);

					const double w = d_lap_weight(distance);

					int type_j = type[columnIndex];

					double a_ij = d_Coe_ViscosTerm * w;

					//value
					as[offset1 + iNeighbor] = a_ij;
					as[offset2 + iNeighbor] = a_ij;
					as[offset3 + iNeighbor] = a_ij;

					//Index
					csrIndices_viscos[offset1 + iNeighbor] = columnIndex * 3;
					csrIndices_viscos[offset2 + iNeighbor] = columnIndex * 3 + 1;
					csrIndices_viscos[offset3 + iNeighbor] = columnIndex * 3 + 2;

					a_ii -= a_ij;


				}
			}
			//viscosの場合
			//a_ii -= fd / _dt;

			//kinematic_viscosの場合
			a_ii -= (double)1.0 / _dt;

			//value
			as[offset1 + ownNeighbor] = a_ii;
			as[offset2 + ownNeighbor] = a_ii;
			as[offset3 + ownNeighbor] = a_ii;

			//Index
			csrIndices_viscos[offset1 + ownNeighbor] = threadId * 3;
			csrIndices_viscos[offset2 + ownNeighbor] = threadId * 3 + 1;
			csrIndices_viscos[offset3 + ownNeighbor] = threadId * 3 + 2;



		}
	}

	__global__ void change(
		double* vel_x, double* vel_y, double* vel_z,
		double* acc_x, double* acc_y, double* acc_z,
		int* type,
		double* x,
		unsigned int totalParticle)
	{
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] == _fluid || type[threadId] == _air){  //air 6/17


				double3 vel = make_double3(x[threadId * 3], x[threadId * 3 + 1], x[threadId * 3 + 2]);

				if (isnan(vel.x) || isnan(vel.y) || isnan(vel.z)) printf("viscos::%d\n", threadId);

				vel_x[threadId] = vel.x;
				vel_y[threadId] = vel.y;
				vel_z[threadId] = vel.z;

			}

			acc_x[threadId] = 0.0f;
			acc_y[threadId] = 0.0f;
			acc_z[threadId] = 0.0f;
		}

	}

		
		
	__global__ inline void Sub(double* p, double* r, double* y, double beta, double gamma, int n)
	{

		int threadId = blockDim.x * blockIdx.x + threadIdx.x;

		double3 ans = make_double3(0.0, 0.0, 0.0);

		if (threadId < n){


			ans.x = r[threadId * 3] + (beta*(p[threadId * 3] - (gamma*y[threadId * 3])));
			ans.y = r[threadId * 3 + 1] + (beta*(p[threadId * 3 + 1] - (gamma*y[threadId * 3 + 1])));
			ans.z = r[threadId * 3 + 2] + (beta*(p[threadId * 3 + 2] - (gamma*y[threadId * 3 + 2])));

			__syncthreads();

			p[threadId * 3] = ans.x;
			p[threadId * 3 + 1] = ans.y;
			p[threadId * 3 + 2] = ans.z;

		}

	}


	__global__ inline void Update(double* x, double* p, double* s, double* r, double* z, double alpha, double gamma, int totalParticle)
	{

		int threadId = blockDim.x * blockIdx.x + threadIdx.x;

		if (threadId < totalParticle){


			x[threadId * 3] += (alpha * p[threadId * 3]) + (gamma * s[threadId * 3]);
			x[threadId * 3 + 1] += (alpha * p[threadId * 3 + 1]) + (gamma * s[threadId * 3 + 1]);
			x[threadId * 3 + 2] += (alpha * p[threadId * 3 + 2]) + (gamma * s[threadId * 3 + 2]);

			r[threadId * 3] = s[threadId * 3] - (gamma * z[threadId * 3]);
			r[threadId * 3 + 1] = s[threadId * 3 + 1] - (gamma * z[threadId * 3 + 1]);
			r[threadId * 3 + 2] = s[threadId * 3 + 2] - (gamma * z[threadId * 3 + 2]);

		}
	}
};

Viscos::Viscos(const double re, const double lap_re, const double density, const double air_density, const double d_t,  //air_density 6/21
			   const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/17
			   const double gy, const double kvc, const double a_kvc,  //a_kvc 6/21
			   const double vc, const double a_vc, const int dim,  //a_vc 6/21
			   const int max_neighbor)
			   :_r_e(re), _lap_r_e(lap_re), _density(density), _air_density(air_density), _dt(d_t),  //air_density
 _fluid(fluid), _wall(wall), _dwall(dwall), _air(air), _ghost(ghost),  //air 6/17
 _gy(gy), _kinematic_viscosity_coef(kvc), _air_kinematic_viscosity_coef(a_kvc),  //a_kvc
 _viscosity_coef(vc), _air_viscosity_coef(a_vc), _dim(dim),  //a_vc
 _max_neighbor(max_neighbor)
{
	_gf = make_double3(0.0, _gy, 0.0);

	(cudaMemcpyToSymbol(viscos::_r_e, &_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(viscos::_lap_r_e, &_lap_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(viscos::_density, &_density, sizeof(double)));
	(cudaMemcpyToSymbol(viscos::_air_density, &_air_density, sizeof(double)));  //6/21
	(cudaMemcpyToSymbol(viscos::_dt, &_dt, sizeof(double)));

	(cudaMemcpyToSymbol(viscos::_fluid, &_fluid, sizeof(int)));
	(cudaMemcpyToSymbol(viscos::_wall, &_wall, sizeof(int)));
	(cudaMemcpyToSymbol(viscos::_dwall, &_dwall, sizeof(int)));
	(cudaMemcpyToSymbol(viscos::_air, &_air, sizeof(int)));  //6/17
	(cudaMemcpyToSymbol(viscos::_ghost, &_ghost, sizeof(int)));

	(cudaMemcpyToSymbol(viscos::_gf, &_gf, sizeof(double3)));

	(cudaMemcpyToSymbol(viscos::_kinematic_viscosity_coef, &_kinematic_viscosity_coef, sizeof(double)));
	(cudaMemcpyToSymbol(viscos::_air_kinematic_viscosity_coef, &_air_kinematic_viscosity_coef, sizeof(double)));  //6/21

	(cudaMemcpyToSymbol(viscos::_viscosity_coef, &_viscosity_coef, sizeof(double)));
	(cudaMemcpyToSymbol(viscos::_air_viscosity_coef, &_air_viscosity_coef, sizeof(double)));  //6/21

	(cudaMemcpyToSymbol(viscos::_max_neighbor, &_max_neighbor, sizeof(int)));

}

Viscos::~Viscos(void)
{

}


void Viscos::calcViscos(Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
	Vec1iSoa& type, 
	Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
	hVec1dSoa& StandardParams, unsigned int totalParticle)
{

	int numThreadPerBlock = 256;
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;



	//���Ԍv���p
	cudaEvent_t start, stop;
	float elapseTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//���Ԍv���p
	//cudaError�擾�p
	cudaError_t err;

	double lapn0 = StandardParams[1];
	double lam = StandardParams[2];

	//Explicit

	//cudaEventRecord(start, 0);

	//�{���d�͍����z��@�Ȃ̂őΏۊO�����A��@�ŊO�͂Ƃ��ė^����̂Ŏg��
	viscos::SetGravityTerm << <numBlock, numThreadPerBlock >> >(
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		totalParticle);
	

	err=cudaGetLastError();
	if(err != cudaSuccess){
	printf("GravityTerm_error : %s\n", cudaGetErrorString(err));
	}

	double Coe_ViscosTerm = (2.0 * _dim) / (lapn0 * lam);

	viscos::calcViscosTerm << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
		type._rawPointer,
		lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
		Coe_ViscosTerm, totalParticle);
	
	

	err=cudaGetLastError();
	if(err != cudaSuccess){
	printf("ViscousTerm_error : %s\n", cudaGetErrorString(err));
	}
	
}

//void Viscos::calcImplicitViscosCSR(
//	Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
//	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
//	Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
//	Vec1iSoa& type, 
//	Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
//	Vec1iSoa& csrIndices, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
//	hVec1dSoa& StandardParams, unsigned int totalParticle)
//{
//
//	int numThreadPerBlock = 256;
//	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;
//
//	std::wcout << "Make" << std::endl;
//
//	//���Ԍv���p
//	cudaEvent_t start, stop;
//	float elapseTime;
//
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	//���Ԍv���p
//
//
//	//cudaError�擾�p
//	cudaError_t err;
//
//	double lapn0 = StandardParams[1];
//	double lam = StandardParams[2];
//
//	double Coe_ViscosTerm = (2.0 * _dim) / (lapn0 * lam);
//
//	Vec1dSoa b(totalParticle * 3, 0.0);
//	Vec1dSoa x(totalParticle * 3, 0.0);
//
//	int nnzero = thrust::reduce(csrNum.begin(), csrNum.end());
//
//	println(nnzero);
//
//	Vec1dSoa csr_viscos(nnzero * 3);
//	Vec1iSoa csrIndices_viscos(nnzero * 3);
//
//	csr_viscos.assign(nnzero * 3, 0.0);
//	csrIndices_viscos.assign(nnzero * 3, 0);
//
//	size_t N = b.size();
//
//	Vec1iSoa csrOffsets_viscos(totalParticle * 3 + 1, 0);
//	Vec1iSoa viscosNum(totalParticle * 3, 0);
//
//	Vec1iSoa viscosNum1(totalParticle, 0);
//
//	viscos::SetExternalForce << < numBlock, numThreadPerBlock >> >(
//		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
//		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
//		type._rawPointer,
//		totalParticle);
//
//	CHECK(cudaDeviceSynchronize());
//
//	viscos::SetRightSide << < numBlock, numThreadPerBlock >> >(
//		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
//		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
//		type._rawPointer,
//		b._rawPointer,
//		totalParticle);
//
//	CHECK(cudaDeviceSynchronize());
//
//	viscos::SetX << <numBlock, numThreadPerBlock >> >(
//		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
//		x._rawPointer,
//		totalParticle);
//
//	CHECK(cudaDeviceSynchronize());
//
//	viscos::SetOffsets << <numBlock, numThreadPerBlock >> >(
//		viscosNum._rawPointer,
//		csrNum._rawPointer,
//		totalParticle);
//
//	CHECK(cudaDeviceSynchronize());
//
//	thrust::inclusive_scan(viscosNum.begin(), viscosNum.end(), csrOffsets_viscos.begin() + 1);
//
//	viscos::SetMatrix3 << <numBlock, numThreadPerBlock >> >(
//		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
//		type._rawPointer,
//		csr_viscos._rawPointer,
//		csrIndices_viscos._rawPointer,
//		csrIndices._rawPointer,
//		csrNum._rawPointer,
//		csrOffsets._rawPointer,
//		csrOffsets_viscos._rawPointer,
//		Coe_ViscosTerm,
//		totalParticle);
//
//	CHECK(cudaDeviceSynchronize());
//
//
//	hVec1iSoa pre(totalParticle * 3);
//	hVec1iSoa pre2(totalParticle * 3 + 1);
//
//	//bicgstab
//	Vec1dSoa r;
//	Vec1dSoa s;
//	Vec1dSoa r0;
//	Vec1dSoa p;
//	Vec1dSoa y;
//	Vec1dSoa z;
//
//
//	r.resize(totalParticle * 3, 0.0);
//	s.resize(totalParticle * 3, 0.0);
//	r0.resize(totalParticle * 3, 0.0);
//	p.resize(totalParticle * 3, 0.0);
//	y.resize(totalParticle * 3, 0.0);
//	z.resize(totalParticle * 3, 0.0);
//
//	//size_t N = r.size();
//
//	const double a = 1.0;
//	const double bbb = 0.0;
//
//	//残差計算(r <- Ax)
//	cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
//		totalParticle*3, totalParticle*3, nnzero*3,
//		&a, _matDescr,
//		csr_viscos._rawPointer,
//		csrOffsets_viscos._rawPointer,
//		csrIndices_viscos._rawPointer,
//		x._rawPointer,
//		&bbb,
//		r._rawPointer);
//
//	//r <- b - A * x
//	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(b.begin(), r.begin(), r.begin())),
//		thrust::make_zip_iterator(thrust::make_tuple(b.begin(), r.begin(), r.begin())) + N,
//		AXPBY<double, double>(1.0, -1.0));
//
//
//	//Set p_0 and r0
//	//p <- r
//	thrust::copy(r.begin(), r.end(), p.begin());
//	thrust::copy(r.begin(), r.end(), r0.begin());
//
//	double init_r0, rr0, rr1;
//	double ppp, a1, a2, b1, b2, g1, g2;
//	double e = 0.0;
//	int k;
//	double alpha, beta, gamma;
//
//	init_r0 = thrust::inner_product(r.begin(), r.end(), r0.begin(), 0.0);
//
//	println(init_r0);
//
//	double r_r_star_old = init_r0;
//	double r_r_star_new = 0.0;
//
//	std::cout << "Start_BiCGSTAB_Viscos" << std::endl;
//
//	//getchar();
//
//	for (k = 0; k < _max_iter; k++){
//
//		//cudaEventRecord(start, 0);
//
//		//y <- A * p
//		cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
//			totalParticle*3, totalParticle*3, nnzero*3,
//			&a,
//			_matDescr,
//			csr_viscos._rawPointer,
//			csrOffsets_viscos._rawPointer,
//			csrIndices_viscos._rawPointer,
//			p._rawPointer,
//			&bbb,
//			y._rawPointer);
//
//
//		//calc alpha
//		//cublasDdot(h, totalParticle*3, y._rawPointer, 1, r0._rawPointer, 1, &a2);
//		a2 = thrust::inner_product(y.begin(), y.end(), r0.begin(), 0.0);
//
//		alpha = r_r_star_old / a2;
//
//
//		//calc s_k = r_k - (alpha*A*pp)  = r_k - alpha * y
//		//s_k = r_k - alpha * y
//		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(r.begin(), y.begin(), s.begin())),
//			thrust::make_zip_iterator(thrust::make_tuple(r.begin(), y.begin(), s.begin())) + N,
//			AXPBY<double, double>(1.0, -alpha));
//		
//
//		//z = A * s 
//		cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
//			totalParticle * 3, totalParticle * 3, nnzero * 3,
//			&a,
//			_matDescr,
//			csr_viscos._rawPointer,
//			csrOffsets_viscos._rawPointer,
//			csrIndices_viscos._rawPointer,
//			s._rawPointer,
//			&bbb,
//			z._rawPointer);
//
//		//calc gamma
//		g1 = thrust::inner_product(z.begin(), z.end(), s.begin(), 0.0);
//		g2 = thrust::inner_product(z.begin(), z.end(), z.begin(), 0.0);
//
//		gamma = g1 / g2;
//
//
//		//x and r update
//		viscos::Update << <numBlock, numThreadPerBlock >> >(
//			x._rawPointer,
//			p._rawPointer,
//			s._rawPointer,
//			r._rawPointer,
//			z._rawPointer,
//			alpha, gamma, totalParticle);
//
//		rr1 = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.0);
//
//		//error_check
//		e = sqrt(rr1) / sqrt(init_r0);
//		//e = sqrt(rr1);
//		if (e < 1.0e-6/*_error*/){
//
//			k++;
//			break;
//
//		}
//
//		//calc beta
//		r_r_star_new = thrust::inner_product(r.begin(), r.end(), r0.begin(), 0.0);
//
//		beta = (r_r_star_new / r_r_star_old) * (alpha / gamma);
//
//		r_r_star_old = r_r_star_new;
//
//		viscos::Sub << <numBlock, numThreadPerBlock >> >(
//			p._rawPointer,
//			r._rawPointer,
//			y._rawPointer,
//			beta, gamma, totalParticle);
//
//	}
//
//	int max_iter2 = k + 1;
//	double error2 = e;
//
//	std::cout << "BiCGSTAB_Viscos::" << max_iter2 << "::" << error2 << std::endl;
//
//	viscos::change << <numBlock, numThreadPerBlock >> >(
//		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
//		acc_x._rawPointer, acc_y._rawPointer, acc_z._rawPointer,
//		type._rawPointer,
//		x._rawPointer,
//		totalParticle);
//
//	CHECK(cudaDeviceSynchronize());
//
//	err = cudaGetLastError();
//	if (err != cudaSuccess){
//		printf("ViscousTerm_error : %s\n", cudaGetErrorString(err));
//	}
//
//	b.clear();
//	x.clear();
//
//}
