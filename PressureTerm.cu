#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

#include "PressureTerm.cuh"
#include "VecSoa.cuh"
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


namespace csr_press
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

	__device__ double d_lap_weight(double distance)
	{
		int i = 0;
		double weight_ij = 0.0;
		//double re = r_e;

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

	__host__ double h_lap_weight(double distance, double re)
	{
		int i = 0;
		double weight_ij = 0.0;
		//double re = r_e;

		if (distance >= re){

			weight_ij = 0.0;

		}
		else{

			weight_ij = (re / distance) - 1.0;
			//weight_ij = pow((1.0 - (distance / re)), 2);
			//weight_ij = pow((distance / re) - 1.0, 2.0);

		}

		return weight_ij;
	}

	__global__ void SetBoundaryCondition(int* type, double* dens, int* d_BoundaryCondition, double n_0, unsigned int totalParticle)
	{

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			if (type[threadId] == _ghost || type[threadId] == _dwall){
				d_BoundaryCondition[threadId] = _externalparticle;
				
			}
			else if (dens[threadId] < _dirichle*n_0){
				d_BoundaryCondition[threadId] = _surfaceparticle;
			}
			else{
				d_BoundaryCondition[threadId] = _innerparticle;
			}
		}

	}


	__global__ void DivergencecalcB(double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		double* dens, int* type, int* boundary, double* RightSide,
		unsigned int* neighborIndex, unsigned int* neighborNum,
		double n_0, unsigned int totalParticle)
	{

		double d_t = pow(_dt, 2);
		double fd = _density;
		double Source = 0.0;


		int threadId = blockIdx.x * blockDim.x + threadIdx.x;


		if (threadId < totalParticle){

			RightSide[threadId] = 0.0;
			double3 divergence = make_double3(0.0, 0.0, 0.0);

			Source = 0.0;

			if (type[threadId] != _ghost){
				if (type[threadId] != _dwall){
					if (boundary[threadId] == _innerparticle){

						const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);
						const double3 Velocity_i = make_double3(vel_x[threadId], vel_y[threadId], vel_z[threadId]);

						for (int i = 0; i < neighborNum[threadId]; i++){
							const int j = neighborIndex[threadId * _max_neighbor + i];
							if (j == threadId) continue;
							if (type[j] == _ghost) continue;

							const double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);
							const double3 Velocity_j = make_double3(vel_x[j], vel_y[j], vel_z[j]);

							double3 Position_ij;
							double3 Velocity_ij;

							Position_ij.x = Position_j.x - Position_i.x;
							Position_ij.y = Position_j.y - Position_i.y;
							Position_ij.z = Position_j.z - Position_i.z;

							double pre_distance = pow(Position_ij.x, 2) + pow(Position_ij.y, 2) + pow(Position_ij.z, 2);

							double distance = sqrt(pre_distance);

							if (distance == 0.0) printf("DivergenceCalcB::indexNumber = %d\n", threadId);

							Velocity_ij.x = Velocity_j.x - Velocity_i.x;
							Velocity_ij.y = Velocity_j.y - Velocity_i.y;
							Velocity_ij.z = Velocity_j.z - Velocity_i.z;

							if (distance < _r_e){

								const double w = d_weight(distance);

								divergence.x += (Velocity_ij.x / distance) * (Position_ij.x / distance * w);
								divergence.y += (Velocity_ij.y / distance) * (Position_ij.y / distance * w);
								divergence.z += (Velocity_ij.z / distance) * (Position_ij.z / distance * w);

							}
						}

						divergence.x *= (_dim / n_0);
						divergence.y *= (_dim / n_0);
						divergence.z *= (_dim / n_0);

						//変更 6/21
						if (type[threadId] == _fluid){

							fd = _density;

						}
						else if (type[threadId] == _air){

							fd = _air_density;

						}

						//fd = _density;

						double pre_div = (divergence.x + divergence.y + divergence.z);

						const double Divergence = (fd / _dt) * pre_div;

						const double pnd = _relax_coe * (fd / d_t) * ((n_0 - dens[threadId]) / n_0);

						Source = Divergence + pnd;

						RightSide[threadId] = Source;
					}
					else if (boundary[threadId] == _surfaceparticle){
						RightSide[threadId] = 0.0;
					}
				}
			}

		}
	}

	__global__ void calcB(double* dens, int* type, int* boundary, double* RightSide, double n_0, unsigned int totalParticle)
	{

		double d_t = pow(_dt, 2);
		double fd = _density;
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;


		if (threadId < totalParticle){
			RightSide[threadId] = 0.0;
			if (type[threadId] != _ghost){
				if (type[threadId] != _dwall){
					if (boundary[threadId] == _innerparticle){

						//変更 6/21
						if (type[threadId] == _fluid){

							fd = _density;

						}
						else if (type[threadId] == _air){

							fd = _air_density;

						}

						/*RightSide[threadId] = _relax_coe * (fd / d_t) * ((dens[threadId] - n_0) / n_0);*/

						RightSide[threadId] = _relax_coe * (fd / d_t) * ((n_0 - dens[threadId]) / n_0);

					}
					else if (boundary[threadId] == _surfaceparticle){
						RightSide[threadId] = 0.0;
					}
				}
			}

		}
	}

	
	__global__ void SetMatrix3(
		double* pos_x, double* pos_y, double* pos_z,
		int* type,
		int* boundary,
		double* A,
		int* csrIndices,
		int* csrNum,
		int* csrOffsets,
		double a1, double a2,
		unsigned int totalParticle)
	{

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		bool flag = true;

		if (threadId < totalParticle){
			if (boundary[threadId] == _innerparticle){
				const double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

				int ownNeighbor = 0;
				double diag_value = 0.0;

				int offset = csrOffsets[threadId];

				for (int iNeighbor = 0; iNeighbor < csrNum[threadId]; iNeighbor++){
					const int columnIndex = csrIndices[offset + iNeighbor];

					flag = true;

					if (columnIndex == threadId){

						ownNeighbor = iNeighbor;
						flag = false;

					}

					if (boundary[columnIndex] == _externalparticle) continue;

					if (flag == true){

						double3 Position;

						const double3 Position_j = make_double3(pos_x[columnIndex], pos_y[columnIndex], pos_z[columnIndex]);

						Position.x = Position_j.x - Position_i.x;
						Position.y = Position_j.y - Position_i.y;
						Position.z = Position_j.z - Position_i.z;
						double pre_distance = (Position.x*Position.x) + (Position.y*Position.y) + (Position.z*Position.z);
						const double distance = sqrt(pre_distance);

						const double w = d_lap_weight(distance);
						const double a = a1 * a2 * w;

						//if (distance < _lap_r_e){
						//if (distance == 0.0) printf("countnnZ::indexNumber = %d\n", threadId);
						if (a != 0.0){


							A[offset + iNeighbor] = a;

							diag_value += a;
						}
					}

				}

				A[offset + ownNeighbor] += (-1.0)*diag_value;
				A[offset + ownNeighbor] += _compress / (_dt*_dt);
			}
		}

	}


	__global__ void NegativePress(double* press, unsigned int totalParticle)
	{

		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){

			if (press[threadId] < 0.0){

				press[threadId] = 0.0;

			}

		}

	}

	__global__ void modifyPressure(double* press, double* minpress, int* type,
		double* pos_x, double* pos_y, double* pos_z,
		unsigned int* neighborIndex, unsigned int* neighborNum,
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



		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){

			minpress[threadId] = press[threadId];

			if (type[threadId] != _ghost){
				if (type[threadId] != _dwall){

					double3 Position_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

					//for (int j = 0; j < totalParticle; j++){
					for (int l = 0; l < neighborNum[threadId]; l++){
						const int j = neighborIndex[threadId * _max_neighbor + l];
						if (j == threadId) continue;
						if (type[j] == _ghost) continue;
						if (type[j] == _dwall) continue;

						double3 Position_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

						x_ij = Position_j.x - Position_i.x;
						y_ij = Position_j.y - Position_i.y;
						z_ij = Position_j.z - Position_i.z;

						pre_distance = pow(x_ij, 2) + pow(y_ij, 2) + pow(z_ij, 2);
						distance = sqrt(pre_distance);

						if (distance < _r_e){
							if (minpress[threadId] > press[j]){

								minpress[threadId] = press[j];

							}
						}
					}
				}
			}
		}
	}

	__global__ inline void Sub(double* p, double* r, double* y, double beta, double gamma, int n)
	{

		int threadId = blockDim.x * blockIdx.x + threadIdx.x;

		double ans = 0.0;

		if (threadId < n){


			ans = r[threadId] + (beta*(p[threadId] - (gamma*y[threadId])));

			__syncthreads();

			p[threadId] = ans;

		}

	}

	__global__ inline void Update(double* x, double* p, double* s, double* r, double* z, double alpha, double gamma, int n)
	{

		int threadId = blockDim.x * blockIdx.x + threadIdx.x;

		if (threadId < n){

			x[threadId] += (alpha * p[threadId]) + (gamma * s[threadId]);

			r[threadId] = s[threadId] - (gamma * z[threadId]);

		}
	}

};

PressureTerm::PressureTerm(const double re, const double lap_re, const double r_min, const double density, const double air_density, const double d_t,  //air_density 6/21
						   const int fluid, const int wall, const int dwall, const int air, const int ghost,  //air 6/21
						   const int externalparticle, const int innerparticle, const int surfaceparticle, const int dim, const int max_neighbor,
						   const double dirichle, const double relax_coe, const double compress, 
						   const int max_iter, const double error, cusparseHandle_t cusparse, cusparseMatDescr_t matDescr)
:_r_e(re), _lap_r_e(lap_re), _r_min(r_min), _density(density), _air_density(air_density), _dt(d_t),  //air_density 6/21
 _fluid(fluid), _wall(wall), _dwall(dwall), _air(air), _ghost(ghost),  //air 6/21
 _externalparticle(externalparticle), _innerparticle(innerparticle), _surfaceparticle(surfaceparticle),
 _dim(dim), _max_neighbor(max_neighbor), _dirichle(dirichle), _relax_coe(relax_coe), _compress(compress),
 _max_iter(max_iter), _error(error), _cusparse(cusparse), _matDescr(matDescr)
{

	(cudaMemcpyToSymbol(csr_press::_r_e, &_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(csr_press::_lap_r_e, &_lap_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(csr_press::_r_min, &_r_min, sizeof(double)));
	(cudaMemcpyToSymbol(csr_press::_density, &_density, sizeof(double)));
	(cudaMemcpyToSymbol(csr_press::_air_density, &_air_density, sizeof(double)));
	(cudaMemcpyToSymbol(csr_press::_dt, &_dt, sizeof(double)));

	(cudaMemcpyToSymbol(csr_press::_fluid, &_fluid, sizeof(int)));
	(cudaMemcpyToSymbol(csr_press::_wall, &_wall, sizeof(int)));
	(cudaMemcpyToSymbol(csr_press::_dwall, &_dwall, sizeof(int)));
	(cudaMemcpyToSymbol(csr_press::_air, &_air, sizeof(int)));  //6/21
	(cudaMemcpyToSymbol(csr_press::_ghost, &_ghost, sizeof(int)));

	(cudaMemcpyToSymbol(csr_press::_externalparticle, &_externalparticle, sizeof(int)));
	(cudaMemcpyToSymbol(csr_press::_innerparticle, &_innerparticle, sizeof(int)));
	(cudaMemcpyToSymbol(csr_press::_surfaceparticle, &_surfaceparticle, sizeof(int)));

	(cudaMemcpyToSymbol(csr_press::_dim, &_dim, sizeof(int)));

	(cudaMemcpyToSymbol(csr_press::_max_neighbor, &_max_neighbor, sizeof(int)));

	(cudaMemcpyToSymbol(csr_press::_relax_coe, &_relax_coe, sizeof(double)));
	(cudaMemcpyToSymbol(csr_press::_dirichle, &_dirichle, sizeof(double)));
	(cudaMemcpyToSymbol(csr_press::_compress, &_compress, sizeof(double)));

	(cudaMemcpyToSymbol(csr_press::_max_iter, &_max_iter, sizeof(int)));
	(cudaMemcpyToSymbol(csr_press::_error, &_error, sizeof(double)));
	
	cusparseCreate(&_cusparse);

	cusparseCreateMatDescr(&_matDescr);
	cusparseSetMatType(_matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(_matDescr, CUSPARSE_INDEX_BASE_ZERO);
}

PressureTerm::~PressureTerm(void)
{

}



void PressureTerm::calcCSRPressureTerm(
	Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1iSoa& type,
	Vec1dSoa& press, Vec1dSoa& minpress, Vec1dSoa& dens,
	Vec1iSoa& boundary,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
	Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
	Vec1dSoa& csr, Vec1iSoa& csrIndices, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
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

	//cudaEventRecord(start, 0);


	//init
	boundary.assign(totalParticle, 0);
	press.assign(totalParticle, 0.0);
	minpress.assign(totalParticle, 0.0);
	double n0 = StandardParams[0];
	double lap_n0 = StandardParams[1];
	double lam = StandardParams[2];

	//bicgstab
	Vec1dSoa r;
	Vec1dSoa s;
	Vec1dSoa r0;
	Vec1dSoa p;
	Vec1dSoa y;
	Vec1dSoa z;


	r.resize(totalParticle, 0.0);
	s.resize(totalParticle, 0.0);
	r0.resize(totalParticle, 0.0);
	p.resize(totalParticle, 0.0);
	y.resize(totalParticle, 0.0);
	z.resize(totalParticle, 0.0);



	//Boundary Condition
	csr_press::SetBoundaryCondition << <numBlock, numThreadPerBlock >> >(
		type._rawPointer, dens._rawPointer, boundary._rawPointer,
		n0, totalParticle);

	CHECK(cudaDeviceSynchronize());

	//Calc SourceTerm
	Vec1dSoa b;
	Vec1dSoa x;
	b.resize(totalParticle, 0.0);
	x.resize(totalParticle, 0.0);

	csr_press::calcB << <numBlock, numThreadPerBlock >> >(
		dens._rawPointer, type._rawPointer, boundary._rawPointer,
		b._rawPointer,
		n0, totalParticle);


	CHECK(cudaDeviceSynchronize());

	//Set Matrix
	double a_1 = 1;
	double a_2 = 2.0*_dim / (lam*lap_n0);

	thrust::fill(csr.begin(), csr.end(), 0.0);

	size_t N = b.size();

	int nnzero = thrust::reduce(csrNum.begin(), csrNum.end());

	println(nnzero);

	csr_press::SetMatrix3 << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		type._rawPointer,
		boundary._rawPointer,
		csr._rawPointer,
		csrIndices._rawPointer,
		csrNum._rawPointer,
		csrOffsets._rawPointer,
		a_1, a_2,
		totalParticle);

	CHECK(cudaDeviceSynchronize());


	const double a = 1.0;
	const double bbb = 0.0;

	//残差計算

	//using
	cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
		totalParticle, totalParticle, nnzero,
		&a, _matDescr,
		csr._rawPointer,
		csrOffsets._rawPointer,
		csrIndices._rawPointer,
		x._rawPointer,
		&bbb,
		r._rawPointer);

	//using
	
	//r <- b - A * x
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(b.begin(), r.begin(), r.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(b.begin(), r.begin(), r.begin())) + N,
		AXPBY<double, double>(1.0, -1.0));

	//Set p_0 and r0
	//p <- r
	thrust::copy(r.begin(), r.end(), p.begin());
	thrust::copy(r.begin(), r.end(), r0.begin());


	double init_r0, rr0, rr1;
	double ppp, a1, a2, b1, b2, g1, g2;
	double e = 0.0;
	int k;
	double alpha, beta, gamma;

	init_r0 = thrust::inner_product(r.begin(), r.end(), r0.begin(), 0.0);

	double r_r_star_old = init_r0;
	double r_r_star_new = 0.0;

	std::wcout << r_r_star_old << std::endl;

	std::cout << "Start_BiCGSTAB" << std::endl;

	//getchar();

	for (k = 0; k < _max_iter; k++){

		//y = A * p
		cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
			totalParticle, totalParticle, nnzero,
			&a,
			_matDescr,
			csr._rawPointer,
			csrOffsets._rawPointer,
			csrIndices._rawPointer,
			p._rawPointer,
			&bbb,
			y._rawPointer);

		//calc alpha
		a2 = thrust::inner_product(y.begin(), y.end(), r0.begin(), 0.0);

		alpha = r_r_star_old / a2;


		//calc s_k = r_k - (alpha*A*pp)  = r_k - alpha * y
		//s_k = r_k - alpha * y
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(r.begin(), y.begin(), s.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(r.begin(), y.begin(), s.begin())) + N,
			AXPBY<double, double>(1.0, -alpha));
		

		//z = A * s 
		cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
			totalParticle, totalParticle, nnzero,
			&a,
			_matDescr,
			csr._rawPointer,
			csrOffsets._rawPointer,
			csrIndices._rawPointer,
			s._rawPointer,
			&bbb,
			z._rawPointer);

		//calc gamma
		g1 = thrust::inner_product(z.begin(), z.end(), s.begin(), 0.0);
		g2 = thrust::inner_product(z.begin(), z.end(), z.begin(), 0.0);

		gamma = g1 / g2;

		//x and r update
		csr_press::Update << <numBlock, numThreadPerBlock >> >(
			x._rawPointer,
			p._rawPointer,
			s._rawPointer,
			r._rawPointer,
			z._rawPointer,
			alpha, gamma, totalParticle);

		rr1 = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.0);

		//error_check
		e = sqrt(rr1) / sqrt(init_r0);
		if (e < 1.0E-6){

			k++;
			break;

		}

		//calc beta
		r_r_star_new = thrust::inner_product(r.begin(), r.end(), r0.begin(), 0.0);

		beta = (r_r_star_new / r_r_star_old) * (alpha / gamma);

		r_r_star_old = r_r_star_new;

		csr_press::Sub << <numBlock, numThreadPerBlock >> >(
			p._rawPointer,
			r._rawPointer,
			y._rawPointer,
			beta, gamma, totalParticle);
	}

	int max_iter2 = k + 1;
	double error2 = e;

	std::cout << "BiCGSTAB::" << max_iter2 << "::" << error2 << std::endl;

	

	thrust::copy(x.begin(), x.end(), press.begin());

	csr_press::NegativePress << <numBlock, numThreadPerBlock >> >(press._rawPointer, totalParticle);

	CHECK(cudaDeviceSynchronize());

	csr_press::modifyPressure << <numBlock, numThreadPerBlock >> >(
		press._rawPointer, minpress._rawPointer, type._rawPointer,
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	b.clear();
	x.clear();
	
}

void PressureTerm::HighOrdercalcCSRPressureTerm(
	Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1iSoa& type,
	Vec1dSoa& press, Vec1dSoa& minpress, Vec1dSoa& dens,
	Vec1iSoa& boundary,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
	Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
	Vec1dSoa& csr, Vec1iSoa& csrIndices, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
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

	//cudaEventRecord(start, 0);


	//init
	boundary.assign(totalParticle, 0);
	press.assign(totalParticle, 0.0);
	minpress.assign(totalParticle, 0.0);
	double n0 = StandardParams[0];
	double lap_n0 = StandardParams[1];
	double lam = StandardParams[2];

	//bicgstab
	Vec1dSoa r;
	Vec1dSoa s;
	Vec1dSoa r0;
	Vec1dSoa p;
	Vec1dSoa y;
	Vec1dSoa z;


	r.resize(totalParticle, 0.0);
	s.resize(totalParticle, 0.0);
	r0.resize(totalParticle, 0.0);
	p.resize(totalParticle, 0.0);
	y.resize(totalParticle, 0.0);
	z.resize(totalParticle, 0.0);



	//Boundary Condition
	csr_press::SetBoundaryCondition << <numBlock, numThreadPerBlock >> >(
		type._rawPointer, dens._rawPointer, boundary._rawPointer,
		n0, totalParticle);

	CHECK(cudaDeviceSynchronize());

	//Calc SourceTerm
	Vec1dSoa b;
	Vec1dSoa x;
	b.resize(totalParticle, 0.0);
	x.resize(totalParticle, 0.0);

	csr_press::DivergencecalcB << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		dens._rawPointer, type._rawPointer, boundary._rawPointer,
		b._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		n0, totalParticle);


	CHECK(cudaDeviceSynchronize());

	//Set Matrix
	double a_1 = 1;
	double a_2 = 2.0*_dim / (lam*lap_n0);

	thrust::fill(csr.begin(), csr.end(), 0.0);

	size_t N = b.size();

	int nnzero = thrust::reduce(csrNum.begin(), csrNum.end());

	println(nnzero);

	csr_press::SetMatrix3 << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		type._rawPointer,
		boundary._rawPointer,
		csr._rawPointer,
		csrIndices._rawPointer,
		csrNum._rawPointer,
		csrOffsets._rawPointer,
		a_1, a_2,
		totalParticle);

	CHECK(cudaDeviceSynchronize());


	const double a = 1.0;
	const double bbb = 0.0;

	//残差計算

	//using
	cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
		totalParticle, totalParticle, nnzero,
		&a, _matDescr,
		csr._rawPointer,
		csrOffsets._rawPointer,
		csrIndices._rawPointer,
		x._rawPointer,
		&bbb,
		r._rawPointer);

	//using

	//r <- b - A * x
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(b.begin(), r.begin(), r.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(b.begin(), r.begin(), r.begin())) + N,
		AXPBY<double, double>(1.0, -1.0));

	//Set p_0 and r0
	//p <- r
	thrust::copy(r.begin(), r.end(), p.begin());
	thrust::copy(r.begin(), r.end(), r0.begin());


	double init_r0, rr0, rr1;
	double ppp, a1, a2, b1, b2, g1, g2;
	double e = 0.0;
	int k;
	double alpha, beta, gamma;

	init_r0 = thrust::inner_product(r.begin(), r.end(), r0.begin(), 0.0);

	double r_r_star_old = init_r0;
	double r_r_star_new = 0.0;

	std::wcout << r_r_star_old << std::endl;

	std::cout << "Start_BiCGSTAB" << std::endl;

	//getchar();

	for (k = 0; k < _max_iter; k++){

		//y = A * p
		cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
			totalParticle, totalParticle, nnzero,
			&a,
			_matDescr,
			csr._rawPointer,
			csrOffsets._rawPointer,
			csrIndices._rawPointer,
			p._rawPointer,
			&bbb,
			y._rawPointer);

		//calc alpha
		a2 = thrust::inner_product(y.begin(), y.end(), r0.begin(), 0.0);

		alpha = r_r_star_old / a2;


		//calc s_k = r_k - (alpha*A*pp)  = r_k - alpha * y
		//s_k = r_k - alpha * y
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(r.begin(), y.begin(), s.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(r.begin(), y.begin(), s.begin())) + N,
			AXPBY<double, double>(1.0, -alpha));


		//z = A * s 
		cusparseDcsrmv(_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
			totalParticle, totalParticle, nnzero,
			&a,
			_matDescr,
			csr._rawPointer,
			csrOffsets._rawPointer,
			csrIndices._rawPointer,
			s._rawPointer,
			&bbb,
			z._rawPointer);

		//calc gamma
		g1 = thrust::inner_product(z.begin(), z.end(), s.begin(), 0.0);
		g2 = thrust::inner_product(z.begin(), z.end(), z.begin(), 0.0);

		gamma = g1 / g2;

		//x and r update
		csr_press::Update << <numBlock, numThreadPerBlock >> >(
			x._rawPointer,
			p._rawPointer,
			s._rawPointer,
			r._rawPointer,
			z._rawPointer,
			alpha, gamma, totalParticle);

		rr1 = thrust::inner_product(r.begin(), r.end(), r.begin(), 0.0);

		//error_check
		e = sqrt(rr1) / sqrt(init_r0);
		if (e < 1.0E-6){

			k++;
			break;

		}

		//calc beta
		r_r_star_new = thrust::inner_product(r.begin(), r.end(), r0.begin(), 0.0);

		beta = (r_r_star_new / r_r_star_old) * (alpha / gamma);

		r_r_star_old = r_r_star_new;

		csr_press::Sub << <numBlock, numThreadPerBlock >> >(
			p._rawPointer,
			r._rawPointer,
			y._rawPointer,
			beta, gamma, totalParticle);
	}

	int max_iter2 = k + 1;
	double error2 = e;

	std::cout << "BiCGSTAB::" << max_iter2 << "::" << error2 << std::endl;

	thrust::copy(x.begin(), x.end(), press.begin());

	csr_press::NegativePress << <numBlock, numThreadPerBlock >> >(press._rawPointer, totalParticle);

	CHECK(cudaDeviceSynchronize());

	csr_press::modifyPressure << <numBlock, numThreadPerBlock >> >(
		press._rawPointer, minpress._rawPointer, type._rawPointer,
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	b.clear();
	x.clear();

}