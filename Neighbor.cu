#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "VecSoa.cuh"
#include "Neighbor.cuh"
#include "Particle.cuh"
#include "common.cuh"
#include "Some.cuh"


namespace neighbor
{

	__constant__ double _cellSizeX;
	__constant__ double _cellSizeY;
	__constant__ double _cellSizeZ;

	__constant__ uint3 _blockSize;

	__constant__ int _max_neighbor;

	__constant__ double _r_e;
	__constant__ double _lap_r_e;



	__device__ int3 CalcGridPos(const double3 Pos)
	{
		int3 GridPos;

		GridPos.x = floor(Pos.x / _cellSizeX);
		GridPos.y = floor(Pos.y / _cellSizeY);
		GridPos.z = floor(Pos.z / _cellSizeZ);


		return GridPos;

	}

	__device__ unsigned int CalcGridHash(const int3 GridPos)
	{

		int3 cellPos;
		cellPos.x = GridPos.x & (_blockSize.x - 1);
		cellPos.y = GridPos.y & (_blockSize.y - 1);
		cellPos.z = GridPos.z & (_blockSize.z - 1);

		return  __umul24(__umul24(cellPos.z, _blockSize.y), _blockSize.x) + __umul24(cellPos.y, _blockSize.x) + cellPos.x;


	}

	//�O���b�h�v�Z
	__global__ void calcGrid(double* x, double* y, double* z,
		unsigned int* GridHash, int* SortIndex,
		int* Start, int* End, unsigned int totalParticle)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= totalParticle) return;


		const double3 Pos_i = make_double3(x[index], y[index], z[index]);

		int3 GridPos = CalcGridPos(Pos_i);
		unsigned int Gridhash = CalcGridHash(GridPos);

		GridHash[index] = Gridhash;
		SortIndex[index] = index;

	}

	__global__ void calcSortedIndex(int* sortIndices, int* oriIndices, int totalParticle)
	{
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			const int oriIndex = oriIndices[threadId];
			sortIndices[oriIndex] = threadId;

		}
	}


	__global__ void calcfindIndices(int* Start, int* End, unsigned int* GridHash, int totalParticle)
	{

		extern __shared__ unsigned int next_hashes[];

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int hash;


		if (threadId < totalParticle){
			hash = GridHash[threadId];
			next_hashes[threadIdx.x + 1] = hash;


			if (threadId > 0 && threadIdx.x == 0){
				unsigned int beforeHash = GridHash[threadId - 1];
				//printf("threadIdx.x=%d\n",threadIdx.x);
				next_hashes[0] = beforeHash;
			}
		}

		__syncthreads();

		if (threadId < totalParticle){

			if (threadId == 0 || hash != next_hashes[threadIdx.x]){

				Start[hash] = threadId;
				if (threadId > 0){
					End[next_hashes[threadIdx.x]] = threadId;
				}
			}

			if (threadId == totalParticle - 1){

				End[hash] = threadId + 1;
			}
			//}
		}

	}

	__global__ void calcSortedParticle(
		double* pos_x, double* pos_y, double* pos_z,
		double* vel_x, double* vel_y, double* vel_z,
		int* type, double* dens,
		double* ax, double* ay, double* az,
		double* avx, double* avy, double* avz,
		int* atype, double* adens,
		int* ori,
		unsigned int totalParticle)
	{
		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < totalParticle){
			const int oriIndex = ori[threadId];

			ax[threadId] = pos_x[oriIndex];
			ay[threadId] = pos_y[oriIndex];
			az[threadId] = pos_z[oriIndex];

			avx[threadId] = vel_x[oriIndex];
			avy[threadId] = vel_y[oriIndex];
			avz[threadId] = vel_z[oriIndex];

			atype[threadId] = type[oriIndex];

			adens[threadId] = dens[oriIndex];

		}

	}

	__global__ void calcNeighbor(double* x, double* y, double* z, 
		unsigned int* neighIndex, unsigned int* IndexNum, 
		unsigned int* CgIndex, unsigned int* CgNum, 
		int* Start, int* End, 
		unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;


		if (threadId < totalParticle){
			int numNeighbor = 0;
			int CgnumNeighbor = 0;
			double3 Pos_i = make_double3(x[threadId], y[threadId], z[threadId]);

			int3 cellPos_i = CalcGridPos(Pos_i);

			for (int zz = -1; zz <= 1; zz++){
				for (int yy = -1; yy <= 1; yy++){
					for (int xx = -1; xx <= 1; xx++){

						int3 Pre_cellPos_j;
						Pre_cellPos_j.x = xx + cellPos_i.x;
						Pre_cellPos_j.y = yy + cellPos_i.y;
						Pre_cellPos_j.z = zz + cellPos_i.z;

						const int3 cellPos_j = Pre_cellPos_j;

						const int hash_j = CalcGridHash(cellPos_j);

						const int StartIndex = Start[hash_j];

						const int EndIndex = End[hash_j];

						for (int j = StartIndex; j < EndIndex; j++){


							const double3 Pos_j = make_double3(x[j], y[j], z[j]);

							double3 Pos_ij;

							Pos_ij.x = Pos_j.x - Pos_i.x;
							Pos_ij.y = Pos_j.y - Pos_i.y;
							Pos_ij.z = Pos_j.z - Pos_i.z;

							double sqrt_Pos_length_ij = sqrt((Pos_ij.x*Pos_ij.x) + (Pos_ij.y*Pos_ij.y) + (Pos_ij.z*Pos_ij.z));
							const double Pos_length_ij = sqrt_Pos_length_ij;

							//if (Pos_length_ij == 0.0) printf("calcNeighbor::indexNumber = %d\n", threadId);

							if (Pos_length_ij < _r_e){
								neighIndex[threadId * _max_neighbor + numNeighbor] = j;
								numNeighbor++;

							}

							if (Pos_length_ij < _lap_r_e){
								CgIndex[threadId * _max_neighbor + CgnumNeighbor] = j;
								CgnumNeighbor++;
							}

						}
					}
				}

			}

			IndexNum[threadId] = numNeighbor;
			CgNum[threadId] = CgnumNeighbor;
		}


	}

	__global__ void calcNeighborStorageCSR(
		double* pos_x, double* pos_y, double* pos_z,
		int* csrcolumnIndices,
		int* csrNum,
		int* csrOffsets,
		unsigned int* lapneighborIndex,
		unsigned int* lapneighborNum,
		unsigned int totalParticle)
	{

		const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		//ell�p�i�p�r���Ⴄ����CgNumNeighbor�ƕ��p�͏o���Ȃ��̂Œ���
		int iNeighbor = 0;

		if (threadId < totalParticle){

			double3 Pos_i = make_double3(pos_x[threadId], pos_y[threadId], pos_z[threadId]);

			int index_i, index_j;
			index_i = threadId;

			int offset = csrOffsets[threadId];

			for (int l = 0; l < lapneighborNum[threadId]; l++){
				const int j = lapneighborIndex[threadId * _max_neighbor + l];

				if (j == threadId){

					index_j = j;

					csrcolumnIndices[offset + iNeighbor] = index_j;

					iNeighbor++;

					//printf("**");

				}
				else{

					const double3 Pos_j = make_double3(pos_x[j], pos_y[j], pos_z[j]);

					double3 Pos_ij;

					Pos_ij.x = Pos_j.x - Pos_i.x;
					Pos_ij.y = Pos_j.y - Pos_i.y;
					Pos_ij.z = Pos_j.z - Pos_i.z;

					double sqrt_Pos_length_ij = sqrt((Pos_ij.x*Pos_ij.x) + (Pos_ij.y*Pos_ij.y) + (Pos_ij.z*Pos_ij.z));
					const double Pos_length_ij = sqrt_Pos_length_ij;

					if (Pos_length_ij == 0.0) printf("calcNeighborStrageELL::indexNumber = %d\n", threadId);

					if (Pos_length_ij < _lap_r_e){


						index_j = j;

						csrcolumnIndices[offset + iNeighbor] = index_j;

						iNeighbor++;

						//printf("****");

					}
				}
			}

			csrNum[threadId] = iNeighbor;

		}
	}

};

Neighbor::Neighbor(const double re, const double lap_re, const int blocksize, const int max_neighbor, const int max_particle)
:_r_e(re), _lap_r_e(lap_re), _blocksize(blocksize), _max_neighbor(max_neighbor), _max_particle(max_particle)
{

	_cell = 3.0 * _lap_r_e;
	_three_dim_block = pow(_blocksize, 3);
	uint3 _blocksized = make_uint3(_blocksize, _blocksize, _blocksize);

	(cudaMemcpyToSymbol(neighbor::_blockSize, &_blocksized, sizeof(uint3)));

	(cudaMemcpyToSymbol(neighbor::_cellSizeX, &_cell, sizeof(double)));
	(cudaMemcpyToSymbol(neighbor::_cellSizeY, &_cell, sizeof(double)));
	(cudaMemcpyToSymbol(neighbor::_cellSizeZ, &_cell, sizeof(double)));

	(cudaMemcpyToSymbol(neighbor::_r_e, &_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(neighbor::_lap_r_e, &_lap_r_e, sizeof(double)));
	(cudaMemcpyToSymbol(neighbor::_max_neighbor, &_max_neighbor, sizeof(int)));

}

Neighbor::~Neighbor(void)
{


}

void Neighbor::calcNeighborStorageCSR(
	Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
	Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
	Vec1iSoa& type, Vec1dSoa& dens,
	Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
	Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
	Vec1dSoa& csr, Vec1iSoa& csrIndcies, Vec1iSoa& csrNum, Vec1iSoa& csrOffsets,
	unsigned int totalParticle)
{

	std::cout << totalParticle << std::endl;

	Vec1iSoa Start(_three_dim_block);
	Vec1iSoa End(_three_dim_block);
	Vec1uiSoa GridHash(totalParticle);
	Vec1iSoa SortIndex(totalParticle);

	int numThreadPerBlock = 256;	// �u���b�N������̃X���b�h��
	int numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;	// �O���b�h������̃u���b�N��

	neighbor::calcGrid << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		GridHash._rawPointer, SortIndex._rawPointer,
		Start._rawPointer, End._rawPointer,
		totalParticle);

	Vec1iSoa oriIndices(_max_particle);
	Vec1iSoa sortIndices(_max_particle);

	thrust::sequence(oriIndices.begin(), oriIndices.begin() + totalParticle, 0, 1);

	thrust::sort_by_key(thrust::device, GridHash.begin(), GridHash.begin() + totalParticle, oriIndices.begin());

	int numThread = 256;
	numBlock = (totalParticle + numThread - 1) / numThread;
	int smemSize = sizeof(int)*(numThread + 1);

	neighbor::calcSortedIndex << <numBlock, numThread, smemSize >> >(
		sortIndices._rawPointer,
		oriIndices._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	thrust::fill(Start.begin(), Start.end(), 0xffffffff);
	thrust::fill(End.begin(), End.end(), 0xffffffff);

	neighbor::calcfindIndices << <numBlock, numThread, smemSize >> >(
		Start._rawPointer, End._rawPointer, GridHash._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	Vec1dSoa apos_x(totalParticle);
	Vec1dSoa apos_y(totalParticle);
	Vec1dSoa apos_z(totalParticle);
	Vec1dSoa avel_x(totalParticle);
	Vec1dSoa avel_y(totalParticle);
	Vec1dSoa avel_z(totalParticle);
	Vec1iSoa atype(totalParticle);
	Vec1dSoa adens(totalParticle);

	numBlock = (totalParticle + numThreadPerBlock - 1) / numThreadPerBlock;

	neighbor::calcSortedParticle << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		vel_x._rawPointer, vel_y._rawPointer, vel_z._rawPointer,
		type._rawPointer, dens._rawPointer,
		apos_x._rawPointer, apos_y._rawPointer, apos_z._rawPointer,
		avel_x._rawPointer, avel_y._rawPointer, avel_z._rawPointer,
		atype._rawPointer, adens._rawPointer,
		oriIndices._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	thrust::copy(apos_x.begin(), apos_x.end(), pos_x.begin());
	thrust::copy(apos_y.begin(), apos_y.end(), pos_y.begin());
	thrust::copy(apos_z.begin(), apos_z.end(), pos_z.begin());
	thrust::copy(avel_x.begin(), avel_x.end(), vel_x.begin());
	thrust::copy(avel_y.begin(), avel_y.end(), vel_y.begin());
	thrust::copy(avel_z.begin(), avel_z.end(), vel_z.begin());
	thrust::copy(atype.begin(), atype.end(), type.begin());
	thrust::copy(adens.begin(), adens.end(), dens.begin());

	neighbor::calcNeighbor << <numBlock, numThreadPerBlock >> >(
		apos_x._rawPointer, apos_y._rawPointer, apos_z._rawPointer,
		neighborIndex._rawPointer, neighborNum._rawPointer,
		lapneighborIndex._rawPointer, lapneighborNum._rawPointer,
		Start._rawPointer, End._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	Vec1uiSoa ellneighborNum(totalParticle);

	thrust::copy(lapneighborNum.begin(), lapneighborNum.begin() + totalParticle, ellneighborNum.begin());

	csrOffsets.resize(totalParticle + 1, 0);

	hVec1iSoa pre;
	pre.resize(totalParticle + 1, 0);

	
	thrust::inclusive_scan(lapneighborNum.begin(), lapneighborNum.end(), csrOffsets.begin() + 1);

	println(thrust::reduce(lapneighborNum.begin(), lapneighborNum.end()));

	int nnzero = thrust::reduce(lapneighborNum.begin(), lapneighborNum.end());

	csr.resize(nnzero, 0.0);
	csrIndcies.resize(nnzero, 0);
	csrNum.resize(totalParticle, 0);

	neighbor::calcNeighborStorageCSR << <numBlock, numThreadPerBlock >> >(
		pos_x._rawPointer, pos_y._rawPointer, pos_z._rawPointer,
		csrIndcies._rawPointer,
		csrNum._rawPointer,
		csrOffsets._rawPointer,
		lapneighborIndex._rawPointer,
		lapneighborNum._rawPointer,
		totalParticle);

	CHECK(cudaDeviceSynchronize());

	println(thrust::reduce(csrNum.begin(), csrNum.end()));

	hVec1iSoa pre2;
	pre2.resize(totalParticle, 0);
	thrust::copy(csrOffsets.begin(), csrOffsets.end(), pre.begin());
	thrust::copy(lapneighborNum.begin(), lapneighborNum.end(), pre2.begin());


	Start.clear();
	End.clear();
	GridHash.clear();
	SortIndex.clear();
	oriIndices.clear();
	sortIndices.clear();
	apos_x.clear();
	apos_y.clear();
	apos_z.clear();
	avel_x.clear();
	avel_y.clear();
	avel_z.clear();
	atype.clear();
	adens.clear();

}