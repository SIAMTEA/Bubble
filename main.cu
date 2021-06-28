#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include "VecSoa.cuh"
#include "Some.cuh"
#include "Particle.cuh"
#include "Output.cuh"
#include "Neighbor.cuh"
#include "Viscos.cuh"
#include "ParticleDensity.cuh"
#include "Move.cuh"
#include "Collision.cuh"
#include "PressureTerm.cuh"
#include "PressureGradient.cuh"
#include "SurfaceTension.cuh"
#include "Potential.cuh"

//void particle(double* pre);

Particle*			_particle;         //粒子
Output*				_output;           //出力
Neighbor*			_neighbor;         //?
Viscos*				_viscos;           //粘度?
ParticleDensity*	_particledensity;  //粒子数密度?
Move*				_move;             //?
Collision*			_collision;        //衝突?接触?
PressureTerm*		_pressure;         //圧力項
PressureGradient*	_gradient;         //圧力勾配
SurfaceTension*     _surface;          //表面張力
Potential*          _potential;        //ポテンシャル,伸び?


void Init(cusparseHandle_t cusparse, cusparseMatDescr_t matDescr)
{

	_particle = new Particle(r_e, lap_r_e, FPD, FLUID, WALL, D_WALL, AIR, Max_Neighbor);  //AIR 6/17
	
	_neighbor = new Neighbor(r_e, lap_r_e, blockSized, Max_Neighbor, MaxParticle);
	
	_particledensity = new ParticleDensity(r_e, Max_Neighbor, GHOST);
	
    _potential = new Potential(lap_r_e, FPD, DENSITY, dt,
                               FLUID, WALL, D_WALL, AIR, GHOST,  //AIR 6/21
                               SURFACETENSION_COEF, Max_Neighbor);
	
	_viscos = new Viscos(r_e, lap_r_e, DENSITY, AIR_DENSITY, dt, FLUID, WALL, D_WALL, AIR, GHOST,  //AIR 6/17,  AIR_DENSITY 6/21
		                 G_Y, KINEMATIC_VISCOSITY_COEF, AIR_KINEMATIC_VISCOSITY_COEF,  //AIR_KINEMATIC_VISCOSITY_COEF 6/21
		                 VISCOSITY_COEF, AIR_VISCOSITY_COEF, DIM,  //AIR_VISCOSITY_COEF 6/21
						 Max_Neighbor);
	
	_move = new Move(FPD, dt, FLUID, AIR);  //AIR 6/21
	
	_collision = new Collision(r_e, COLLISION_LIMIT, DENSITY, AIR_DENSITY, dt,  //AIR_DENSITY 6/21
							   FLUID, WALL, D_WALL, AIR, GHOST,  //AIR 6/21
							   Max_Neighbor, COLLISION_RATE);
	
	_pressure = new PressureTerm(r_e, lap_r_e, FPD, DENSITY, AIR_DENSITY, dt, FLUID, WALL, D_WALL, AIR, GHOST,  //AIR 6/21, AIR_DENSITY 6/21
								 External_Particle, Inner_Particle, Surface_Particle, DIM, Max_Neighbor,
								 DIRICHLET, PRESSURE_RELAX_COEF, COMPRESSIBILITY, MAX_ITER, ERROR, cusparse, matDescr);

	_gradient = new PressureGradient(r_e, FPD, DENSITY, AIR_DENSITY, dt, FLUID, WALL, D_WALL, AIR, GHOST, Max_Neighbor);  //AIR 6/21, AIR_DENSITY 6/21


	_particle->InitParticlePosition(_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z());
	
	_particle->InitParticleType_and_FirstVel(_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
                                             _particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
                                             _particle->gethParticleType(), _particle->getSize());

    println(_particle->getSize());

	_particle->UpdataResize(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getPressure(), _particle->getminPressure(),
		_particle->getDensity(), _particle->getParticleType(),
        _particle->getBoundary(), 
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
		_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
		_particle->gethAcceleration_x(), _particle->gethAcceleration_y(), _particle->gethAcceleration_z(),
		_particle->gethPressure(), _particle->gethminPressure(), 
		_particle->gethDensity(), _particle->gethParticleType(),
        _particle->gethBoundary(), 
		_particle->getSize());

	println(_particle->getSize());

	_particle->Convert(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getPressure(), _particle->getminPressure(),
		_particle->getDensity(), _particle->getParticleType(),
        _particle->getBoundary(), 
		_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
		_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
		_particle->gethAcceleration_x(), _particle->gethAcceleration_y(), _particle->gethAcceleration_z(),
		_particle->gethPressure(), _particle->gethminPressure(),
		_particle->gethDensity(), _particle->gethParticleType(),
        _particle->gethBoundary(),
		_particle->getSize());

	println(_particle->getSize());
	_particle->calcInitStandardParams(_particle->getStandardParams());

    _particle->PotentialCoef(_particle->getStandardParams());

	println(_particle->getSize());

	_output->WritingData_vtk(
			_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
			_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
			_particle->gethPressure(), _particle->gethParticleType(), _particle->gethBoundary(),
			0, _particle->getSize());

	_output->WritingData_txt(_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
		_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
		_particle->gethPressure(), _particle->gethParticleType(), _particle->gethDensity(), _particle->gethBoundary(),
		0, 0.0, _particle->getSize());

	

}



void simulation()
{

	_neighbor->calcNeighborStorageCSR(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getParticleType(), _particle->getDensity(), 
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getCSRMatrix(), _particle->getCSRIndices(), _particle->getCSRNum(), _particle->getCSROffsets(),
		_particle->getSize());

	_viscos->calcViscos(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(), 
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getStandardParams(), _particle->getSize());

	/*_viscos->calcImplicitViscosCSR(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(), 
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getCSRIndices(), _particle->getCSRNum(), _particle->getCSROffsets(),
		_particle->getStandardParams(), _particle->getSize());*/

	_move->UpdataParams(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(), 
		_particle->getSize());

	_collision->calcCollisionTerm(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(), 
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getSize());

	_particledensity->calcParticleDensity(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getParticleType(), _particle->getDensity(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getSize());

	_pressure->calcCSRPressureTerm(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getParticleType(),
		_particle->getPressure(), _particle->getminPressure(),
		_particle->getDensity(), _particle->getBoundary(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getCSRMatrix(), _particle->getCSRIndices(), _particle->getCSRNum(), _particle->getCSROffsets(),
		_particle->getStandardParams(), _particle->getSize());

	_gradient->calcPressureGradient(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(), _particle->getPressure(), _particle->getminPressure(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getStandardParams(), _particle->getSize());

	_move->ModifyParams(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(),
		_particle->getSize());

	_particle->Convert(_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getPressure(), _particle->getminPressure(),
		_particle->getDensity(), _particle->getParticleType(),
		_particle->getBoundary(), 
		_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
		_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
		_particle->gethAcceleration_x(), _particle->gethAcceleration_y(), _particle->gethAcceleration_z(),
		_particle->gethPressure(), _particle->gethminPressure(),
		_particle->gethDensity(), _particle->gethParticleType(),
		_particle->gethBoundary(),
		_particle->getSize());

}

void HighOrdersimulation()
{

	_neighbor->calcNeighborStorageCSR(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getParticleType(), _particle->getDensity(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getCSRMatrix(), _particle->getCSRIndices(), _particle->getCSRNum(), _particle->getCSROffsets(),
		_particle->getSize());


	_particledensity->calcParticleDensity(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getParticleType(), _particle->getDensity(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getSize());

	_viscos->calcViscos(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(),
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getStandardParams(), _particle->getSize());


	/*_viscos->calcImplicitViscosCSR(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(),
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getCSRIndices(), _particle->getCSRNum(), _particle->getCSROffsets(),
		_particle->getStandardParams(), _particle->getSize());*/

	_move->HighOrderUpdataParams(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(),
		_particle->getSize());


	_collision->HighOrdercalcCollisionTerm(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getSize());


	_pressure->HighOrdercalcCSRPressureTerm(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getParticleType(),
		_particle->getPressure(), _particle->getminPressure(),
		_particle->getDensity(), _particle->getBoundary(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getlapNeighborIndex(), _particle->getlapNeighborNum(),
		_particle->getCSRMatrix(), _particle->getCSRIndices(), _particle->getCSRNum(), _particle->getCSROffsets(),
		_particle->getStandardParams(), _particle->getSize());

	_gradient->HighOrdercalcPressureGradient(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(), _particle->getPressure(), _particle->getminPressure(),
		_particle->getNeighborIndex(), _particle->getNeighborNum(),
		_particle->getStandardParams(), _particle->getSize());

	_move->HighOrderModifyParams(
		_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getParticleType(),
		_particle->getSize());

	_particle->Convert(_particle->getPositions_x(), _particle->getPositions_y(), _particle->getPositions_z(),
		_particle->getVelocity_x(), _particle->getVelocity_y(), _particle->getVelocity_z(),
		_particle->getAcceleration_x(), _particle->getAcceleration_y(), _particle->getAcceleration_z(),
		_particle->getPressure(), _particle->getminPressure(),
		_particle->getDensity(), _particle->getParticleType(),
		_particle->getBoundary(),
		_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
		_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
		_particle->gethAcceleration_x(), _particle->gethAcceleration_y(), _particle->gethAcceleration_z(),
		_particle->gethPressure(), _particle->gethminPressure(),
		_particle->gethDensity(), _particle->gethParticleType(),
		_particle->gethBoundary(),
		_particle->getSize());

}

int main(int argc, char* argv[])
{

	double time = 0.0;
	unsigned int fileNumber = 0;
	double calcTime = 0.0;
	cudaEvent_t start, stop;
	
	cusparseHandle_t cusparse;
	cusparseMatDescr_t matDescr;
	
	
	Init(cusparse, matDescr);

	std::wcout << "Start" << std::endl;
	getchar();

	while (1){

		float elapseTime = 0.0;


		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);


		//simulation();

		HighOrdersimulation();

		time += dt;
		fileNumber++;
        std::cout << fileNumber << std::endl;
		if (fileNumber % 100 == 0){


			_output->WritingData_vtk(
					_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
					_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
					_particle->gethPressure(), _particle->gethParticleType(), _particle->gethBoundary(),
					fileNumber, _particle->getSize());

			_output->WritingData_txt(_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
				_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
				_particle->gethPressure(), _particle->gethParticleType(), _particle->gethDensity(), _particle->gethBoundary(),
				fileNumber, time, _particle->getSize());

			_output->WritingPressureGradient(_particle->gethPositions_y(),
				_particle->gethPressure(), _particle->gethParticleType(),
				fileNumber, _particle->getSize());

		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapseTime, start, stop);

		elapseTime *= 1.0E-3;

		calcTime += elapseTime;

        std::wcout << "totalParticle::" << _particle->getSize() << std::endl;
		std::wcout << "Step::" << fileNumber << "||Time::" << time << std::endl;
		std::wcout << "Calculation time:" << elapseTime << "(s)" << "totalTime:" << calcTime << "(s)" << std::endl;
		std::wcout << "---------------------------------------------------" << std::endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		if (time >= Fin_time){
			
			_output->WritingData_vtk(
				_particle->gethPositions_x(), _particle->gethPositions_y(), _particle->gethPositions_z(),
				_particle->gethVelocity_x(), _particle->gethVelocity_y(), _particle->gethVelocity_z(),
				_particle->gethPressure(), _particle->gethParticleType(), _particle->gethBoundary(),
				fileNumber, _particle->getSize());

			break; 
		}

	}

	delete _particle;
	delete _neighbor;
	delete _particledensity;
    delete _surface;
	delete _viscos;
	delete _move;
	delete _collision;
	delete _pressure;
	delete _gradient;

	cusparseDestroy(cusparse);

	std::cout << "ok" << std::endl;

	getchar();

	return 0;

}