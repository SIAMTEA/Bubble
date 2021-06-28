#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <ostream>
#include <string>

#include "common.cuh"
#include "VecSoa.cuh"
#include "Output.cuh"
#include "Some.cuh"

Output::~Output(void)
{


}

void Output::WritingData_vtk(
        hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z,
	    hVec1dSoa& vel_x, hVec1dSoa& vel_y, hVec1dSoa& vel_z,
	    hVec1dSoa& press, hVec1iSoa& type, hVec1iSoa& boundary,
	    unsigned int fileNumber, unsigned int totalParticle)
{

	int i;
	char fileName[1024];
	double absoluteValueOfVelocity;

	sprintf(fileName, "./Output\\vtk/param_%04d.vtu", fileNumber);
    std::ofstream file(fileName);

	file << "<?xml version='1.0' encoding='UTF-8'?>" << std::endl;
	file << "<VTKFile xmlns='VTK' byte_order='LittleEndian' version='0.1' type='UnstructuredGrid'>" << std::endl;
	file << "<UnstructuredGrid>" << std::endl;
	file << "<Piece NumberOfCells='" << totalParticle << "' " << "NumberOfPoints='" << totalParticle << "'>" << std::endl;

	//Position output
	file << "<Points>" << std::endl;
	file << "<DataArray NumberOfComponents='3' type='Float32' Name='Position' format='ascii'>" << std::endl;
	for (int i = 0; i < totalParticle; i++){

		file << pos_x[i] << " " << pos_y[i] << " " << pos_z[i] << std::endl;

	}
	file << "</DataArray>" << std::endl;

	file << "</Points>" << std::endl;

	//type output
	file << "<PointData>" << std::endl;
	file << "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleType' format='ascii'>" << std::endl;
	for (int i = 0; i < totalParticle; i++){

		file << type[i] << std::endl;

	}
	file << "</DataArray>" << std::endl;


	//tensionBoundary output
	file << "<DataArray NumberOfComponents='1' type='Int32' Name='tensionBoundary' format='ascii'>" << std::endl;
	for (int i = 0; i < totalParticle; i++){

		file << boundary[i] << std::endl;

	}
	file << "</DataArray>" << std::endl;

	//velocity output
	file << "<DataArray NumberOfComponents='3' type='Float32' Name='Velocity' format='ascii'>" << std::endl;
	for(int i = 0; i < totalParticle; i++){


	file << vel_x[i] <<" " << vel_y[i] <<" " << vel_z[i] << std::endl;

	}
	file << "</DataArray>" <<std::endl;


	//pressure output
	file << "<DataArray NumberOfComponents='1' type='Float32' Name='Pressure' format='ascii'>" << std::endl;
	for (int i = 0; i < totalParticle; i++){

		file << press[i] << std::endl;

	}
	file << "</DataArray>" << std::endl;
	file << "</PointData>" << std::endl;

	file << "<Cells>" << std::endl;
	file << "<DataArray type='Int32' Name='connectivity' format='ascii'>" << std::endl;
	for (int i = 0; i < totalParticle; i++){

		file << i << std::endl;

	}
	file << "</DataArray>" << std::endl;

	file << "<DataArray type='Int32' Name='offsets' format='ascii'>" << std::endl;
	for (int i = 0; i < totalParticle; i++){

		file << i + 1 << std::endl;

	}
	file << "</DataArray>" << std::endl;

	file << "<DataArray type='UInt8' Name='types' format='ascii'>" << std::endl;
	for (int i = 0; i < totalParticle; i++){

		file << 1 << std::endl;

	}
	file << "</DataArray>" << std::endl;

	file << "</Cells>" << std::endl;
	file << "</Piece>" << std::endl;
	file << "</UnstructuredGrid>" << std::endl;
	file << "</VTKFile>" << std::endl;

	

	file.close();
}

void Output::WritingData_txt(hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z,
	hVec1dSoa& vel_x, hVec1dSoa& vel_y, hVec1dSoa& vel_z,
	hVec1dSoa& press, hVec1iSoa& type, hVec1dSoa& dens, hVec1iSoa& boundary,
	unsigned int fileNumber, double timer, unsigned int totalParticle)
{

	char fileName[1024];
	double absoluteValueOfVelocity;
	//FILE *file;

    //used gcc
	sprintf(fileName, "./Output\\txt/param_%04d.txt", fileNumber);
    std::ofstream file(fileName);
    file << "time:" << timer << std::endl;
    for(int i = 0; i < totalParticle; i++){

        file << i << " " << type[i] << " " << pos_x[i] << " " << pos_y[i] << " " << pos_z[i] << std::endl;

    }

}

void Output::WritingPressureGradient(
	hVec1dSoa& pos_y,
	hVec1dSoa& press, hVec1iSoa& type,
	unsigned int fileNumber, unsigned int totalParticle)
{

	char fileName[1024];
	double absoluteValueOfVelocity;
	double out_time = 0.0;
	//FILE *file;

    sprintf(fileName, "./Output\\csv/gradient_%04d.csv", fileNumber);
    std::ofstream file(fileName);
	for (int i = 0; i < totalParticle; i++){
		if (type[i] == FLUID){
			file << press[i] << "," << pos_y[i] << std::endl;
		}
	}
}
