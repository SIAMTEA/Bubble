#pragma once

#include "VecSoa.cuh"

class Output
{

public:

	virtual ~Output(void);

	void WritingData_vtk(
            hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z,
		    hVec1dSoa& vel_x, hVec1dSoa& vel_y, hVec1dSoa& vel_z,
		    hVec1dSoa& press, hVec1iSoa& type, hVec1iSoa& boudanry, 
		    unsigned int fileNumber, unsigned int totalParticle);


	void WritingData_txt(hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z,
		hVec1dSoa& vel_x, hVec1dSoa& vel_y, hVec1dSoa& vel_z,
		hVec1dSoa& press, hVec1iSoa& type, hVec1dSoa& dens, hVec1iSoa& boundary,
		unsigned int fileNumber, double timer, unsigned int totalParticle);


	void WritingPressureGradient(
		hVec1dSoa& pos_y,
		hVec1dSoa& press, hVec1iSoa& type,
		unsigned int fileNumber, unsigned int totalParticle);

};