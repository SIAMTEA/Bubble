#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "VecSoa.cuh"
#include "Some.cuh"
#include "Particle.cuh"
#include "common.cuh"

#include <iostream>
#include <fstream>

Particle::Particle(const double re, const double lap_re, const double fpd,
                   const int fluid, const int wall, const int dwall, const int air, const int max_neighbor)  //air 6/17
:_r_e(re), _lap_r_e(lap_re), _fpd(fpd), _fluid(fluid), _wall(wall), _dwall(dwall), _air(air), _max_neighbor(max_neighbor)  //air 6/17
{


}


Particle::~Particle(void)
{

}


void Particle::InitParticlePosition(hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z)
{

	double ix, iy, iz;

	double nx, ny, nz;

	double x, y, z;

    double pre_x, pre_z;

    double pre_distance, distance;
	
	bool flag = false;

	nx = (double)(X_WIDTH / _fpd) + 5;
	ny = (double)(Y_WIDTH / _fpd) + 5;
	nz = (double)(Z_WIDTH / _fpd) + 5;


	for (ix = -4; ix < nx; ix++){
		for (iy = -4; iy < ny; iy++){
			for (iz = -4; iz < nz; iz++){


				x = _fpd*(double)(ix);
				y = _fpd*(double)(iy);
				z = _fpd*(double)(iz);

				flag = false;


				/* dummy wall region */
				if (((x>-5.0*_fpd) && (x <= X_WIDTH + 4.0*_fpd)) && ((y>0.0 - 5.0*_fpd) && (y <= Y_WIDTH)) && ((z>0.0 - 5.0*_fpd) && (z <= Z_WIDTH + 4.0*_fpd))){
                
					flag = true;

				}

				//* empty region */
				if ((((x>-0.0001) && (x <= X_WIDTH)) && (y >= WATER_LEVEL)) && ((z>-0.0001) && (z <= Z_WIDTH))){

					flag = false;

				}


				if (flag == true){

					pos_x.push_back(x);
					pos_y.push_back(y);
					pos_z.push_back(z);

				}
			}
		}
	}
}


void Particle::InitParticleType_and_FirstVel(hVec1dSoa& pos_x, hVec1dSoa& pos_y, hVec1dSoa& pos_z,
								             hVec1dSoa& vel_x, hVec1dSoa& vel_y, hVec1dSoa& vel_z,
								             hVec1iSoa& Type, const int totalParticle)
{

	int ix, iy, iz;
	int nx, ny, nz;

	double x, y, z;
	int type = _dwall;

    double pre_x, pre_z;
    double pre_distance, distance;


	for (int i = 0; i < totalParticle; i++){

		x = pos_x[i];
		y = pos_y[i];
		z = pos_z[i];

		vel_x.push_back(0.0);
		vel_y.push_back(0.0);
		vel_z.push_back(0.0);

		//中心から距離を求める
		pre_distance = (x - X_CENTER)*(x - X_CENTER) + (y - Y_CENTER)*(y - Y_CENTER) + (z - Z_CENTER)*(z - Z_CENTER);
		distance = sqrt(pre_distance);

		
		if (((x > -(5.0*_fpd)) && (x < (X_WIDTH + 5.0*_fpd))) && (y > -(5.0*_fpd)) && ((z > -(5.0*_fpd)) && (z < (Z_WIDTH + 5.0*_fpd)))){
		
			Type.push_back(_dwall);

		}

		if (((x > -(3.0*_fpd)) && (x <= (X_WIDTH + 2.0*_fpd))) && (y > -(3.0*_fpd)) && ((z > -(3.0*_fpd)) && (z <= (Z_WIDTH + 2.0*_fpd)))){  //変更　3.0→2.0

			Type[i] = _wall;

		}


		if (((x>-(1.0*_fpd)) && (x <= (X_WIDTH))) && (y>-(1.0*_fpd)) && ((z>-(1.0*_fpd)) && (z <= (Z_WIDTH)))){

			Type[i] = _fluid;

		}

		//粒子が中心より上にあり、
		//中心からの距離が、泡の半径内だったら
		if ((y > Y_CENTER) && (distance < bubble_r)){

		Type[i] = _air;  //空気粒子にする

		}

	}

}


void Particle::UpdataResize(
        Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
        Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
        Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
        Vec1dSoa& press, Vec1dSoa& minpress,
        Vec1dSoa& dens, Vec1iSoa& type,
        Vec1iSoa& boundary,
        Vec1uiSoa& neighborIndex, Vec1uiSoa& neighborNum,
        Vec1uiSoa& lapneighborIndex, Vec1uiSoa& lapneighborNum,
        hVec1dSoa& hpos_x, hVec1dSoa& hpos_y, hVec1dSoa& hpos_z,
        hVec1dSoa& hvel_x, hVec1dSoa& hvel_y, hVec1dSoa& hvel_z,
        hVec1dSoa& hacc_x, hVec1dSoa& hacc_y, hVec1dSoa& hacc_z,
        hVec1dSoa& hpress, hVec1dSoa& hminpress,
        hVec1dSoa& hdens, hVec1iSoa& htype,
        hVec1iSoa& hboundary,
	    unsigned int totalParticle)
{

	pos_x.resize(totalParticle);
	pos_y.resize(totalParticle);
	pos_z.resize(totalParticle);

	thrust::copy(hpos_x.begin(), hpos_x.end(), pos_x.begin());
	thrust::copy(hpos_y.begin(), hpos_y.end(), pos_y.begin());
	thrust::copy(hpos_z.begin(), hpos_z.end(), pos_z.begin());

	vel_x.resize(totalParticle, 0.0);
	vel_y.resize(totalParticle, 0.0);
	vel_z.resize(totalParticle, 0.0);

	thrust::copy(hvel_x.begin(), hvel_x.end(), vel_x.begin());
	thrust::copy(hvel_y.begin(), hvel_y.end(), vel_y.begin());
	thrust::copy(hvel_z.begin(), hvel_z.end(), vel_z.begin());

	acc_x.resize(totalParticle, 0.0);
	acc_y.resize(totalParticle, 0.0);
	acc_z.resize(totalParticle, 0.0);

	press.resize(totalParticle, 0.0);
	minpress.resize(totalParticle, 0.0);

	dens.resize(totalParticle, 0.0);
	type.resize(totalParticle, 0);

	thrust::copy(htype.begin(), htype.end(), type.begin());

	boundary.resize(totalParticle, 0);

	neighborIndex.resize(totalParticle*Max_Neighbor, 0);
	neighborNum.resize(totalParticle, 0);
	lapneighborIndex.resize(totalParticle*Max_Neighbor, 0);
	lapneighborNum.resize(totalParticle, 0);

	hacc_x.resize(totalParticle, 0.0);
	hacc_y.resize(totalParticle, 0.0);
	hacc_z.resize(totalParticle, 0.0);

	hpress.resize(totalParticle, 0.0);
	hminpress.resize(totalParticle, 0.0);

	hdens.resize(totalParticle, 0.0);
	hboundary.resize(totalParticle, 0);

}

namespace weight_func
{
	double d_weight(double distance, double re)
	{

		int i = 0;
		double weight_ij = 0.0;

		if (distance >= re){
			weight_ij = 0.0;
		}
		else{

			weight_ij = (re / distance) - 1.0; //original

			//weight_ij = pow((distance / re) - 1.0, 2.0);

		}

		return weight_ij;

	}

    int unit_weight(double distance, double re){

        int weight_ij = 0;

        if(distance >= re){
            weight_ij = 0;
        }else{

            weight_ij = 1;

        }

        return weight_ij;
    }

    int surface_weight(double distance, double re){

        int weight_ij = 0;

        if (distance >= re){

            weight_ij = 0;

        }
        else{

            weight_ij = 1;

        }

        return weight_ij;
    }

    double Potensial_weight(double distance, double re){

        double weight_ij = 0.0;

        double a = (3.0f / 2.0f) * (double)FPD;
        double b = re / 2.0f;

        if(distance < re){

            double pre = distance - re;

            //weight_ij = ((distance - a + b) * pow(pre, 2)) / 3.0f;
            weight_ij = ((a - b - distance) * pow(pre, 2)) / 3.0;

        }else{

            weight_ij = 0.0;

        }

        return  weight_ij;

    }

};


void Particle::calcInitStandardParams(hVec1dSoa& stan)
{

    int ix, iy, iz;
    int iz_start, iz_end;
    double xi, yi, zi;
    double xj, yj, zj;
    double distance, pre_distance;
    double n = 0.0;
    double L = 0.0;
    double lam = 0.0;
    int boundary_st = 0;
    int boundary_st_lap = 0;
    int st = 0;
    int st_lap = 0;
    unsigned int count = 0;

    iz_start = -4;
    iz_end = 5;

    xi = yi = zi = 0.0;

    for (ix = -4; ix < 5; ix++){
        for (iy = -4; iy < 5; iy++){
            for (iz = iz_start; iz < iz_end; iz++){
                if (((ix == 0) && (iy == 0) && (iz == 0))) continue;
                xj = _fpd * (double)(ix);
                yj = _fpd * (double)(iy);
                zj = _fpd * (double)(iz);
                pre_distance = (xj - xi)*(xj - xi) + (yj - yi)*(yj - yi) + (zj - zi)*(zj - zi);
                distance = sqrt(pre_distance);
                n += weight_func::d_weight(distance, _r_e);
                L += weight_func::d_weight(distance, _lap_r_e);
                lam += pre_distance * weight_func::d_weight(distance, _lap_r_e);
                boundary_st += weight_func::surface_weight(distance, _r_e);
                boundary_st_lap += weight_func::surface_weight(distance, _lap_r_e);

            }
        }
    }
    stan.push_back(n);
    stan.push_back(L);
    stan.push_back(lam / L);
    stan.push_back(boundary_st);
    stan.push_back(boundary_st_lap);

    xi = yi = 0.0;
    zi = -4.0*_fpd;

    for (ix = -4; ix < 5; ix++){
        for (iy = -4; iy < 5; iy++){
            for (iz = iz_start; iz < iz_end; iz++){
                if (((ix == 0) && (iy == 0) && (iz == -4))) continue;
                xj = _fpd * (double)(ix);
                yj = _fpd * (double)(iy);
                zj = _fpd * (double)(iz);
                pre_distance = (xj - xi)*(xj - xi) + (yj - yi)*(yj - yi) + (zj - zi)*(zj - zi);
                distance = sqrt(pre_distance);

                st_lap += weight_func::surface_weight(distance, _lap_r_e);
                st += weight_func::surface_weight(distance, _r_e);

            }
        }
    }

    stan.push_back(st);
    stan.push_back(st_lap);

    println(n);
    println(L);
    println(lam/L);
    println(boundary_st);
    println(boundary_st_lap);
    println(st);
    println(st_lap);

}


void Particle::PotentialCoef(hVec1dSoa& Stan)
{

    hVec1dSoa PotentialPos_x;
    hVec1dSoa PotentialPos_y;
    hVec1dSoa PotentialPos_z;
    hVec1iSoa Potentialtype;



    for(int x = 0; x < 9; x++){
        for(int y = 0; y <= 4; y++){
            for(int z = 0; z < 9; z++){

                double xx = x * _fpd;
                double yy = y * _fpd;
                double zz = z * _fpd;

                PotentialPos_x.push_back(xx);
                PotentialPos_y.push_back(yy);
                PotentialPos_z.push_back(zz);

                Potentialtype.push_back(BBB);

            }
        }
    }

    for(int y = 5; y < 9; y++){

        double xx = _fpd * 4.0;
        double yy = y * _fpd;
        double zz = _fpd * 4.0;

        PotentialPos_x.push_back(xx);
        PotentialPos_y.push_back(yy);
        PotentialPos_z.push_back(zz);

        Potentialtype.push_back(AAA);

    }


    double result = 0.0;
    double Coef_ff = 0.0;
    double Coef_fs1 = 0.0;
	double Coef_fs2 = 0.0;
	double angle = CONTACT_ANGLE;// 20.0;// 118.0;// 150.0;

    for(int i = 0; i < PotentialPos_x.size(); i++){
        if(Potentialtype[i] == AAA){

            double b = 0.0;

            for(int j = 0; j < PotentialPos_x.size(); j++){
                if(Potentialtype[j] == BBB){

                    double x = PotentialPos_x[j] - PotentialPos_x[i];
                    double y = PotentialPos_y[j] - PotentialPos_y[i];
                    double z = PotentialPos_z[j] - PotentialPos_z[i];

                    double pre_dis = pow(x, 2) + pow(y, 2) + pow(z, 2);

                    double distance = sqrt(pre_dis);

                    if(distance < _lap_r_e){

                        double a = weight_func::Potensial_weight(distance, _lap_r_e);

                        b+= a;

                    }

                }
            }

            result += b;

        }
    }

    Coef_ff = (2.0f * SURFACETENSION_COEF * pow(_fpd, 2)) / result;

    double rad = angle  * 3.141519f /180.0f;

    Coef_fs1 = (1.0f/2.0f) * (1.0f + cos(rad)) * Coef_ff;

    Stan.push_back(Coef_ff);
    Stan.push_back(Coef_fs1);

    println(Coef_ff);
    println(Coef_fs1);

}


void Particle::Convert(
        Vec1dSoa& pos_x, Vec1dSoa& pos_y, Vec1dSoa& pos_z,
        Vec1dSoa& vel_x, Vec1dSoa& vel_y, Vec1dSoa& vel_z,
        Vec1dSoa& acc_x, Vec1dSoa& acc_y, Vec1dSoa& acc_z,
        Vec1dSoa& press, Vec1dSoa& minpress,
        Vec1dSoa& dens, Vec1iSoa& type,
        Vec1iSoa& boundary, 
        hVec1dSoa& hpos_x, hVec1dSoa& hpos_y, hVec1dSoa& hpos_z,
        hVec1dSoa& hvel_x, hVec1dSoa& hvel_y, hVec1dSoa& hvel_z,
        hVec1dSoa& hacc_x, hVec1dSoa& hacc_y, hVec1dSoa& hacc_z,
        hVec1dSoa& hpress, hVec1dSoa& hminpress,
        hVec1dSoa& hdens, hVec1iSoa& htype,
        hVec1iSoa& hboundary, 
        unsigned int totalParticle)
{

	thrust::copy(pos_x.begin(), pos_x.end(), hpos_x.begin());
	thrust::copy(pos_y.begin(), pos_y.end(), hpos_y.begin());
	thrust::copy(pos_z.begin(), pos_z.end(), hpos_z.begin());

	thrust::copy(vel_x.begin(), vel_x.end(), hvel_x.begin());
	thrust::copy(vel_y.begin(), vel_y.end(), hvel_y.begin());
	thrust::copy(vel_z.begin(), vel_z.end(), hvel_z.begin());

	thrust::copy(press.begin(), press.end(), hpress.begin());

	thrust::copy(dens.begin(), dens.end(), hdens.begin());
	thrust::copy(type.begin(), type.end(), htype.begin());

	thrust::copy(boundary.begin(), boundary.end(), hboundary.begin());

   
}