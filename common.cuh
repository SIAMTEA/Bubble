//単位系はmks

#pragma once

//初期粒子間距離
//#define FPD 1.0	//0.1mm
#define FPD 0.001  //1mm

//水槽サイズ
//#define X_WIDTH 30
//#define Y_WIDTH 20
//#define Z_WIDTH 10
#define X_WIDTH 0.070
#define Y_WIDTH 0.050
#define Z_WIDTH 0.050

//水位
//#define WATER_LEVEL 15
#define WATER_LEVEL 0.030

//中心
#define X_CENTER 0.035
#define Y_CENTER 0.010
#define Z_CENTER 0.025

//泡の半径
#define bubble_r 0.011 //10mm

//影響半径
#define r_e 2.1*FPD      //グラディエント用影響半径
#define lap_r_e 3.1*FPD  //ラプラシアン用影響半径

//粒子タイプ
#define FLUID 0           //液体
#define WALL 1            //壁
#define D_WALL 2          //ダミー壁
#define AIR 3             //空気
#define GHOST -1		  //計算対象外用


//自由表面粒子タイプ
#define Surface_Particle 1   //自由表面粒子
#define Inner_Particle 0     //内部粒子
#define External_Particle -1 //外部粒子


//ポテンシャル用
#define AAA 0	//ポテンシャル力用
#define BBB 1	//ポテンシャル力用
#define CCC 2


//近傍粒子探索
#define blockSized 128							    //探索用の小領域
#define BlockSized blockSized*blockSized*blockSized //探索用分割範囲
#define MaxParticle 300000						    //予想される最大粒子数(特に理由ない)
#define Max_Neighbor 300						    //予想される最大近傍粒子数(とりあえず300あればok)

//時間関連
//#define dt 1.0E-2 //タイムステップ
#define dt 1.0E-5 //6/17
#define Fin_time 30.0//10.0   //終了時刻
#define STOP_TIME 30.0

//パラメータ(Fは水、Vは粘性物質)
//動粘性係数([cSt(mm^2/s)])
#define KINEMATIC_VISCOSITY_COEF 1.00E-6          //水の動粘性係数(水温による若干の変化はあるが現在は対象外)
#define AIR_KINEMATIC_VISCOSITY_COEF 3.60E-8      //空気の動粘性係数

//密度([kg/m^3])
#define DENSITY 1000.0    //水の密度(水温による若干の変化はあるが現在は対象外)
#define AIR_DENSITY 500.0    //空気の密度

//粘性係数([Pa.s]
#define VISCOSITY_COEF 1.00E-3      //水の粘性係数(水温による若干の変化はあるが現在は対象外)
#define AIR_VISCOSITY_COEF 1.80E-5  //空気の粘性係数

//表面張力係数([N/s])
#define SURFACETENSION_COEF 72.8E-3//10.8E-3   //水と塞栓剤の表面張力係数

//重力加速度([m/s^2])
#define G_X 0.0  //x方向
#define G_Y -9.8 //y方向
#define G_Z 0.0  //z方向

//次元数
#define DIM 3

//衝突判定用距離限界
#define COLLISION_LIMIT 0.5*FPD

//衝突緩和係数
#define COLLISION_RATE 1.2  //(1.0 + 0.2)

//圧力計算用緩和係数
#define PRESSURE_RELAX_COEF 0.02

//水の圧縮率
#define COMPRESSIBILITY (0.45E-9)

//ディリクレ境界条件
#define DIRICHLET 0.97

//濡れ性（ポテンシャル）
#define w_boundary 1
#define other_particle 0
#define slide_boundary 2

//BiCGSTAB
#define ERROR pow(2.0, -50)
#define MAX_ITER 100000

#define CONTACT_ANGLE 90.0
