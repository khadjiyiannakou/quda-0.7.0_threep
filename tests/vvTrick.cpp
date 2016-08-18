#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#ifdef QMP_COMMS
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <gauge_qio.h>
#include <gsl/gsl_rng.h>

#define MAX(a,b) ((a)>(b)?(a):(b))


//#define	TESTPOINT
//#define	RANDOM_CONF
#define	CROSSCHECK

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <contractQuda.h>
#include <cufft.h>

//#include <randCuda.h>

extern bool			tune;
extern int			device;
extern QudaDslashType		dslash_type;
extern int			xdim;
extern int			ydim;
extern int			zdim;
extern int			tdim;
extern int			Lsdim;
extern int			numberHP;
extern int			nConf;
extern int			numberLP;
extern int			MaxP;
extern int			gridsize_from_cmdline[];
extern QudaReconstructType	link_recon;
extern QudaPrecision		prec;
extern QudaReconstructType	link_recon_sloppy;
extern QudaPrecision		prec_sloppy;
extern QudaInverterType		inv_type;
extern int			multishift;			// whether to test multi-shift or standard solver

extern char			latfile[];

extern void			usage(char**);

int	genDataArray	(const int nSources, int *dataLP, int &flag)
{
	int	count = 0, power = 128, accum = nSources;

	if	(nSources < power)
	{
		dataLP[count]	= nSources;
		count		= 1;
		flag		= 0;

		return	count;
	}

	do
	{
		dataLP[count]	= power;

		accum	-= power;
		power	*= 2;
		count++;
	}	while	(accum > 0);

	if	(accum == 0)
	{
		flag	 = 0;
		return	count;
	}
	else
	{
		flag	 = 1;
		accum	+= power/2;

		dataLP[count-1]	 = accum;

		return	count;
	}
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

void	genRandomSource	(void *spinorIn, QudaInvertParam *inv_param, gsl_rng *rNum)
{
#ifdef	TESTPOINT
	if	(inv_param->cpu_prec == QUDA_SINGLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		if	(comm_rank() == 0)
			((float*) spinorIn)[18]		 = 1.;		//t-Component
	}
	else if	(inv_param->cpu_prec == QUDA_DOUBLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		if	(comm_rank() == 0)
			((double*) spinorIn)[18]	 = 1.;
	}
#else
	if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION) 
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		for	(int i = 0; i<V*12; i++)
		{
			int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);
	
			switch	(randomNumber)
			{
				case 0:
	
				((float*) spinorIn)[i*2]	= 1.;
				break;

				case 1:
	
				((float*) spinorIn)[i*2]	= -1.;
				break;
	
				case 2:
	
				((float*) spinorIn)[i*2+1]	= 1.;
				break;
	
				case 3:
	
				((float*) spinorIn)[i*2+1]	= -1.;
				break;
			}
		}
	}
	else
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		for	(int i = 0; i<V*12; i++)
		{
			int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);
	
			switch	(randomNumber)
			{
				case 0:
	
				((double*) spinorIn)[i*2]	= 1.;
				break;
	
				case 1:
	
				((double*) spinorIn)[i*2]	= -1.;
				break;
	
				case 2:
	
				((double*) spinorIn)[i*2+1]	= 1.;
				break;
	
				case 3:
	
				((double*) spinorIn)[i*2+1]	= -1.;
				break;
			}
		}
	}
#endif
}

void	reOrder	(double *array1, double *array2, const int arrayOffset)
{
	if	(array1 != array2)
	{
		for	(int i = 0; i<V*arrayOffset; i++)
			array2[i]	= 0.;
	}

	for	(int i = 0; i<V*arrayOffset; i++)
	{
		int	pointT		=	i/arrayOffset;
		int	offset		=	i%arrayOffset;
		int	oddBit		=	0;

		if	(pointT >= V/2)
		{
			pointT	-= V/2;
			oddBit	 = 1;
		}

		int za		 = pointT/(xdim/2);
		int x1h		 = pointT - za*(xdim/2);
		int zb		 = za/ydim;
		int x2		 = za - zb*ydim;
		int x4		 = zb/zdim;
		int x3		 = zb - x4*zdim;
		int x1odd	 = (x2 + x3 + x4 + oddBit) & 1;
		int x1		 = 2*x1h + x1odd;
		int X		 = x1 + xdim*(x2 + ydim*(x3 + zdim*x4));
		X		*= arrayOffset;
		X		+= offset;

		if	(array1 != array2)
			array2[X]	= array1[i];
		else
		{
			double	temp	 = array2[X];
			array2[X]	 = array1[i];
			array1[i]	 = temp;
		}
	}

	return;
}

int	doCudaFFT	(const int keep, void *cnRes_vv, void *cnRes_gv, void **cnD_vv, void **cnD_gv, void **cnC_vv, void **cnC_gv, QudaPrecision prec)
{
	static cufftHandle	fftPlan;
	static int		init = 0;
	int			nRank[3]	 = {xdim, ydim, zdim};
	const int		Vol		 = xdim*ydim*zdim;

	static cudaStream_t	streamCuFFT;

	if	(!keep)
	{
		if	(init)
		{
			cufftDestroy		(fftPlan);
			cudaStreamDestroy	(streamCuFFT);
		}

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}

		init	 = 0;

		return	0;
	}

	if	(prec == QUDA_DOUBLE_PRECISION)
	{

	if	(!init)
	{
		cudaStreamCreate	(&streamCuFFT);

		if	(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_Z2Z, 16*tdim) != CUFFT_SUCCESS)
		{
			printf	("Error in the FFT!!!\n");
			return 1;
		}

		cufftSetCompatibilityMode	(fftPlan, CUFFT_COMPATIBILITY_NATIVE);
		cufftSetStream			(fftPlan, streamCuFFT);

		printfQuda	("Synchronizing\n");

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}

		init	 = 1;
	}
	else
		printfQuda	("CuFFT plan already initialized\n");

	void	*ctrnS;

	if	((cudaMalloc(&ctrnS, sizeof(double)*32*Vol*tdim)) == cudaErrorMemoryAllocation)
	{
		printf	("Error allocating memory for contraction results in GPU.\n");
		exit	(0);
	}
	cudaMemcpy	(ctrnS, cnRes_vv, sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);

	if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		printf  ("Error executing FFT!!!\n");
		return 1;
	}
	
	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return 1;
	}
	
	cudaMemcpy	(cnRes_vv, ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

	cudaMemcpy	(ctrnS, cnRes_gv, sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);

	if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		printf  ("Error executing FFT!!!\n");
		return 1;
	}
	
	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return 1;
	}
	
	cudaMemcpy	(cnRes_gv, ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

	for	(int mu=0; mu<4; mu++)
	{
		cudaMemcpy	(ctrnS, cnD_gv[mu], sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnD_gv[mu], ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

		cudaMemcpy	(ctrnS, cnD_vv[mu], sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnD_vv[mu], ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

		cudaMemcpy	(ctrnS, cnC_gv[mu], sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnC_gv[mu], ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

		cudaMemcpy	(ctrnS, cnC_vv[mu], sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnC_vv[mu], ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);
	}

	cudaFree	(ctrnS);

	}
	else if	(prec == QUDA_SINGLE_PRECISION)
	{

	if	(!init)
	{
		cudaStreamCreate	(&streamCuFFT);

		if	(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_C2C, 16*tdim) != CUFFT_SUCCESS)
		{
			printf	("Error in the FFT!!!\n");
			return 1;
		}

		cufftSetCompatibilityMode	(fftPlan, CUFFT_COMPATIBILITY_NATIVE);
		cufftSetStream			(fftPlan, streamCuFFT);

		printfQuda	("Synchronizing\n");

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}

		init	 = 1;
	}
	else
		printfQuda	("CuFFT plan already initialized\n");

	void	*ctrnS;

	if	((cudaMalloc(&ctrnS, sizeof(float)*32*Vol*tdim)) == cudaErrorMemoryAllocation)
	{
		printf	("Error allocating memory for contraction results in GPU.\n");
		exit	(0);
	}
	cudaMemcpy	(ctrnS, cnRes_vv, sizeof(float)*32*Vol*tdim, cudaMemcpyHostToDevice);

	if	(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		printf  ("Error executing FFT!!!\n");
		return 1;
	}
	
	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return 1;
	}
	
	cudaMemcpy	(cnRes_vv, ctrnS, sizeof(float)*32*Vol*tdim, cudaMemcpyDeviceToHost);

	cudaMemcpy	(ctrnS, cnRes_gv, sizeof(float)*32*Vol*tdim, cudaMemcpyHostToDevice);

	if	(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		printf  ("Error executing FFT!!!\n");
		return 1;
	}
	
	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return 1;
	}
	
	cudaMemcpy	(cnRes_gv, ctrnS, sizeof(float)*32*Vol*tdim, cudaMemcpyDeviceToHost);

	for	(int mu=0; mu<4; mu++)
	{
		cudaMemcpy	(ctrnS, cnD_gv[mu], sizeof(float)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnD_gv[mu], ctrnS, sizeof(float)*32*Vol*tdim, cudaMemcpyDeviceToHost);

		cudaMemcpy	(ctrnS, cnD_vv[mu], sizeof(float)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnD_vv[mu], ctrnS, sizeof(float)*32*Vol*tdim, cudaMemcpyDeviceToHost);

		cudaMemcpy	(ctrnS, cnC_gv[mu], sizeof(float)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnC_gv[mu], ctrnS, sizeof(float)*32*Vol*tdim, cudaMemcpyDeviceToHost);

		cudaMemcpy	(ctrnS, cnC_vv[mu], sizeof(float)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnC_vv[mu], ctrnS, sizeof(float)*32*Vol*tdim, cudaMemcpyDeviceToHost);
	}

	cudaFree	(ctrnS);

	}

	return	0;
}

void	dumpData	(int nSols, const char *Pref, int **mom, void *cnRes_vv, void *cnRes_gv, void **cnD_vv, void **cnD_gv, void **cnC_vv, void **cnC_gv, const int iDiv, QudaPrecision prec)
{
	FILE		*sfp;
	FILE		*sfpMu;

	char		file_name[256];
	int		nSol = 1;

	const int	Vol = xdim*ydim*zdim;

	if	(iDiv)				//*	Y recuerda cambiar el nSols de abajo por nSol!!!
		nSol	 = nSols;		//*
	else					//*
		nSol	 = 1;			//*

	if	(prec == QUDA_DOUBLE_PRECISION)
	{
		sprintf(file_name, "dOp.loop.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfp = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		sprintf(file_name, "LpsDw.loop.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfpMu = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		for	(int ip=0; ip<Vol; ip++)
			for	(int wt=0; wt<tdim; wt++)
			{
				if	((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= MaxP)
				{
			        	int	rT	 = wt+comm_coord(3)*tdim;

					for	(int gm=0; gm<16; gm++)
					{										// TEST
						fprintf (sfp, "%03d %02d %02d %+d %+d %+d %+.10le %+.10le\n", nSols, rT, gm, mom[ip][0], mom[ip][1], mom[ip][2],
							((double2*)cnRes_gv)[ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2*)cnRes_gv)[ip+Vol*wt+Vol*tdim*gm].y/((double) nSol));

						for	(int mu = 0; mu < 4; mu++)
						{
							fprintf (sfpMu, "%03d %02d %d %02d %+d %+d %+d %+.10le %+.10le %+.10le %+.10le\n", nSols, rT, mu, gm, mom[ip][0], mom[ip][1], mom[ip][2],
								((double2**)cnD_gv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2**)cnD_gv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((double) nSol),
								((double2**)cnC_gv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2**)cnC_gv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((double) nSol));
						}

						fflush  (sfp);
						fflush  (sfpMu);
					}
				}
			}
	}
	else if	(prec == QUDA_SINGLE_PRECISION)
	{
		sprintf(file_name, "dOp.lpSg.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfp = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		sprintf(file_name, "LpsDw.lpSg.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfpMu = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		for	(int ip=0; ip<Vol; ip++)
			for	(int wt=0; wt<tdim; wt++)
			{
				if	((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= MaxP)
				{
			        	int	rT	 = wt+comm_coord(3)*tdim;

					for	(int gm=0; gm<16; gm++)
					{										// TEST
						fprintf (sfp, "%03d %02d %02d %+d %+d %+d %+.7e %+.7e\n", nSols, rT, gm, mom[ip][0], mom[ip][1], mom[ip][2],
							((float2*)cnRes_gv)[ip+Vol*wt+Vol*tdim*gm].x/((float) nSol), ((float2*)cnRes_gv)[ip+Vol*wt+Vol*tdim*gm].y/((float) nSol));

						for	(int mu = 0; mu < 4; mu++)
						{
							fprintf (sfpMu, "%03d %02d %d %02d %+d %+d %+d %+.7e %+.7e %+.7e %+.7e\n", nSols, rT, mu, gm, mom[ip][0], mom[ip][1], mom[ip][2],
								((float2**)cnD_gv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((float) nSol), ((float2**)cnD_gv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((float) nSol),
								((float2**)cnC_gv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((float) nSol), ((float2**)cnC_gv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((float) nSol));
						}
					}

					fflush  (sfp);
					fflush  (sfpMu);
				}
			}
	}

	fclose(sfp);
	fclose(sfpMu);

	if	(prec == QUDA_DOUBLE_PRECISION)
	{
		sprintf(file_name, "Scalar.loop.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfp = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		sprintf(file_name, "Loops.loop.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfpMu = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		for	(int ip=0; ip<Vol; ip++)
			for	(int wt=0; wt<tdim; wt++)
			{
				if	((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= MaxP)
				{
			        	int	rT	 = wt+comm_coord(3)*tdim;
	
					for	(int gm=0; gm<16; gm++)
					{										// TEST
						fprintf (sfp, "%03d %02d %02d %+d %+d %+d %+.10le %+.10le\n", nSols, rT, gm, mom[ip][0], mom[ip][1], mom[ip][2],
							((double2*)cnRes_vv)[ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2*)cnRes_vv)[ip+Vol*wt+Vol*tdim*gm].y/((double) nSol));

						for	(int mu = 0; mu < 4; mu++)
						{
							fprintf (sfpMu, "%03d %02d %d %02d %+d %+d %+d %+.10le %+.10le %+.10le %+.10le\n", nSols, rT, mu, gm, mom[ip][0], mom[ip][1], mom[ip][2],
								((double2**)cnD_vv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2**)cnD_vv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((double) nSol),
								((double2**)cnC_vv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2**)cnC_vv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((double) nSol));
						}
					}
				}

				fflush  (sfp);
				fflush  (sfpMu);
			}
	}
	else if	(prec == QUDA_SINGLE_PRECISION)
	{
		sprintf(file_name, "Scalar.lpSg.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfp = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		sprintf(file_name, "Loops.lpSg.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfpMu = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		for	(int ip=0; ip<Vol; ip++)
			for	(int wt=0; wt<tdim; wt++)
			{
				if	((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= MaxP)
				{
			        	int	rT	 = wt+comm_coord(3)*tdim;
	
					for	(int gm=0; gm<16; gm++)
					{										// TEST
						fprintf (sfp, "%03d %02d %02d %+d %+d %+d %+.7e %+.7e\n", nSols, rT, gm, mom[ip][0], mom[ip][1], mom[ip][2],
							((float2*)cnRes_vv)[ip+Vol*wt+Vol*tdim*gm].x/((float) nSol), ((float2*)cnRes_vv)[ip+Vol*wt+Vol*tdim*gm].y/((float) nSol));

						for	(int mu = 0; mu < 4; mu++)
						{
							fprintf (sfpMu, "%03d %02d %d %02d %+d %+d %+d %+.7e %+.7e %+.7e %+.7e\n", nSols, rT, mu, gm, mom[ip][0], mom[ip][1], mom[ip][2],
								((float2**)cnD_vv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((float) nSol), ((float2**)cnD_vv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((float) nSol),
								((float2**)cnC_vv)[mu][ip+Vol*wt+Vol*tdim*gm].x/((float) nSol), ((float2**)cnC_vv)[mu][ip+Vol*wt+Vol*tdim*gm].y/((float) nSol));
						}
					}
				}

				fflush  (sfp);
				fflush  (sfpMu);
			}
	}

	fclose(sfp);
	fclose(sfpMu);

	return;
}

int	main	(int argc, char **argv)
{
	int	i, k;
	double	precision	 = 4e-10;
	int	iteraHP		 = 40000;
	int	iteraLP		 = 15;

	char	name[16];

	int	dataLP[16];
	int	maxSources, flag;

//	curandStateMRG32k3a	*rngStat;

	for	(i =1;i < argc; i++)
	{
		if	(process_command_line_option(argc, argv, &i) == 0)
			continue;
    
		printf	("ERROR: Invalid option:%s\n", argv[i]);
		usage	(argv);
	}

	// initialize QMP or MPI
#if defined(QMP_COMMS)
	QMP_thread_level_t tl;
	QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
#elif defined(MPI_COMMS)
	MPI_Init(&argc, &argv);
#endif

	initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);

	maxSources	 = genDataArray (numberLP, dataLP, flag);

	printfQuda	("Will dump %d files in ", maxSources);

	for	(i=0; i<maxSources; i++)
		printfQuda	("%d ", dataLP[i]);


	printfQuda	("clusters.\n");

	//	Initialize random number generator

	int	myRank	 = comm_rank();
	int	seed;

	if	(myRank == 0)
		seed	 = (int) clock();

	MPI_Bcast	(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

//	setUpRandomCuda	(state, seed, myRank, 256, 64);


	//	Starts Quda initialization

	if	(prec_sloppy == QUDA_INVALID_PRECISION)
		prec_sloppy		 = prec;

	if	(link_recon_sloppy == QUDA_RECONSTRUCT_INVALID)
		link_recon_sloppy	 = link_recon;


  // *** QUDA parameters begin here.


	dslash_type				 = QUDA_TWISTED_MASS_DSLASH;
//	dslash_type				 = QUDA_TWISTED_CLOVER_DSLASH;

	QudaPrecision cpu_prec			 = QUDA_DOUBLE_PRECISION;
	QudaPrecision cuda_prec			 = prec;
	QudaPrecision cuda_prec_sloppy		 = prec_sloppy;

	QudaGaugeParam gauge_param		 = newQudaGaugeParam();
	QudaInvertParam inv_param		 = newQudaInvertParam();

	gauge_param.X[0]			 = xdim;
	gauge_param.X[1]			 = ydim;
	gauge_param.X[2]			 = zdim;
	gauge_param.X[3]			 = tdim;

	gauge_param.anisotropy			 = 1.0;
	gauge_param.type			 = QUDA_WILSON_LINKS;
	gauge_param.gauge_order			 = QUDA_QDP_GAUGE_ORDER;
	gauge_param.t_boundary			 = QUDA_ANTI_PERIODIC_T;

	gauge_param.cpu_prec			 = cpu_prec;
	gauge_param.cuda_prec			 = cuda_prec;
	gauge_param.reconstruct			 = link_recon;
	gauge_param.cuda_prec_sloppy		 = cuda_prec_sloppy;
	gauge_param.reconstruct_sloppy		 = link_recon_sloppy;
	gauge_param.cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	gauge_param.reconstruct_precondition	 = link_recon_sloppy;
	gauge_param.gauge_fix			 = QUDA_GAUGE_FIXED_NO;

	inv_param.dslash_type = dslash_type;

	double mass = -2.;
	inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

	if	(dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
	{
		inv_param.mu = 0.003;
		inv_param.twist_flavor = QUDA_TWIST_MINUS;
	}

	inv_param.solution_type		 = QUDA_MAT_SOLUTION;
	inv_param.solve_type		 = QUDA_NORMOP_PC_SOLVE;
//	inv_param.solve_type		 = QUDA_DIRECT_PC_SOLVE;
	if	(inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH)
		inv_param.matpc_type		 = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
	else
		inv_param.matpc_type		 = QUDA_MATPC_EVEN_EVEN;

	inv_param.dagger		 = QUDA_DAG_NO;
//	inv_param.mass_normalization	 = QUDA_MASS_NORMALIZATION;
	inv_param.mass_normalization	 = QUDA_KAPPA_NORMALIZATION;
	inv_param.solver_normalization	 = QUDA_DEFAULT_NORMALIZATION;

//	inv_param.inv_type		 = QUDA_BICGSTAB_INVERTER;
	inv_param.inv_type		 = QUDA_CG_INVERTER;

	inv_param.gcrNkrylov		 = 30;
	inv_param.tol			 = precision;
	inv_param.maxiter		 = iteraHP;
	inv_param.reliable_delta	 = 1e-2; // ignored by multi-shift solver

#if __COMPUTE_CAPABILITY__ >= 200
	// require both L2 relative and heavy quark residual to determine convergence
//	inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL | QUDA_HEAVY_QUARK_RESIDUAL);
	inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
	inv_param.tol_hq = precision;	// specify a tolerance for the residual for heavy quark residual
#else
	// Pre Fermi architecture only supports L2 relative residual norm
	inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
#endif

	// domain decomposition preconditioner parameters

	inv_param.inv_type_precondition	 = QUDA_INVALID_INVERTER;
	inv_param.schwarz_type		 = QUDA_ADDITIVE_SCHWARZ;
	inv_param.precondition_cycle	 = 1;
	inv_param.tol_precondition	 = 1e-1;
	inv_param.maxiter_precondition	 = 10;
	inv_param.verbosity_precondition = QUDA_SILENT;
	inv_param.omega			 = 1.0;


	inv_param.cpu_prec		 = cpu_prec;
	inv_param.cuda_prec		 = cuda_prec;
	inv_param.cuda_prec_sloppy	 = cuda_prec_sloppy;
	inv_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
	inv_param.preserve_source	 = QUDA_PRESERVE_SOURCE_NO;
	inv_param.gamma_basis		 = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
//	inv_param.gamma_basis		 = QUDA_UKQCD_GAMMA_BASIS;
	inv_param.dirac_order		 = QUDA_DIRAC_ORDER;

	inv_param.tune			 = QUDA_TUNE_YES;
//	inv_param.tune			 = QUDA_TUNE_NO;
//	inv_param.preserve_dirac	 = QUDA_PRESERVE_DIRAC_NO;

	inv_param.input_location	 = QUDA_CPU_FIELD_LOCATION;
	inv_param.output_location	 = QUDA_CPU_FIELD_LOCATION;

	gauge_param.ga_pad		 = 0; // 24*24*24/2;
	inv_param.sp_pad		 = 0; // 24*24*24/2;
	inv_param.cl_pad		 = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
	int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
	int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
	int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
	int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
	int pad_size =MAX(x_face_size, y_face_size);
	pad_size = MAX(pad_size, z_face_size);
	pad_size = MAX(pad_size, t_face_size);
	gauge_param.ga_pad = pad_size; 
//  inv_param.cl_pad = pad_size; 
//  inv_param.sp_pad = pad_size; 
#endif

	if	(dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
	{
		inv_param.clover_cpu_prec		 = cpu_prec;
		inv_param.clover_cuda_prec		 = cuda_prec;
		inv_param.clover_cuda_prec_sloppy	 = cuda_prec_sloppy;
		inv_param.clover_cuda_prec_precondition	 = QUDA_HALF_PRECISION;
		inv_param.clover_order			 = QUDA_PACKED_CLOVER_ORDER;
	}

	inv_param.verbosity = QUDA_VERBOSE;

	//set the T dimension partitioning flag
	//commDimPartitionedSet(3);

	// *** Everything between here and the call to initQuda() is
	// *** application-specific.

	// set parameters for the reference Dslash, and prepare fields to be loaded
	setDims			(gauge_param.X);

	setSpinorSiteSize	(24);

	size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
	size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

	void *gauge[4];

	for	(int dir = 0; dir < 4; dir++)
		if	((gauge[dir]	 = malloc(V*gaugeSiteSize*gSize)) == NULL)
		{
			printf	("Fatal Error; Couldn't allocate memory in host for gauge fields. Asked for %ld bytes.", V*gaugeSiteSize*gSize);
			exit	(1);
		}

//	totalMem	+= ((double) (V*gaugeSiteSize*gSize*4))/(1024.*1024.*1024.);

	if	(strcmp(latfile,""))			// load in the command line supplied gauge field
	{
		if	(read_custom_binary_gauge_field((double**)gauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline))
		{
			printf	("Fatal Error; Couldn't read gauge conf %s\n", latfile);
			exit	(1);
		}
	}
	else
	{						// else generate a random SU(3) field
		construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
		inv_param.kappa	 = 0.12;
	}

	comm_barrier	();

	const int	Vol	 = xdim*ydim*zdim;

	// initialize the QUDA library
	initQuda(device);

	void	*cnRes_vv;
	void	*cnRes_gv;

	if	((cudaHostAlloc(&cnRes_vv, sizeof(double2)*16*tdim*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnRes_vv\n"), exit(1);
	if	((cudaHostAlloc(&cnRes_gv, sizeof(double2)*16*tdim*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnRes_gv\n"), exit(1);

	cudaMemset	(cnRes_vv, 0, sizeof(double2)*16*tdim*Vol);
	cudaMemset	(cnRes_gv, 0, sizeof(double2)*16*tdim*Vol);

	void	**cnD_vv;
	void	**cnD_gv;
	void	**cnC_vv;
	void	**cnC_gv;

	cnD_vv	 = (void**) malloc(sizeof(double2*)*4);
	cnD_gv	 = (void**) malloc(sizeof(double2*)*4);
	cnC_vv	 = (void**) malloc(sizeof(double2*)*4);
	cnC_gv	 = (void**) malloc(sizeof(double2*)*4);

	if	(cnD_gv == NULL) printf("Error allocating memory cnD_gv_HP\n"), exit(1);
	if	(cnD_vv == NULL) printf("Error allocating memory cnD_vv_HP\n"), exit(1);
	if	(cnC_gv == NULL) printf("Error allocating memory cnC_gv_HP\n"), exit(1);
	if	(cnC_vv == NULL) printf("Error allocating memory cnC_vv_HP\n"), exit(1);


	cudaDeviceSynchronize();

	for	(int mu = 0; mu < 4; mu++)
	{
		if	((cudaHostAlloc(&(cnD_vv[mu]), sizeof(double2)*tdim*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnD_vv)[%d]\n", mu), exit(1);
		cudaMemset	(cnD_vv[mu], 0, tdim*16*Vol*sizeof(double2));

		if	((cudaHostAlloc(&(cnD_gv[mu]), sizeof(double2)*tdim*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnD_gv)[%d]\n", mu), exit(1);
		cudaMemset	(cnD_gv[mu], 0, tdim*16*Vol*sizeof(double2));

		if	((cudaHostAlloc(&(cnC_vv[mu]), sizeof(double2)*tdim*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnC_vv)[%d]\n", mu), exit(1);
		cudaMemset	(cnC_vv[mu], 0, tdim*16*Vol*sizeof(double2));

		if	((cudaHostAlloc(&(cnC_gv[mu]), sizeof(double2)*tdim*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnC_gv)[%d]\n", mu), exit(1);
		cudaMemset	(cnC_gv[mu], 0, tdim*16*Vol*sizeof(double2));
	}

	cudaDeviceSynchronize();

	//	load the gauge field
	loadGaugeQuda	((void*)gauge, &gauge_param);

	inv_param.clover_coeff = 1.57551;
	inv_param.clover_coeff *= inv_param.kappa;

	if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(NULL, NULL, &inv_param);

	void	*spinorIn	 = malloc(V*spinorSiteSize*sSize);
	void	*spinorCheck	 = malloc(V*spinorSiteSize*sSize);

//	totalMem	+= ((double) (V*spinorSiteSize*sSize*3))/(1024.*1024.*1024.);

	void	*spinorOut	 = malloc(V*spinorSiteSize*sSize);

	int	**mom;

	if	((mom = (int **) malloc(sizeof(int*)*Vol)) == NULL)
	{
		printf	("Fatal Error: Couldn't allocate memory for momenta.");
		exit	(1);
	}

	for	(int ip=0; ip<Vol; ip++)
	{
		if	((mom[ip] = (int *) malloc(sizeof(int)*3)) == NULL)
		{
			printf	("Fatal Error: Couldn't allocate memory for momenta.");
			exit	(1);
		}
		else
		{
			mom[ip][0]	 = 0;
			mom[ip][1]	 = 0;
			mom[ip][2]	 = 0;
		}
	}

//	totalMem	+= ((double) (Vol*sizeof(int)*3))/(1024.*1024.*1024.);

	int momIdx	 = 0;
	int totMom	 = 0;

	for	(int pz = 0; pz < zdim; pz++)
		for	(int py = 0; py < ydim; py++)
			for	(int px = 0; px < xdim; px++)
			{
				if	(px < xdim/2)
					mom[momIdx][0]	 = px;
				else
					mom[momIdx][0]	 = px - xdim;

				if	(py < ydim/2)
					mom[momIdx][1]	 = py;
				else
					mom[momIdx][1]	 = py - ydim;

				if	(pz < zdim/2)
					mom[momIdx][2]	 = pz;
				else
					mom[momIdx][2]	 = pz - zdim;

				if	((mom[momIdx][0]*mom[momIdx][0]+mom[momIdx][1]*mom[momIdx][1]+mom[momIdx][2]*mom[momIdx][2])<=MaxP)
					totMom++;

				momIdx++;
			}

	printfQuda	("\nTotal momenta %d\n\n", totMom);

	gsl_rng	*rNum	 = gsl_rng_alloc(gsl_rng_ranlux);
	gsl_rng_set	(rNum, (int) clock());

	printfQuda	("Allocated memory for random number generator\n");


	// start the timer
	double time0 = -((double)clock());

	// perform the inversion

	printfQuda	("Starting inversions\n");

	inv_param.tol		= precision;
	inv_param.tol_hq	= precision;

	for	(i=0; i<numberHP; i++)
	{
		genRandomSource	(spinorIn, &inv_param, rNum);

		#ifdef	CROSSCHECK
			reOrder	((double*)spinorIn, (double*)spinorCheck, 24);

			FILE	*out;

			if	((out = fopen("oneEndTrick.In", "w+")) == NULL)
			{
				printf	("Error creating file.\n");
				return	-1;
			}

			for	(int j=0; j<V*12; j++)
				fprintf	(out, "%+1.1lf %+1.1lf\n", ((double*)spinorCheck)[2*j], ((double*)spinorCheck)[2*j+1]);

			fclose	(out);
		#endif

		inv_param.maxiter	= iteraHP;
		oneEndTrickCG	(spinorOut, spinorIn, &inv_param, cnRes_gv, cnRes_vv, cnD_gv, cnD_vv, cnC_gv, cnC_vv);

		#ifdef	CROSSCHECK
			reOrder	((double*)spinorOut, (double*)spinorCheck, 24);

			if	((out = fopen("oneEndTrick.Out", "w+")) == NULL)
			{
				printf	("Error creating file.\n");
				return	-1;
			}

			for	(int j=0; j<V*12; j++)
				fprintf	(out, "%+2.8le %+2.8le\n", ((double*)spinorCheck)[2*j], ((double*)spinorCheck)[2*j+1]);

			fclose	(out);
		#endif

		doCudaFFT	(1, cnRes_vv, cnRes_gv, cnD_vv, cnD_gv, cnC_vv, cnC_gv, cuda_prec);

		sprintf		(name, "H%03d.S%03d", numberHP, i);
		dumpData	(1, name, mom, cnRes_vv, cnRes_gv, cnD_vv, cnD_gv, cnC_vv, cnC_gv, 0, cuda_prec);

		if	(cuda_prec == QUDA_DOUBLE_PRECISION)
		{
			cudaMemset	(cnRes_vv, 0, tdim*16*Vol*sizeof(double2));
			cudaMemset	(cnRes_gv, 0, tdim*16*Vol*sizeof(double2));

			for	(int nu=0; nu<4; nu++)
			{
				cudaMemset	(cnD_vv[nu], 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnD_gv[nu], 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnC_vv[nu], 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnC_gv[nu], 0, tdim*16*Vol*sizeof(double2));
			}
		}
		else if	(cuda_prec == QUDA_SINGLE_PRECISION)
		{
			cudaMemset	(cnRes_vv, 0, tdim*16*Vol*sizeof(float2));
			cudaMemset	(cnRes_gv, 0, tdim*16*Vol*sizeof(float2));

			for	(int nu=0; nu<4; nu++)
			{
				cudaMemset	(cnD_vv[nu], 0, tdim*16*Vol*sizeof(float2));
				cudaMemset	(cnD_gv[nu], 0, tdim*16*Vol*sizeof(float2));
				cudaMemset	(cnC_vv[nu], 0, tdim*16*Vol*sizeof(float2));
				cudaMemset	(cnC_gv[nu], 0, tdim*16*Vol*sizeof(float2));
			}
		}

		if	(numberLP > 0)
		{
			inv_param.maxiter	= iteraLP;
			oneEndTrickCG	(spinorOut, spinorIn, &inv_param, cnRes_gv, cnRes_vv, cnD_gv, cnD_vv, cnC_gv, cnC_vv);

			doCudaFFT	(1, cnRes_vv, cnRes_gv, cnD_vv, cnD_gv, cnC_vv, cnC_gv, cuda_prec);

			sprintf		(name, "M%03d.S%03d", numberHP, i);
			dumpData	(1, name, mom, cnRes_vv, cnRes_gv, cnD_vv, cnD_gv, cnC_vv, cnC_gv, 0, cuda_prec);

			if	(cuda_prec == QUDA_DOUBLE_PRECISION)
			{
				cudaMemset	(cnRes_vv, 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnRes_gv, 0, tdim*16*Vol*sizeof(double2));

				for	(int mu=0; mu<4; mu++)
				{
					cudaMemset	(cnD_vv[mu], 0, tdim*16*Vol*sizeof(double2));
					cudaMemset	(cnD_gv[mu], 0, tdim*16*Vol*sizeof(double2));
					cudaMemset	(cnC_vv[mu], 0, tdim*16*Vol*sizeof(double2));
					cudaMemset	(cnC_gv[mu], 0, tdim*16*Vol*sizeof(double2));
				}
			}
			else if	(cuda_prec == QUDA_SINGLE_PRECISION)
			{
				cudaMemset	(cnRes_vv, 0, tdim*16*Vol*sizeof(float2));
				cudaMemset	(cnRes_gv, 0, tdim*16*Vol*sizeof(float2));

				for	(int mu=0; mu<4; mu++)
				{
					cudaMemset	(cnD_vv[mu], 0, tdim*16*Vol*sizeof(float2));
					cudaMemset	(cnD_gv[mu], 0, tdim*16*Vol*sizeof(float2));
					cudaMemset	(cnC_vv[mu], 0, tdim*16*Vol*sizeof(float2));
					cudaMemset	(cnC_gv[mu], 0, tdim*16*Vol*sizeof(float2));
				}
			}
		}
	}

	inv_param.maxiter	= iteraLP;

	for	(k=0; k<maxSources; k++)
	{
		if	((maxSources == 1) && (dataLP[k] == 0))
			continue;

		for	(i=0; i<dataLP[k]; i++)
		{
			printfQuda	("\nSource LP %04d\n", i);
			genRandomSource	(spinorIn, &inv_param, rNum);
			oneEndTrickCG	(spinorOut, spinorIn, &inv_param, cnRes_gv, cnRes_vv, cnD_gv, cnD_vv, cnC_gv, cnC_vv);
		}

		doCudaFFT	(1, cnRes_vv, cnRes_gv, cnD_vv, cnD_gv, cnC_vv, cnC_gv, cuda_prec);

		if	(flag && (k == (maxSources - 1)))
			sprintf	(name, "L9999");
		else
			sprintf	(name, "L%04d", dataLP[k]);

		dumpData	(dataLP[k], name, mom, cnRes_vv, cnRes_gv, cnD_vv, cnD_gv, cnC_vv, cnC_gv, 0, cuda_prec);	//dataLP[k] -> numberLP, el 1 se va

		if	(cuda_prec == QUDA_DOUBLE_PRECISION)
		{
			cudaMemset	(cnRes_vv, 0, tdim*16*Vol*sizeof(double2));
			cudaMemset	(cnRes_gv, 0, tdim*16*Vol*sizeof(double2));

			for	(int mu=0; mu<4; mu++)							//*Todo esto se va al carajo
			{
				cudaMemset	(cnD_vv[mu], 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnD_gv[mu], 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnC_vv[mu], 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnC_gv[mu], 0, tdim*16*Vol*sizeof(double2));
			}										//*---Hasta aquí---
		}
		else
		{
			cudaMemset	(cnRes_vv, 0, tdim*16*Vol*sizeof(float2));
			cudaMemset	(cnRes_gv, 0, tdim*16*Vol*sizeof(float2));

			for	(int mu=0; mu<4; mu++)							//*Todo esto se va al carajo
			{
				cudaMemset	(cnD_vv[mu], 0, tdim*16*Vol*sizeof(float2));
				cudaMemset	(cnD_gv[mu], 0, tdim*16*Vol*sizeof(float2));
				cudaMemset	(cnC_vv[mu], 0, tdim*16*Vol*sizeof(float2));
				cudaMemset	(cnC_gv[mu], 0, tdim*16*Vol*sizeof(float2));
			}										//*---Hasta aquí---
		}

	}

//	doCudaFFT	(0, NULL, NULL, NULL, NULL, NULL, NULL);

  // stop the timer
	double timeIO	 = -((double)clock());
	time0		+= clock();
	time0		/= CLOCKS_PER_SEC;
    
	printfQuda	("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", inv_param.spinorGiB, gauge_param.gaugeGiB);
	printfQuda	("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

	gsl_rng_free(rNum);

	for	(int ip=0; ip<Vol; ip++)
		free	(mom[ip]);

	free		(mom);
	cudaFreeHost	(cnRes_gv);
	cudaFreeHost	(cnRes_vv);

	for	(int mu=0; mu<4; mu++)
	{
		cudaFreeHost	(cnD_vv[mu]);
		cudaFreeHost	(cnD_gv[mu]);
		cudaFreeHost	(cnC_vv[mu]);
		cudaFreeHost	(cnC_gv[mu]);
	}

	free(cnD_vv);
	free(cnD_gv);
	free(cnC_vv);
	free(cnC_gv);

	if	(inv_param.solution_type == QUDA_MAT_SOLUTION)
	{
		if	(dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
			tm_mat	(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
		else
			wil_mat	(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);

		if	(inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
			ax	(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	}
	else if	(inv_param.solution_type == QUDA_MATPC_SOLUTION)
	{   
		if	(dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
			tm_matpc	(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
		else
			wil_matpc	(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);

		if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
			ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	}

	for	(int dir=0; dir<4; dir++)
		free	(gauge[dir]);

	mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);

	double nrm2	 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	double src2	 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);

	printfQuda	("Relative residual: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

	freeGaugeQuda	();

  // finalize the QUDA library
	endQuda		();

  // end if the communications layer
	MPI_Finalize	();


	free	(spinorIn);
	free	(spinorCheck);
	free	(spinorOut);

	timeIO		+= clock();
	timeIO		/= CLOCKS_PER_SEC;

	printf		("%g seconds spent on IO\n", timeIO);
	fflush		(stdout);

	return	0;
}


