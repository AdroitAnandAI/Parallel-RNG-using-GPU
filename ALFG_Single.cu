
#include <stdio.h>
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


#define nums2Generate 10000
#define SUBSEQUENCES  100

// LAG2 denotes p-q value
#define LAG1 127
#define LAG2 30
#define MODBIT 32
//rngType = 0 for ALFG, rngType = 1 for GFSR
#define RNGTYPE 0


// #define TILE_DIM 32
#define TILESIZE 32
#define BLOCKSIZE 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, unsigned long long int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%llu", &matrix[i*cols+j]);
		}
	}
}


// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, unsigned long long int *matrix, int rows, int cols, int skip) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			if ((i*cols+j) % (skip + 1) == 0) {
				fprintf(outputFilePtr, "%llu ", matrix[i*cols+j]);
				fprintf(outputFilePtr, "\n");
			}
		}		
	}
}


/**
 * Prints any 1D array in the form of a matrix 
 * */
void printMatrix(unsigned long long int *arr, int rows, int cols) {

	printf("\n%d %d\n", rows, cols);
	// outfile.open(filename);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			cout<<arr[i * cols + j]<<" ";
		}
		cout<<"\n";
	}
	// outfile.close();
}


// __device__ unsigned int inCount = 0, outCount = 0;


__global__ void monteCarloSimulation(unsigned long long int *rndNums, unsigned long long randLength, 
										unsigned long long *inCount, unsigned long long *outCount) {



	// x = (values[ptr])/(max)
	// y = (values[ptr+1])/(max)

	// z = math.sqrt(x*x+y*y)
	// if (z<1):
	// 	inval=inval+1
	// else:
	// 	outval=outval+1

	unsigned long long int modvalue = 1;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// __shared__ int inCount = 0, outCount = 0;

	// printf("gid = %d, p = %d, SUBSEQUENCES = %d\n", gid, p, SUBSEQUENCES);
	
	if (gid < randLength/2) {

		float x = (float)rndNums[2*gid]/(modvalue<<MODBIT);

		float y = (float)rndNums[2*gid+1]/(modvalue<<MODBIT);

		float z = sqrt(x*x+y*y);

		if (z < 1) {
			atomicAdd(inCount, 1);
		} else {
			atomicAdd(outCount, 1);
		}

		// printf("seeds = %ld", seeds[gid*subLength]);

		// for (int i = 0; i < p; i++) {
			
		// 	// printf("seed = %ld", seeds[gid][i]);
		// 	rndNums[gid * subLength + i] = seeds[gid * p + i];
		// }

		// for (int i = p; i < subLength; i++) {

		// 	if (RNGTYPE == 0)

		// 		rndNums[gid * subLength + i] = (rndNums[gid * subLength + i - p] +
		// 													rndNums[gid * subLength + i - p + q]) % (modvalue<<MODBIT);
		// 	else if (RNGTYPE == 1)
		// 		rndNums[gid * subLength + i] = (rndNums[gid * subLength + i - p] ^
		// 													rndNums[gid * subLength + i - p + q]) % (modvalue<<MODBIT);

		// }
	}
}




// computeLFG<<<(SUBSEQUENCES+512-1)/512, 512>>>(Mexps, rndSeq, subLength, p, q)
/**The variable p is the seed length, which is same as LAG1
 * and the variable q is the LAG2 **/
__global__ void computeLFG(unsigned long long int *rndNums, unsigned long long int *seeds, int subLength, int p, int q) {

	unsigned long long int modvalue = 1;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// printf("gid = %d, p = %d, SUBSEQUENCES = %d\n", gid, p, SUBSEQUENCES);
	
	if (gid < SUBSEQUENCES) {

		// printf("seeds = %ld", seeds[gid*subLength]);

		for (int i = 0; i < p; i++) {
			
			// printf("seed = %ld", seeds[gid][i]);
			rndNums[gid * subLength + i] = seeds[gid * p + i];
		}

		for (int i = p; i < subLength; i++) {

			if (RNGTYPE == 0)

				rndNums[gid * subLength + i] = (rndNums[gid * subLength + i - p] +
															rndNums[gid * subLength + i - p + q]) % (modvalue<<MODBIT);
			else if (RNGTYPE == 1)
				rndNums[gid * subLength + i] = (rndNums[gid * subLength + i - p] ^
															rndNums[gid * subLength + i - p + q]) % (modvalue<<MODBIT);

		}
	}
}


__global__ void computeMatMul(int *A, int *B, int *C, int q) {


	// strided access to be done only once to shared memory
	extern __shared__ int columnB[];

	/** These are the indices of matrix A
	 * p = blockDim.x; row = threadIdx.x 
	 * r = gridDim.x; column = blockIdx.x **/

	// int gidA = blockIdx.x * blockDim.x + threadIdx.x;
	// int gidB = threadIdx.x * gridDim.x + blockIdx.x;


	if (threadIdx.x == 0) {

		for (int i=0; i < q; i++)
			columnB[i] = B[gridDim.x*i + blockIdx.x];
	}

	__syncthreads();

	// if (threadIdx.x == 0) {

	// 	for (int i=0; i < q; i++)
	// 		printf("b = %d, v = %d \n", blockIdx.x, columnB[i]);
	// }
		

	int sum = 0;
	for (int i=0; i < q; i++) {

		sum += A[threadIdx.x * q + i] * columnB[i];
	}
	
	C[threadIdx.x * q + blockIdx.x] = sum;

}



__global__ void MatMulOpt(unsigned long long int* A, unsigned long long int* B, unsigned long long int* C, int p, int q, int r) {
    
    unsigned long long int subMatMult = 0;
    unsigned long long int modvalue = 1;
    // int q = p, r = p;

    // shared memory to store the tile
    __shared__ unsigned long long int subA[TILESIZE][TILESIZE];
    __shared__ unsigned long long int subB[TILESIZE][TILESIZE];

    // Row and Col represents the thread location in the matrix A and B
    int row = blockIdx.y*TILESIZE + threadIdx.y;
    int col = blockIdx.x*TILESIZE + threadIdx.x;


    for (int k = 0; k < (q + TILESIZE - 1)/TILESIZE; k++) {

         if (k*TILESIZE + threadIdx.x < q && row < p)
             subA[threadIdx.y][threadIdx.x] = A[row*q + k*TILESIZE + threadIdx.x];
         else
             subA[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILESIZE + threadIdx.y < q && col < r)
             subB[threadIdx.y][threadIdx.x] = B[(k*TILESIZE + threadIdx.y)*r + col];
         else
             subB[threadIdx.y][threadIdx.x] = 0.0;

         // to insure that every entry of the submatrices of A and B have been  
         // loaded into shared memory before any thread begins its computations
         __syncthreads();


         for (int n = 0; n < TILESIZE; ++n) {
         	 // if (subA[threadIdx.y][n] * subB[n][threadIdx.x] != 0)
         	 // 	printf("%d * %d \n", subA[threadIdx.y][n], subB[n][threadIdx.x]);
             subMatMult += subA[threadIdx.y][n] * subB[n][threadIdx.x];
         }

         // to ensure that every element of the submatrix of C has been processed 
         // before we begin loading the next submatrix of A or B
         __syncthreads();
    }

    if (row < p && col < r) {
    	// if matrix exponent then no need to take mod
    	if (p == r)
	        C[((blockIdx.y * blockDim.y + threadIdx.y)*r) +
	           (blockIdx.x * blockDim.x)+ threadIdx.x] = subMatMult;
	    else
	        C[((blockIdx.y * blockDim.y + threadIdx.y)*r) +
	           (blockIdx.x * blockDim.x)+ threadIdx.x] = subMatMult % (modvalue<<MODBIT);	    	
    }
}


/** Efficient Copy instead of expensive cudaDeviceToDevice copy **/
__global__ void copy_kernel(unsigned long long int *output, const unsigned long long int * input, int N)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < N) {

		output[gid] = input[gid];
	}

    // for (int i = blockIdx.x * blockDim.x + threadIdx.x;  i < N; i += blockDim.x * gridDim.x) 
}



// function to compute the matrix exponent M^iL
unsigned long long int *computeMatExp(unsigned long long int *MiL, unsigned long long int *M, int exp, int p, int count) {

	// printf("Inside compute");

	unsigned long long int *g_temp, *g_MiL;

	int times;

	if (count == 1)
		times = exp - 1;
	else
		times = exp;


	cudaMalloc((void**)&g_MiL, sizeof(unsigned long long int) * p * p);

	copy_kernel<<<ceil((p*p + 512-1)/512), 512>>>(g_MiL, MiL, p*p);

	// allocate memory...
	// cudaMalloc((void**)&g_MiL, sizeof(int) * p * p);
	cudaMalloc((void**)&g_temp, sizeof(unsigned long long int) * p * p);

	// copy the values...
	// cudaMemcpy(g_M, h_M, sizeof(int) * p * p, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)p / dimBlock.x), ceil((float)p / dimBlock.y));

	// MatMulOpt<<<dimGrid, dimBlock>>>(M, M, g_MiL, p);
	// printf(" times = %d\n ", times);

	// iterate only times-1 times as M is already there
	for (int e = 0; e < times; e++) {

		// if (count <= 10) 
			// printf("\nMult M = %d times", e+1);

		if (e % 2 == 0) {
			// printf("calling 1\n");
			MatMulOpt<<<dimGrid, dimBlock>>>(g_MiL, M, g_temp, p, p, p);
			// multiply_matrices<<<dim3((p+31)/32, (p+31)/32), dim3(32, 32)>>>(g_MiL, M, g_temp, p);
		}
		else {
			// printf("calling 2\n");
			MatMulOpt<<<dimGrid, dimBlock>>>(g_temp, M, g_MiL, p, p, p);
			// multiply_matrices<<<dim3((p+31)/32, (p+31)/32), dim3(32, 32)>>>(g_temp, M, g_MiL, p);
		}
		
	}
	
	if (times % 2 == 0)
		return g_MiL;
	else
		return g_temp;
}


// function to compute the output matrix
void compute(int p, int q, unsigned long long int *seed, unsigned long long int *rngOut) {

	struct timeval t1, t2;
	double seconds, microSeconds;

	// variable declarations...
	unsigned long long int *M_iL, *h_M, *g_M, *g_M_tmp, *test, *M_iL_tmp, *g_seed_prop, *h_seed_prop, *g_seed;
	unsigned long long int subLength = nums2Generate/ SUBSEQUENCES;

	h_M = (unsigned long long int*) malloc(p * p * sizeof(unsigned long long int));
	M_iL_tmp = (unsigned long long int*) malloc(p * p * sizeof(unsigned long long int)); //delete

	int columnCounter = 1;
	// compute M matrix in h_M
	for (int i = 0; i < p; i++) {
		if (i != p-1) {
			for (int j = 0; j < p; j++) {
				if (j == columnCounter) 
					h_M[i*p +j] = 1;
				else
					h_M[i*p +j] = 0;
			}
		} else {
			for (int j = 0; j < p; j++) {
				if (j == 0 || j == q) 
					h_M[i*p +j] = 1;
				else
					h_M[i*p +j] = 0;					
			}
		}
		columnCounter++;
	}
	// printf("Testing-1");
	printMatrix(h_M, p, p);

	// allocate memory...
	cudaMalloc((void**)&g_M, sizeof(unsigned long long int) * p * p);
	cudaMalloc((void**)&g_M_tmp, sizeof(unsigned long long int) * p * p);

	// cudaMalloc((void**)&test, sizeof(int) * p * p); // del
	// cudaMemset(test, 1, sizeof(int) * p * p);
	// cudaMemcpy(M_iL_tmp, test, sizeof(int) * p * p, cudaMemcpyDeviceToHost); // del

	// printMatrix(M_iL_tmp, p, p);

	// copy the values...
	cudaMemcpy(g_M, h_M, sizeof(unsigned long long int) * p * p, cudaMemcpyHostToDevice);
	cudaMemcpy(g_M_tmp, h_M, sizeof(unsigned long long int) * p * p, cudaMemcpyHostToDevice);

	// printf("\nsubLength = %d\n", subLength);



	int seqCounter = 1;
	unsigned long long int **Mexps = (unsigned long long int **)malloc(SUBSEQUENCES * sizeof(unsigned long long int*));

	for (int i = 0; i < SUBSEQUENCES; i++) {

		Mexps[i] = (unsigned long long int *)malloc(sizeof(unsigned long long int)* p * p);
	}

	Mexps[0] = g_M_tmp;

	/*****************************/
	// Mexps[1] = computeMatExp(Mexps[0], g_M, subLength, p);
	// //copy the result back...
	// cudaMemcpy(M_iL_tmp, Mexps[1], sizeof(int) * p * p, cudaMemcpyDeviceToHost); // del
	// printMatrix(M_iL_tmp, p, p);
	/*****************************/

	//compute M ^ iL matrix in GPU
	for (int i = 1; i < SUBSEQUENCES; i++) { // change to SUBSEQUENCES

		//stores the pointer in GPU DRAM
		Mexps[seqCounter] = computeMatExp(Mexps[seqCounter-1], g_M, subLength, p, i);
		seqCounter++; 

		cudaMemcpy(M_iL_tmp, Mexps[seqCounter-1], sizeof(unsigned long long int) * p * p, cudaMemcpyDeviceToHost); // del

		if ((i)*subLength <= 250) {
			// printf("Matrix for Subsequence = %d, M^{%d}", i, (i)*subLength);
		
			printMatrix(M_iL_tmp, p, p);
		}

	}
	// printf("First****");

	// //copy the result back...
	// cudaMemcpy(M_iL_tmp, Mexps[1], sizeof(unsigned long long int) * p * p, cudaMemcpyDeviceToHost); // del

	// printMatrix(M_iL_tmp, p, p);

	// printf("Second****");

	// //copy the result back...
	// cudaMemcpy(M_iL_tmp, Mexps[2], sizeof(unsigned long long int) * p * p, cudaMemcpyDeviceToHost); // del

	// printMatrix(M_iL_tmp, p, p);


	/****************Seed Generation for different SUBSEQUENCES***************/

	// Need to store p seeds for each subsequence
	unsigned long long int *rngSeeds = (unsigned long long int *)malloc(SUBSEQUENCES * sizeof(unsigned long long int) * p);

	for (int i = 0; i < p; i++) {

		// printf("seed %d = %ld\n", i, seed[i]);
		rngSeeds[i] = seed[i];
	}

	// unsigned long long int **seedGen;//, *seedSubSeq;
	// cudaMalloc(&seedGen, sizeof(unsigned long long int*) * SUBSEQUENCES);
	// // cudaMalloc(&seedSubSeq, sizeof(unsigned long long int) * p);

	// printf("Allocating Seeds1:\n");

	// unsigned long long int *seedSubSeq[SUBSEQUENCES];

	// for(int i = 0; i < SUBSEQUENCES; ++i)
 //       cudaMalloc(&seedSubSeq[i], sizeof(unsigned long long int) * p);

 //    cudaMemcpy(seedGen, seedSubSeq, sizeof(seedSubSeq), cudaMemcpyHostToDevice);

	// // cudaMalloc(&seedGen[0], sizeof(unsigned long long int) * p);
	// cudaMemcpy(&seedGen[0], seed, sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice); // copy seed

	printf("Allocating Seeds2:\n");
	cudaMalloc(&g_seed, sizeof(unsigned long long int) * p);
	cudaMemcpy(g_seed, seed, sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice);

	printf("Allocating Seeds3:\n");


	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)p / dimBlock.x), ceil((float)p / dimBlock.y));

	// h_seed_prop = (unsigned long long int*) malloc(p * sizeof(unsigned long long int));

	for (int i = 1; i < SUBSEQUENCES; i++) {
		
		cudaMalloc(&g_seed_prop, sizeof(unsigned long long int) * p);
		MatMulOpt<<<dimGrid, dimBlock>>>(Mexps[i], g_seed, g_seed_prop, p, p, 1);

		cudaMemcpy(&rngSeeds[i*p], g_seed_prop, sizeof(unsigned long long int) * p, cudaMemcpyDeviceToHost);

		// printf("%p-", g_seed_prop);
		//stores the seeds pointer in GPU DRAM. Reuse of memory
		// seedGen[i] = g_seed_prop;

		// copy_kernel<<<ceil((p + 512-1)/512), 512>>>(seedGen[i], g_seed_prop, p);

		// cudaMemcpy(&seedGen[i], g_seed_prop, sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice);

		// cudaMemcpy(&seedGen[i], g_seed_prop, sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice);

		// cudaMemcpy(h_seed_prop, g_seed_prop, sizeof(unsigned long long int) * p, cudaMemcpyDeviceToHost); 
		// cudaMemcpy(&seedGen[i], h_seed_prop, sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice);

		if (i == 1 || i == 2) {
			h_seed_prop = (unsigned long long int*) malloc(p * sizeof(unsigned long long int));
			//copy the result back...
			cudaMemcpy(h_seed_prop, g_seed_prop, sizeof(unsigned long long int) * p, cudaMemcpyDeviceToHost); // del
			printMatrix(h_seed_prop, p, 1);	
		}


	}


	for (int i = 0; i < 5*p; i++) {

		// printf("seed %d = %ld\n", i, rngSeeds[i]);
		// rngSeeds[i] = seed[i];
	}

	//Setting the seeds in GPU
	unsigned long long int *g_rngSeeds;// = (unsigned long long int *)malloc(SUBSEQUENCES * sizeof(unsigned long long int) * p);
	cudaMalloc(&g_rngSeeds, SUBSEQUENCES * sizeof(unsigned long long int) * p);
	cudaMemcpy(g_rngSeeds, rngSeeds, SUBSEQUENCES * sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice);

	/**************************************************************************/


	printf("Listing out Seeds:\n");

	// h_seed_prop = (unsigned long long int*) malloc(p * sizeof(unsigned long long int));
	// //copy the result back...
	// cudaMemcpy(h_seed_prop, &seedGen[2], sizeof(unsigned long long int) * p, cudaMemcpyDeviceToHost); // del
	// printMatrix(h_seed_prop, p, 1);	

	// cudaMemcpy(h_seed_prop, &seedGen[3], sizeof(unsigned long long int) * p, cudaMemcpyDeviceToHost); // del
	// printMatrix(h_seed_prop, p, 1);		


	/** Need to launch threads = SUBSEQUENCES to do parallel generation **/

	printf("subLength = %llu", subLength);

	// unsigned long long int **rndSeq = (unsigned long long int **)malloc(SUBSEQUENCES * sizeof(unsigned long long int*));

	// unsigned long long int *randomSubSeq;
	// for (int i = 0; i < SUBSEQUENCES; i++) {

	// 	cudaMalloc(&randomSubSeq, sizeof(unsigned long long int) * subLength);
	// 	rndSeq[i] = randomSubSeq;
	// 	// printf("addr = %p", randomSubSeq);
	// }

	unsigned long long int *randomNums;// = (unsigned long long int *)malloc(SUBSEQUENCES * sizeof(unsigned long long int) * p);
	cudaMalloc(&randomNums, SUBSEQUENCES * sizeof(unsigned long long int) * subLength);

	computeLFG<<<(ceil(SUBSEQUENCES+512-1)/512), 512>>>(randomNums, g_rngSeeds, subLength, p, q);




	unsigned long long *inCount, *outCount;
	cudaMalloc((void **)&inCount,sizeof(unsigned long long));
	cudaMalloc((void **)&outCount,sizeof(unsigned long long));

	cudaMemset(inCount, 0, sizeof(unsigned long long));
	cudaMemset(outCount, 0, sizeof(unsigned long long));

	dim3 dimBlockMC(512);
	dim3 dimGridMC(ceil((float)SUBSEQUENCES*subLength + 512 -1 / 512));

	monteCarloSimulation<<<dimGridMC, dimBlockMC>>>(randomNums, SUBSEQUENCES*subLength, inCount, outCount);

	unsigned long long int inCount_h, outCount_h;
	// inCount_h = (unsigned int*) malloc(sizeof(unsigned int));
	// outCount_h = (unsigned int*) malloc(sizeof(unsigned int));

	// inCount_h = (int) malloc(sizeof(int));

	cudaMemcpy(&inCount_h, inCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&outCount_h, outCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	printf("inCount_h = %llu, outCount_h = %llu, %f\n", inCount_h, outCount_h, 4.0*inCount_h/(inCount_h+outCount_h));

	cudaMemcpy(rngOut, randomNums, SUBSEQUENCES * sizeof(unsigned long long int) * subLength, cudaMemcpyDeviceToHost);




	// unsigned long long int *subSeqHost = (unsigned long long int*) malloc(subLength * sizeof(unsigned long long int));
	// for (int i = 0; i < SUBSEQUENCES; i++) {

	// 	cudaMemcpy(subSeqHost, rndSeq[i], sizeof(unsigned long long int) * subLength, cudaMemcpyDeviceToHost);
	// 	// cudaMalloc((void**)&randomSubSeq, sizeof(unsigned long long int) * subLength);

	// 	for (int j = 0; j < subLength; j++) { 

	// 		rngOut[i*subLength+j] = subSeqHost[j];
	// 	}
	// }


	// 	rndSeq[i] = randomSubSeq;
	// }



	// call the kernels for doing required computations..
	// computeMatSum<<<p, q>>>(g_matrixA, g_matrixB);

	// matSumCoalesced<<<gridSize, BLOCKSIZE>>>(g_matrixA, g_matrixB);


	// ***************************************************************//
	// gettimeofday(&t1, NULL);

	// computeMatMul<<<r, p, q * sizeof(int)>>>(g_matrixA, g_matrixC, g_temp, q);
	// cudaDeviceSynchronize();

	// gettimeofday(&t2, NULL);

	// // print the time taken by the compute function
	// seconds = t2.tv_sec - t1.tv_sec;
	// microSeconds = t2.tv_usec - t1.tv_usec;
	// printf("Time taken for Mat Mul 1: %.3f\n", 1000*seconds + microSeconds/1000);
	// ***************************************************************//


	
	// copy the result back...
	// cudaMemcpy(h_matrixX, g_matrixX, sizeof(int) * p * s, cudaMemcpyDeviceToHost); // change to p s


	// deallocate the memory...
	// gpuErrchk(cudaFree(g_M));

}



int main(int argc, char **argv) {

	// variable declarations
	int p = LAG1, q = LAG2; // lag values
	// int m = 1<<MODBIT; // mod value

	unsigned long long int *seed, *rngOut;

	unsigned long long int modvalue = 1;

	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *outputFileName = argv[1];

	FILE *outputFilePtr;

	// allocate memory and read input matrices
	seed = (unsigned long long int*) malloc(p * sizeof(unsigned long long int)); //is it p?

	// printf("***********************");
	// initialize p seeds
	for (int i=0; i < p; i++) {
		//random number between 1 and (1<<MODBIT)
		seed[i] = 1 + (rand() % (modvalue<<MODBIT)); 
		// printf("%llu, ", seed[i]);
	}
	
	// allocate memory for output matrix
	rngOut = (unsigned long long int*) malloc(nums2Generate * sizeof(unsigned long long int)); // change to p s


	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, seed, rngOut);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, rngOut, nums2Generate, 1, 0);

	// close files
    fclose(outputFilePtr);

	// deallocate memory
	free(seed);

	return 0;
}