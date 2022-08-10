
#include <stdio.h>
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


#define nums2Generate 500000000  // total number of RNs to generate
#define oneIteration  10000000   // number of RNs in each iteration
#define SUBSEQUENCES  5000 	  // number of cores

/** Sample p q values
j	7	5	24	65	128	6	31	97	353	168	334	273	418
k	10	17	55	71	159	31	63	127	521	521	607	607	1279
**/

// Note that LAG2 denotes p-q value
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
void writeMatrix(FILE *outputFilePtr, unsigned long long int *matrix, int rows, int cols, 
									int skip, int isLastIter, unsigned long long num2Print) {

	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			if ((i*cols+j) % (skip + 1) == 0) {

				if ((isLastIter == 0) || (isLastIter == 1 && i*cols+j < num2Print)) {
					fprintf(outputFilePtr, "%llu ", matrix[i*cols+j]);
					fprintf(outputFilePtr, "\n");
				}
			}
		}		
	}
}


/**
 * Prints any 1D array in the form of a matrix 
 * */
void printMatrix(unsigned long long int *arr, int rows, int cols) {

	// printf("\n%d %d\n", rows, cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			cout<<arr[i * cols + j]<<" ";
		}
		cout<<"\n";
	}

}



__global__ void monteCarloSimulation(unsigned long long int *rndNums, unsigned long long randLength, 
										unsigned long long *inCount, unsigned long long *outCount) {


	unsigned long long int modvalue = 1;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// Take pairs of numbers. Hence only half length
	if (gid < randLength/2) {

		// Convert numbers between 0 and 1
		float x = (float)rndNums[2*gid]/(modvalue<<MODBIT);
		// Convert numbers between 0 and 1
		float y = (float)rndNums[2*gid+1]/(modvalue<<MODBIT);

		// Compute the distance
		float z = sqrt(x*x+y*y);

		// Increment inCount if within circle 
		// otherwise increment outCounter
		if (z < 1) {

			// need to add 2 as we do once for 2 numbers
			atomicAdd(inCount, 2);

		} else {
			// need to add 2 as we do once for 2 numbers
			atomicAdd(outCount, 2);
		}

	}
}




// computeLFG<<<(SUBSEQUENCES+512-1)/512, 512>>>(Mexps, rndSeq, subLength, p, q)
/**The variable p is the seed length, which is same as LAG1
 * and the variable q is the LAG2 **/
__global__ void computeLFG(unsigned long long int *rndNums, unsigned long long int *seeds, int subLength, int p, int q) {

	unsigned long long int modvalue = 1;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Need only threads = 'subsequences' as we can generate those many only in parallel.
	if (gid < SUBSEQUENCES) {

		for (int i = 0; i < p; i++) {
			
			// copy the seed from the 'p' indexed location for each subsequence
			rndNums[gid * subLength + i] = seeds[gid * p + i];
		}

		for (int i = p; i < subLength; i++) {

				// This is the formula for ALFG. MODBIT can set the max value
				rndNums[gid * subLength + i] = (rndNums[gid * subLength + i - p] +
															rndNums[gid * subLength + i - p + q]) % (modvalue<<MODBIT);

		}
	}
}

/** Computation: A * B (Matrix Multiplication - Non Optimized)
 * We can load each column of the matrix B into shared memory first.
 * This can be done by just one thread (idx = 0), and then reuse the 
 * column values in shared memory to do repeated access to compute 
 * matrix multiplication value for each cell, in output matrix C. 
 * Note that the kernel has thread divergence, as only one thread 
 * would run when BLOCKSIZE - 1 threads would be masked out. 
 * 
 * This is the non-optimized version of matrix multiplication.
 * As matrix operation is called many times, an optimized version
 * is implemented. Keeping it for fallback. **/

__global__ void computeMatMul(int *A, int *B, int *C, int q) {


	// strided access to be done only once to shared memory
	extern __shared__ int columnB[];

	/** These are the indices of matrix A
	 * p = blockDim.x; row = threadIdx.x 
	 * r = gridDim.x; column = blockIdx.x **/

	if (threadIdx.x == 0) {

		for (int i=0; i < q; i++)
			columnB[i] = B[gridDim.x*i + blockIdx.x];
	}

	__syncthreads();

	int sum = 0;
	for (int i=0; i < q; i++) {

		sum += A[threadIdx.x * q + i] * columnB[i];
	}
	
	C[threadIdx.x * q + blockIdx.x] = sum;

}


/** Computation: A * B (Matrix Multiplication - Optimized)
 * 
 * We can load a sub-matrix of matrix A and B into shared memory and then multiply
 * and add the sub-matrices to find out the submatrix of matrix, C. As the multiplication
 * happens in the shared memory tile, the repeated access would be much faster (L1 cache) 
 * 
 * The idea is from the below source (Page 18 of the PDF below)
 * http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
 * 
 * The iteration to get the submatrix is done till the edge of the matrix and padding is done 
 * for the last tile (when matrix size is not a multiple of tile size). Note that, the tile size 
 * is maintained at multiples of 32 so as to minimize the bank conflicts.*/

__global__ void MatMulOpt(unsigned long long int* A, unsigned long long int* B, unsigned long long int* C, int p, int q, int r) {
    
    unsigned long long int subMatMult = 0;
    unsigned long long int modvalue = 1;

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

		//simple parallel copy from input to output
		output[gid] = input[gid];
	}

}


/************The function to compute the matrix exponent M^iL***************/
/**********Ping pong computation in GPU memory is done to compute***********/
/***************************************************************************/
 
unsigned long long int *computeMatExp(unsigned long long int *MiL, unsigned long long int *M, int exp, int p, int count) {


	unsigned long long int *g_temp, *g_MiL;

	int times;

	if (count == 1)
		times = exp - 1;
	else
		times = exp;


	cudaMalloc((void**)&g_MiL, sizeof(unsigned long long int) * p * p);

	copy_kernel<<<ceil((p*p + 512-1)/512), 512>>>(g_MiL, MiL, p*p);

	// allocate memory...
	cudaMalloc((void**)&g_temp, sizeof(unsigned long long int) * p * p);

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)p / dimBlock.x), ceil((float)p / dimBlock.y));


	// iterate only times-1 times as M is already there
	for (int e = 0; e < times; e++) {


		if (e % 2 == 0) {

			// Storing the value in g_temp in even iteration
			MatMulOpt<<<dimGrid, dimBlock>>>(g_MiL, M, g_temp, p, p, p);
		}
		else {

			// Storing the value in g_MiL in odd iteration
			MatMulOpt<<<dimGrid, dimBlock>>>(g_temp, M, g_MiL, p, p, p);
		}
		
	}
	
	// To check where the final output is!
	if (times % 2 == 0)
		return g_MiL;
	else
		return g_temp;
}


/****************Seed Generation for all SUBSEQUENCES***********************/
/** The propagation of initial seed to n-1 subsequences using matrix mult **/
/***************************************************************************/
unsigned long long int *generateSubSeqSeed(int p, int q, unsigned long long int *seed, unsigned long long int **Mexps) {

	// variable declarations...
	unsigned long long int *g_seed_prop, *g_seed;

	// Need to store p seeds for each subsequence
	unsigned long long int *rngSeeds = (unsigned long long int *)malloc(SUBSEQUENCES * sizeof(unsigned long long int) * p);

	for (int i = 0; i < p; i++) {

		rngSeeds[i] = seed[i];
	}

	cudaMalloc(&g_seed, sizeof(unsigned long long int) * p);
	cudaMemcpy(g_seed, seed, sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice);


	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)p / dimBlock.x), ceil((float)p / dimBlock.y));


	for (int i = 1; i < SUBSEQUENCES; i++) {
		
		cudaMalloc(&g_seed_prop, sizeof(unsigned long long int) * p);

		MatMulOpt<<<dimGrid, dimBlock>>>(Mexps[i], g_seed, g_seed_prop, p, p, 1);

		cudaMemcpy(&rngSeeds[i*p], g_seed_prop, sizeof(unsigned long long int) * p, cudaMemcpyDeviceToHost);
	}


	//Setting the seeds in GPU
	unsigned long long int *g_rngSeeds;

	cudaMalloc(&g_rngSeeds, SUBSEQUENCES * sizeof(unsigned long long int) * p);
	cudaMemcpy(g_rngSeeds, rngSeeds, SUBSEQUENCES * sizeof(unsigned long long int) * p, cudaMemcpyHostToDevice);

	return g_rngSeeds;

}


// function to compute the output matrix
void compute(int p, int q, unsigned long long int *seed, unsigned long long int *rngOut, FILE *outputFilePtr) {

	struct timeval t1, t2, t3, t4;
	double seconds, microSeconds, seconds_gen, microSeconds_gen;

	// variable declarations...
	unsigned long long int *h_M, *g_M, *g_M_tmp; //, *test;//, *M_iL_tmp;//, *g_seed_prop, *h_seed_prop, *g_seed;
	unsigned long long int subLength;


	// Irrespective of # of RNs to generate we keep 
	// the subsequence length the same
	subLength = oneIteration/ SUBSEQUENCES;


	h_M = (unsigned long long int*) malloc(p * p * sizeof(unsigned long long int));
	// M_iL_tmp = (unsigned long long int*) malloc(p * p * sizeof(unsigned long long int)); //delete


	printf ("\nAbout to compute matrix, M.\n");
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

	printf ("Compute matrix, M done!\n\n");

	// allocate memory...
	cudaMalloc((void**)&g_M, sizeof(unsigned long long int) * p * p);
	cudaMalloc((void**)&g_M_tmp, sizeof(unsigned long long int) * p * p);

	// copy the values...
	cudaMemcpy(g_M, h_M, sizeof(unsigned long long int) * p * p, cudaMemcpyHostToDevice);
	cudaMemcpy(g_M_tmp, h_M, sizeof(unsigned long long int) * p * p, cudaMemcpyHostToDevice);


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

	printf ("Doing the pre-processing step: Generate M ^ iL matrix in GPU. Note that it is a one time processing step. Once computed, it can be kept in GPU DRAM or disk.................\n\n");
	
	//compute M ^ iL matrix in GPU
	for (int i = 1; i < SUBSEQUENCES; i++) { // change to SUBSEQUENCES

		//stores the pointer in GPU DRAM
		Mexps[seqCounter] = computeMatExp(Mexps[seqCounter-1], g_M, subLength, p, i);
		seqCounter++; 

		// cudaMemcpy(M_iL_tmp, Mexps[seqCounter-1], sizeof(unsigned long long int) * p * p, cudaMemcpyDeviceToHost); // del

	}

	printf ("Pre-processing step: M ^ iL matrix generation done.\n\n");


	unsigned long long int inCount_h, outCount_h;
	unsigned long long int *randomNums;// = (unsigned long long int *)malloc(SUBSEQUENCES * sizeof(unsigned long long int) * p);
	cudaMalloc(&randomNums, SUBSEQUENCES * sizeof(unsigned long long int) * subLength);


	unsigned long long *inCount, *outCount;
	cudaMalloc((void **)&inCount,sizeof(unsigned long long));
	cudaMalloc((void **)&outCount,sizeof(unsigned long long));


	dim3 dimBlockMC(512);
	dim3 dimGridMC(ceil((float)SUBSEQUENCES*subLength + 512 -1 / 512));


	int iteration = ceil((float)nums2Generate / oneIteration);

	unsigned long long int lastIterCount = nums2Generate % oneIteration;
	
	int lastIter = 0;

	printf ("Generating random numbers iteratively using CST:\n\n");

	for (int i = 0; i < iteration; i++) {

		// After 1st iteration, copy last 'p' values of generated random numbers as seed.
		if (i > 0) 

			for (int j = 0; j < p; j++) {

				seed[j] = rngOut[SUBSEQUENCES * subLength - p + j];
			}


		// Generate seeds for next iteration
		unsigned long long int *g_rngSeeds = generateSubSeqSeed(p, q, seed, Mexps);

		// Main computation function: RNG
		gettimeofday(&t1, NULL);
		computeLFG<<<(ceil(SUBSEQUENCES+512-1)/512), 512>>>(randomNums, g_rngSeeds, subLength, p, q);
		cudaDeviceSynchronize();
		gettimeofday(&t2, NULL);

		// Initialize the counters to 0 before doing Monte Carlo Simulation		
		cudaMemset(inCount, 0, sizeof(unsigned long long));
		cudaMemset(outCount, 0, sizeof(unsigned long long));

		// Launch Monte Carlo Simulation with the set of generated random numbers in this iteration
		monteCarloSimulation<<<dimGridMC, dimBlockMC>>>(randomNums, SUBSEQUENCES*subLength, inCount, outCount);

		cudaMemcpy(&inCount_h, inCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		cudaMemcpy(&outCount_h, outCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

		printf("\nIteration %d:\nRNs inside Unit Circle = %llu, Outside = %llu, %f\n", 
													i, inCount_h, outCount_h, 4.0*inCount_h/(inCount_h+outCount_h));

		gettimeofday(&t3, NULL);
		cudaMemcpy(rngOut, randomNums, SUBSEQUENCES * sizeof(unsigned long long int) * subLength, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		gettimeofday(&t4, NULL);

		// print the time taken by the RNG
		
		// compute the Gen. Speed (not including Transfer time)
		seconds_gen = t2.tv_sec - t1.tv_sec;
		microSeconds_gen = t2.tv_usec - t1.tv_usec;
		float genTime = 1000*seconds_gen + microSeconds_gen/1000;

		// compute the Gen. + Transfer Speed (including Transfer time)
		seconds = t2.tv_sec - t1.tv_sec + t4.tv_sec - t3.tv_sec;
		microSeconds = t2.tv_usec - t1.tv_usec + t4.tv_usec - t3.tv_usec;
		float totalTime = 1000*seconds + microSeconds/1000;

		printf("Time taken to generate %llu RNs: %.3f (ms). \nGen + Transfer. Speed = %.2f MRS. Gen. Speed = %.2f MRS\n\n", 
				SUBSEQUENCES*subLength, totalTime, SUBSEQUENCES*subLength/(totalTime*1000), SUBSEQUENCES*subLength/(genTime*1000));

		// Inform the file writer if less values are to be printed in last iteration
		// This happens when the total RNs to generate is not divisible by RNs that can be generated in 1 iteration.
		if (i == iteration - 1 && lastIterCount != 0)
			lastIter = 1;

		writeMatrix(outputFilePtr, rngOut, oneIteration, 1, 0, lastIter, lastIterCount);
	
	}

}



int main(int argc, char **argv) {

	// variable declarations
	int p = LAG1, q = LAG2; // lag values

	unsigned long long int *seed, *rngOut;

	unsigned long long int modvalue = 1;

	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *outputFileName = argv[1];

	FILE *outputFilePtr;

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w+");

	// allocate memory and read input matrices
	seed = (unsigned long long int*) malloc(p * sizeof(unsigned long long int)); //is it p?

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
	compute(p, q, seed, rngOut, outputFilePtr);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Total Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);


	// close files
    fclose(outputFilePtr);

	// deallocate memory
	free(seed);

	return 0;
}