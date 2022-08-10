
#include <stdio.h>
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


#define nums2Generate 500000000  		// total number of RNs to generate
#define oneIteration  10000000 	 	// number of RNs in each iteration
#define SUBSEQUENCES  5000 	 	// number of cores


/** Sample r q values
r 98 q 27
r 250 q 103
r 1279 q 216 418 **/

// LAG2 denotes p-q value
#define LAG1 250
#define LAG2 147
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




/**The variable p is the seed length, which is same as LAG1
 * and the variable q is the LAG2 **/
__global__ void computeGFSR(unsigned long long int *rndNums, unsigned long long int *seeds, int subLength, int p, int q) {

	unsigned long long int modvalue = 1;

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// printf("gid = %d, p = %d, SUBSEQUENCES = %d\n", gid, p, SUBSEQUENCES);
	
	if (gid < SUBSEQUENCES) {

		// copy the seeds as init
		for (int i = 0; i < p; i++) {
			
			// printf("seed = %ld", seeds[gid][i]);
			rndNums[gid * subLength + i] = seeds[gid * p + i];
		}

		for (int i = p; i < subLength; i++) {

				// This is the core GFSR formula
				rndNums[gid * subLength + i] = (rndNums[gid * subLength + i - p] ^
															rndNums[gid * subLength + i - p + q]) % (modvalue<<MODBIT);

		}
	}
}




/** To generate seeds for each processor, x_i, x_i+n, x_i+2*n etc.
 * This function rearranges n*p input seeds to be consumed by 'n' 
 * processors to generate RN in a leapfrog way **/

__global__ void genLeapFrogSeeds(unsigned long long int *seeds, unsigned long long int *prop_seeds, int p) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < SUBSEQUENCES) {

		for (int i = 0; i < p; i++) {

			// jump the 'subsequences' number of rngs to make it leap frog
			prop_seeds[gid * p + i] = seeds[gid + SUBSEQUENCES * i];
		}

	}

}


/** To rearrange the numbers from leapfrog sequence to normal sequence
 * Input: x_i, x_i+n, x_i+2*n etc. Output: x_i, x_i+1, x_i+2 etc.
 * This function rearranges all output RNs **/

__global__ void arrangeLeapFrog(unsigned long long int *input, unsigned long long int *output, 
														int p, unsigned long long int subLength) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < SUBSEQUENCES) {

		for (int i = 0; i < subLength; i++) {

			// arrange in 'subsequences' number to convert from LP to normal sequence
			output[gid + SUBSEQUENCES * i] = input[gid * subLength + i];
		}

	}

}

/*****************Seed Generation for all SUBSEQUENCES*******************************/
/*** The propagation of initial seed to n-1 subsequences using matrix mult for ALFG**/
/************************************************************************************/

unsigned long long int *generateSubSeqSeed(int p, int q, unsigned long long int *seed) {

	//To store the propagated seeds in GPU
	unsigned long long int *g_seed_prop, *g_seed;//, *g_rngSeeds;

	cudaMalloc(&g_seed_prop, sizeof(unsigned long long int) * SUBSEQUENCES * p);

	cudaMalloc(&g_seed, sizeof(unsigned long long int) * SUBSEQUENCES * p);
	cudaMemcpy(g_seed, seed, sizeof(unsigned long long int) * SUBSEQUENCES * p, cudaMemcpyHostToDevice);

	// To generate seeds for each processor, x_i, x_i+n, x_i+2*n etc
	genLeapFrogSeeds<<<ceil((float)(SUBSEQUENCES + 512 -1) / 512), 512>>>(g_seed, g_seed_prop, p);
	cudaDeviceSynchronize();

	return g_seed_prop;

}

/************************************************************************************/
/******************** The function to compute the output RNGS************************/
/************************************************************************************/

void compute(int p, int q, unsigned long long int *seed, unsigned long long int *rngOut, FILE *outputFilePtr) {

	// variable declarations...
	unsigned long long int subLength;

	struct timeval t1, t2, t3;
	double seconds, microSeconds, seconds_gen, microSeconds_gen;

	// Irrespective of # of RNs to generate we keep 
	// the subsequence length the same
	subLength = oneIteration/ SUBSEQUENCES;


	unsigned long long int *randomNums, *randomNums_LP;// = (unsigned long long int *)malloc(SUBSEQUENCES * sizeof(unsigned long long int) * p);
	cudaMalloc(&randomNums, SUBSEQUENCES * sizeof(unsigned long long int) * subLength);
	cudaMalloc(&randomNums_LP, SUBSEQUENCES * sizeof(unsigned long long int) * subLength);

	unsigned long long int *rndNum_LP_temp = (unsigned long long int *)malloc(SUBSEQUENCES * sizeof(unsigned long long int) * subLength);


	int iteration = ceil((float)nums2Generate / oneIteration);

	unsigned long long int lastIterCount = nums2Generate % oneIteration;
	
	int lastIter = 0;

	for (int i = 0; i < iteration; i++) {

		// After 1st iteration, copy last 'p' values of generated random numbers as seed.
		if (i > 0) 
			for (int j = 0; j < SUBSEQUENCES*p; j++) {

				seed[j] = rngOut[SUBSEQUENCES * subLength - SUBSEQUENCES*p + j];
			}

		// Generate seeds of all subsequences for next iteration - FOR GFSR RNG 
		unsigned long long int *g_rngSeeds = generateSubSeqSeed(p, q, seed);

		gettimeofday(&t1, NULL);
		computeGFSR<<<(ceil(SUBSEQUENCES+512-1)/512), 512>>>(randomNums_LP, g_rngSeeds, subLength, p, q);

		// To generate seeds for each processor, x_i, x_i+n, x_i+2*n etc
		arrangeLeapFrog<<<ceil((float)(SUBSEQUENCES + 512 -1) / 512), 512>>>(randomNums_LP, randomNums, p, subLength);

		cudaDeviceSynchronize();
		gettimeofday(&t2, NULL);

		cudaMemcpy(rngOut, randomNums, SUBSEQUENCES * sizeof(unsigned long long int) * subLength, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		gettimeofday(&t3, NULL);

		// compute the Gen. Speed (not including Transfer time)
		seconds_gen = t2.tv_sec - t1.tv_sec;
		microSeconds_gen = t2.tv_usec - t1.tv_usec;
		float genTime = 1000*seconds_gen + microSeconds_gen/1000;

		// compute the Gen. + Transfer Speed (including Transfer time)
		seconds = t2.tv_sec - t1.tv_sec + t3.tv_sec - t2.tv_sec;
		microSeconds = t2.tv_usec - t1.tv_usec + t3.tv_usec - t2.tv_usec;
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

	// Allocate memory and read input matrices. 
	// Seeds required for LeapFrog = Number of procesors * p
	seed = (unsigned long long int*) malloc(SUBSEQUENCES * p * sizeof(unsigned long long int));

	// printf("***********************");
	// initialize p seeds
	for (int i=0; i < SUBSEQUENCES*p; i++) {

		//random number between 1 and (1<<MODBIT)
		seed[i] = 1 + (rand() % (modvalue<<MODBIT)); 
		// printf("%llu, ", seed[i]);
	}
	
	// allocate memory for output matrix
	rngOut = (unsigned long long int*) malloc(nums2Generate * sizeof(unsigned long long int));


	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, seed, rngOut, outputFilePtr);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);


	// close files
    fclose(outputFilePtr);

	// deallocate memory
	free(seed);

	return 0;
}