// Copyright Giorgio Gamba
// This algorithm is an example of a sum reduction algorithm with warp divergence

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <device_launch_parameters.h>

#define TBSIZE 256				// size of a single thread block
#define SHMEMSIZE TBSIZE * 4	// number of threads timers the integer size

// Basically we instantiate the shared memory to contains thread for each chunk of vector,
// but in the end we use only half of them at each iteration (respect to the previous one)

// @param v input
// @param r output
__global__ void sum_reduction(int* v, int* r)
{
	// Allocate shared memory
	__shared__ int partialSum[SHMEMSIZE];

	// Compute thread ID
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements inside shared memory
	partialSum[threadIdx.x] = v[tid];
	__syncthreads();

	// Iteration based on a log base 2 block dimension
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		// Given the pattern of the algorithm, we reduce the number of working threads at each iteration by 2

		// Each thread adds adjecents blocks together (e.g.: thread 0 adds block 0 and block 1)

		if (threadIdx.x % (2 * i) == 0)
			partialSum[threadIdx.x] += partialSum[threadIdx.x + i];

		__syncthreads();
	}

	// At the end on calculation, only the thread at position 0 will be active
	if (threadIdx.x == 0)
		r[blockIdx.x] = partialSum[0];
}

int sum_reduction()
{
	// Initialize vector of 2^16 elements
	const int length = 1 << 16; // 2^16
	const size_t bytes = length * sizeof(int);

	// Create pointers for original vector and resulting vector, both on CPU and GPU
	int *hv, *hr;
	int* dv, * dr;

	// CPU Memory allocation
	hv = (int*)malloc(bytes);
	hr = (int*)malloc(bytes);

	// GPU Memory allocation
	cudaMalloc(&dv, bytes);
	cudaMalloc(&dr, bytes);

	// Random vector initialization
	for (size_t i = 0; i < length; ++i)
		hv[i] = rand() % 99;

	cudaMemcpy(dv, hv, bytes, cudaMemcpyHostToDevice);

	// Create 256 thread blocks of size 256, so that we can perform partial sum on each one of them
	// and then sum them together

	const int gridSize = (int)ceil(length / TBSIZE);

	// First, we computer inside 256 block 256 partial sums
	sum_reduction <<<gridSize, TBSIZE >>> (dv, dr);

	// Then, we will have a vector of 256 elements, each one representing a partial sum, and we reduce it
	// (note that we use the resulting vector as input also
	sum_reduction<<<1, TBSIZE>>>(dr, dr);

	cudaMemcpy(hr, dr, bytes, cudaMemcpyDeviceToHost);

	// Print the result
	printf("Result %d", hr[0]);

	return 0;
}