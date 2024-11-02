// Giorgio Gamba 11/2/2024
// Basic CUDA vector sum code using Unified Memory Access

// Cuda libraries
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Standard libraries
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

__global__ void vectorSumUM(int* a, int* b, int* c, int length)
{
	const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < length)
		c[tid] = a[tid] + b[tid];
}

void initVector(int* v, int length)
{
	for (int i = 0; i < length; i++)
	{
		v[i] = rand() % 100;
	}
}

int main()
{
	int ID = cudaGetDevice(&ID);

	int length = 1 << 16;

	size_t bytes = length * sizeof(int);

	// Unified memory pointers
	int* a = nullptr;
	int* b = nullptr;
	int* c = nullptr;

	initVector(a, length);
	initVector(b, length);

	// Using this "managed" version permits to automatically move from CPU to GPU and viceversa when needed
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	const int BLOCK_SIZE = 256;								// Number of threads per block
	const int GRID_SIZE = (int)ceil(length / BLOCK_SIZE);	// Number of blocks per grid

	// Starts transferring information from CPU to GPU
	cudaMemPrefetchAsync(a, bytes, ID);
	cudaMemPrefetchAsync(b, bytes, ID);

	vectorSumUM<<<GRID_SIZE, BLOCK_SIZE >> > (a, b, c, length);

	// Waits for all the events of the GPU to be finished before continuing,
	// because CUDA operations are asynchronous. This is needed becase we use the Unified Memory Access.
	// In the other vector sum example, we didn't need it because the cudaMemCopy
	// was the synchronization barrier for all threads
	cudaDeviceSynchronize();

	// Starts collecting information from GPU
	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	return 0;
}