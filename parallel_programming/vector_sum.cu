// Giorgio Gamba 11/2/2024
// Basic CUDA vector sum code

// Cuda libraries
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// Standard libraries
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Adds vectors a and b in c using GPU
__global__ void vectorAdd(int* a, int* b, int* c, int length)
{
	// Each thread computes a specific vector member addition
	// Threads are are organized in block, thus in every block there are blockIdx.x * blockDim.x threads
	const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < length)
		c[tid] = a[tid] + b[tid];
}

// Initializes the passed vector with a random number in [0, 99]
void initVector(int* m, const int size)
{
	for (int i = 0; i < size; i++)
	{
		m[i] = rand() % 100;
	}
}

int main()
{
	int length = 1 << 16; // number composed by 16 1s (65536 elements)
	int* h_a;
	int* h_b;
	int* h_c;

	int* d_a;
	int* d_b;
	int* d_c;

	// Total number of bytes needed to store a vector (int is 4 bytes)
	size_t bytesNum = sizeof(int) * length;

	// Allocates vectors space
	h_a = (int*)malloc(bytesNum);
	h_b = (int*)malloc(bytesNum);
	h_c = (int*)malloc(bytesNum);

	// Allocates memory on GPU
	cudaMalloc(&d_a, bytesNum);
	cudaMalloc(&d_b, bytesNum);
	cudaMalloc(&d_c, bytesNum);

	initVector(h_a, length);
	initVector(h_b, length);

	// Copies data from CPU to GPU
	cudaMemcpy(d_a, h_a, bytesNum, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, h_a, bytesNum, cudaMemcpyHostToDevice);

	// 1D space
	const int NUM_THREADS = 256;
	const int NUM_BLOCKS = (int)ceil(length / NUM_THREADS);

	// Computes vector addition
	vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, length);

	// Copies back result from GPU to CPU
	cudaMemcpy(h_c, d_c, bytesNum, cudaMemcpyDeviceToHost);

	return 0;
}