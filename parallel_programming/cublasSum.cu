// Giorgio Gamba
// Simple script for Cublas testing

#include <cublas_v2.h>

#include <cuda_runtime.h>
#include <stdlib.h>

// This method explores the cublas sum
int cublasSum()
{
	const int size = 1 << 2;
	const int bytes = size * sizeof(float);
	

	float* h_a = (float*)malloc(bytes);
	float* h_b = (float*)malloc(bytes);
	float* h_c = (float*)malloc(bytes);

	float* d_a;
	float* d_b;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	// NOTE as we are using cublas we don't need d_c

	// Vector initialization
	// Usual rand code

	// Create Cublas objects
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	// Copy vectors over the device
	// 1s are stepsizes
	cublasSetVector(size, sizeof(float), h_a, 1, d_a, 1);
	cublasSetVector(size, sizeof(float), h_b, 1, d_b, 1);

	// Execute vector addition using saxpy
	// (single precision a * x + y)
	const float scale = 1.f;
	cublasSaxpy(handle, size, &scale, d_a, 1, d_b, 1);

	// Copies result vector
	// NOTE: The result is saved inside the second vector
	cublasGetVector(size, sizeof(float), d_b, 1, h_c, 1);

	cublasDestroy(handle);

	cudaFree(d_a);
	cudaFree(d_b);
	free(h_a);
	free(h_b);

	return 0;
}