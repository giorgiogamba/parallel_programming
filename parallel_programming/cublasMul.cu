// Copyright Giorgio Gamba
// Cublas multiplication testing

#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// NOTE: Cublas is assuming that the memory is made of contiguous COLUMNS inside the memory, 
// on the opposite of a CPU layout where there are contiguous ROWS

// For the float multiplication check we use an epsilon to check if we are close enough
// we are not looking for the exact multiplication

int cublasMul()
{
	int n = 1 << 10;
	size_t matrixSize = n * n * sizeof(float);

	// Vectors creation and memory allocation
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;

	h_a = (float*)malloc(matrixSize);
	h_b = (float*)malloc(matrixSize);
	h_c = (float*)malloc(matrixSize);

	cudaMalloc(&d_a, matrixSize);
	cudaMalloc(&d_b, matrixSize);
	cudaMalloc(&d_c, matrixSize);

	// Random values generation using cuda random generation framework
	curandGenerator_t randomGen;
	curandCreateGenerator(&randomGen, CURAND_RNG_PSEUDO_DEFAULT);

	// Define the random generation using the system clock as seed
	curandSetPseudoRandomGeneratorSeed(randomGen, (unsigned long long)clock());

	// Defines the two input matrices in a random way using GPU randomization defining n*n elements
	// There's no need to make copies since we can create values and save them directly on GPU
	curandGenerateUniform(randomGen, d_a, n * n);
	curandGenerateUniform(randomGen, d_b, n * n);

	// Define multiplication operation
	cublasHandle_t multiplicationHandle;
	cublasCreate(&multiplicationHandle);

	// Parameters used for generalized multiplication
	// sgemm performs the following multiplication: c = (alpha * a) * b + (beta * c)
	// thus we cande fine alpha = 1 and beta + 0 to get a regular multiplication
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasSgemm(multiplicationHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

	cudaMemcpy(h_a, d_a, matrixSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b, d_b, matrixSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_c, d_c, matrixSize, cudaMemcpyDeviceToHost);

	// Solution verification can be done in the classic way, which is to assume that the matrix is composed by contiguous vectors,
	// but keeping in mind that they are placed by columns

	return 0;
}