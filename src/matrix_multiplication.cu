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

// MAtrix multiplication strategy
// 1. Assign a thread for each element ofC
// 2. Since every element of C is a multiplication sum of A rows and B cols,
// each thread traverses A row and B col
// 3. Each thread writes its result

// Indexing
// 1. BlockIdx: number of the block we are in
// 2. ThreadIdx: which thread are we in in a thread block in a X and Y dimension
// 3. BlockDim constant: number of thread * thread block Idx

// Supponendo di avere una griglia di blocchi 2x2, ogni blocco contenente 2x2 thread,
// vogliamo capire a che indirizzo 'e il thread in alto a sx del blocco in basso a dx
// row = blockIdx.Y * blockDim.y + threadIdx.y
// col = blockIdx.x * blockDim.x + threadIdx.x
// sapendo che blockIdx(1, 1) e threadIdx(0, 0)

// Coalescing writes (scritture coalizzate)
// Basically there's the problem that, depending on which data we need,
// we would need to access elements on a matrix row or column. About the row, 
// we see that elements are consecutively stored and thus we can use LOCALITY
// to access it and avoid wasted of memory and time (coleasced access). But, about column,
// the problem is that elements are placed in separate rows, thus their are far
// and thus the memory access is way more expensive
// We can solve it by doing to preprocessing

// Initializes the passed matrix with random values in [0, 99]
void initMatrix(int* m, const int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			m[i][j] = rand() % 100;
		}
	}
}

__global__ void matrixMultiplication(int* a, int* b, int* c, int size)
{
	// Compute thread location inside the grid
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	int tmp = 0;
	if (row < size && col < size)
	{
		for (int k = 0; k < n; k++)
		{
			// In this computation we need to think about a single thread because they have unique row and col
			// Thus wedon't need 3 for loops but just 1
			tmp += a[row * n + k] * b[k * n + col];
		}

		c[row * n + col] = tmp;
	}
}

int main()
{
	// Instantiate a 1024x1024 matrix
	const int size = 1 << 10;
	const int bytes = size * size * sizeof(int);

	// CPU pointers
	// Factors matrices
	int* h_a = (int*)malloc(bytes);
	int* h_b = (int*)malloc(bytes);

	// Resulting matrix
	int* h_c = (int*)malloc(bytes);

	// GPU pointers
	int* d_a;
	int* d_b;
	int* d_c;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	initMatrix(h_a);
	initMatrix(h_b);
	initMatrix(h_c);

	// Copy elements to the GPU
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

	const int BLOCK_SIZE = 256;							// Number of threads for each block
	const int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);	// Number of blocks in the grid

	const dim3 grid(GRID_SIZE, GRID_SIZE); // Defines a 3D CUDA Matrix of size GRID_SIZE x GRID_SIZE x 1
	const dim3 threads(BLOCK_SIZE, BLOCK_SIZE); // BLOCK_SIZE x BLOCK_SIZE x 1

	matrixMultiplication<<<grid, thread>>> (d_a, d_b, d_c, size);

	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	return 0;
}
