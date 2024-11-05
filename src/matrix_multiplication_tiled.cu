// Matrix multiplication using TILING approach as optimization

// Everything starts from the fact that the DRam is slow, so we would like
// to avoid moving data to and from it. The solution is to use the SharedMemory for CUDA (scratchpad)
// A scratchpad is a user-managed L1 memory, private per-thread block
// Scratchpad permits the programmer to write diectly on it and get info in 5/10 cycles instead of 100

// Basically we are wrinting in the cache only pieces of input that are actually needed in that moment and
// not the entire large input
// Given the large input, we define "tile" a subset of it. 

// Given two matrices to multiply, we need to traverse them in two ways:
// 1. Constant row, varying column
// 2. Viceversa
// We get them with A[y][k] * B[k][x], where row is loop invariant for A and col is for B

#define SHMEM_SIZE 16 * 16 * 4  // where 16 is one dimension of a block * size of the type (integer = 4)
								// this is because every thread is loading an element, thus every thread needs 4 bytes

__global__ void matrixMultiplicationTiled(int* d_a, int* d_b, int* d_c, const int size, const int tileSize)
{
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	const int row = blockIdx.y * tileSize + threadIdx.y;
	const int col = blockIdx.x * tileSize + threadIdx.x;

	int tmp = 0;
	
	// Instead of doing a single pass over the matrix in a linear way, we are analyzing it for each tile
	for (int i = 0; i < (n / tileSize); i++)
	{
		// Every thread is threadBlock loads one element into shared memory.
		// The element location in shared memory corresponds to the thread's position in the threadBlock

		// Write inside shared memory

		// Different parameters for this computation
		
		// A
		// row * n = Indexes teh global row for this thread (loop invariant) -> in A we are moving over cols
		// i * tileSize = Indexes the new set of cols for each iteration -> the cols are divided in set (tile)
		// threadIdx.x = Indexes the column within that set -> offset inside the current tile

		// B
		// i * tileSize * n = Indexes the new set of rows each iteration -> current set of rows we have to user
		// threadIdx.y * size = Indexes the row within that set -> offset inside the current tile
		// col = Indexes the global column (loop invariant) -> in B we are moving over rows

		A[threadIdx.y * tileSize + threadIdx.x] = a[row * n + (i * tileSize + threadIdx.x)];
		B[threadIdx.y * tileSize + threadIdx.x] = b[i * tileSize * size + threadIdx.y * size + col];

		__synchThread(); // gate that ensures threads are correctly loaded

		for (int j = 0; j < tileSize; j++)
		{
			tmp += A[threadIdx.y * tileSize + j] * B[j * tileSize + threadIdx.x];
		}

		__ synchThread();
	}

	c[row * size + col] = tmp;
}

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

	const int BLOCK_SIZE = 16;
	const int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);	// Number of blocks in the grid

	const dim3 grid(GRID_SIZE, GRID_SIZE); // Defines a 3D CUDA Matrix of size GRID_SIZE x GRID_SIZE x 1
	const dim3 threads(BLOCK_SIZE, BLOCK_SIZE); // BLOCK_SIZE x BLOCK_SIZE x 1
}