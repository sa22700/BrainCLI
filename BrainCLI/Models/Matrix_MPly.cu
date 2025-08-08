// Matrix_MPly.cu (CUDA-ytimellä)

__global__ void matmply_kernel(const float* A, const float* B, float* C, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    if (idx < total) {
        C[idx] = A[idx] * B[idx];
    }
}

extern "C" 
void matmply(const float* A, const float* B, float* C, int n, int m) {
    float *d_A, *d_B, *d_C;
    int size = n * m * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n * m + blockSize - 1) / blockSize;
    matmply_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n, m);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

