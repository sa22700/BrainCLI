// Matrix_Dot.cu (CUDA-ytimellä)

extern "C"
__global__ void matmul_kernel(const float* A, const float* B, float* C, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i)
            sum += A[row * m + i] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

extern "C"
void matmul(const float* A, const float* B, float* C, int n, int m, int k) {
    float *d_A, *d_B, *d_C;
    size_t size_A = n * m * sizeof(float);
    size_t size_B = m * k * sizeof(float);
    size_t size_C = n * k * sizeof(float);
    cudaMalloc(&d_A, size_A); cudaMalloc(&d_B, size_B); cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 numBlocks((k + 15)/16, (n + 15)/16);
    matmul_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n, m, k);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

