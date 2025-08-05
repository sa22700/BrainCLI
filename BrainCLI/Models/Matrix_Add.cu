// Matrix_Add.cu (CUDA-ytimellä)

extern "C"
__global__ void addmat_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C"
void addmat(const float* A, const float* B, float* C, int n, int m) {
    int size = n * m;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addmat_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, size);
    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
