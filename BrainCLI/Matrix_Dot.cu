//
// Created by android on 26.7.2025.
//
// Matrix_Dot.cu

/*
Copyright [2025] [Pirkka Toivakka]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// This project uses model weights licensed under CC BY 4.0 (see /Models/LICENSE)

extern "C" {
    __global__ void matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if(row < N && col < K) {
            float sum = 0.0f;
            for(int i = 0; i < M; ++i)
                sum += A[row * M + i] * B[i * K + col];
            C[row * K + col] = sum;
        }
    }

    void matmul(const float* A, const float* B, float* C, int N, int M, int K) {
        float *d_A, *d_B, *d_C;
        size_t size_A = N * M * sizeof(float);
        size_t size_B = M * K * sizeof(float);
        size_t size_C = N * K * sizeof(float);

        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);

        cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((K + 15)/16, (N + 15)/16);

        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N, M, K);

        cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
}
