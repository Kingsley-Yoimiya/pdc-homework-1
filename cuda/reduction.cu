#include "reduction.cuh"

// naive reduction kernel
// This kernel is barely optimized and serves as a baseline
__global__ void reductionNaive(int *output, const int *input,
                               const int length) {
    int i = blockIdx.x * NAIVE_BLOCK_DIM;
    int partialSum = 0;
    
    for (int j = 0; j < NAIVE_BLOCK_DIM; j++) {
        if (i + j < length) {
            partialSum += input[i + j];
        }
    }
    atomicAdd(output, partialSum);
}

#define LIMITED_KERNEL_LOOP(x, n, step) \
    for (x = (threadIdx.x + blockIdx.x * blockDim.x) * step; x < n; x += step * (gridDim.x * blockDim.x))

__device__ int warpSumReduce(int val) {
    unsigned mask = 0xFFFFFFFF; 
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Your optimized kernel implementations go here
__global__ void reductionOptimized(int *output, const int *input, const int length) {
    __shared__ int smem[OPTIM_BLOCK_SIZE];
    int4 partialSum = {0, 0, 0, 0};
    int x;
    LIMITED_KERNEL_LOOP(x, length - OPTIM_HANDLE_NUM, OPTIM_HANDLE_NUM) {
        partialSum.x += input[x];
        partialSum.y += input[x + 1];
        partialSum.z += input[x + 2];
        partialSum.w += input[x + 3];
    }
    int sum = partialSum.x + partialSum.y + partialSum.z + partialSum.w;
    for(; x < length; x++) sum += input[x];
    sum = warpSumReduce(sum);
    if (threadIdx.x % WARP_SIZE == 0) smem[threadIdx.x / WARP_SIZE] = sum;
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        sum = threadIdx.x < OPTIM_BLOCK_SIZE / WARP_SIZE ? smem[threadIdx.x] : 0;
        sum = warpSumReduce(sum);
        if (threadIdx.x == 0) atomicAdd(output, sum);
    }
}