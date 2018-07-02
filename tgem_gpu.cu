#include "apsp.h"

inline __device__ double MIN(double a, double b) {
  return a < b ? a : b;
}
inline __device__ uint64_t idx(uint64_t n, uint64_t i, uint64_t j) {
  return j + n*i;
}

__global__ void tgem_kernel(uint64_t n, double *c) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= n || j >= n) return;

  for (uint64_t k = 0; k < n; ++k)
    c[idx(n,i,j)] = MIN(c[idx(n,i,j)], c[idx(n,i,k)] + c[idx(n,k,i)]);
}

extern "C" {

void tgem(uint64_t n, double *c) {
  uint64_t sz = n*n*sizeof(double);

  double *d_c;
  cudaMalloc((void**)&d_c, sz);
  cudaMemcpy(d_c, c, sz, cudaMemcpyHostToDevice);

  dim3 dimBlock(16, 16), dimGrid;
  dimGrid.x = (n + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (n + dimBlock.y - 1) / dimBlock.y;

  tgem_kernel<<<dimGrid, dimBlock>>>(n, d_c);

  cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost);
}

}
