#include "apsp.h"
#include <stdio.h>

static inline __host__ __device__ double MIN(double a, double b) {
  return a < b ? a : b;
}
static inline __host__ __device__ uint64_t idx(uint64_t n, uint64_t i, uint64_t j) {
  return j + n*i;
}

__global__ void init_kernel(uint64_t n, double *a) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= n || j >= n) return;

  a[idx(n,i,j)] = INFINITY;
}

__global__ void tgem_kernel(uint64_t n, double *a, double *c) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= n || j >= n) return;

  for (uint64_t k = 0; k < n; ++k)
    c[idx(n,i,j)] = MIN(c[idx(n,i,j)], a[idx(n,i,k)] + a[idx(n,k,j)]);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" {

void tgem(uint64_t n, double *c) {
  uint64_t sz = n*n*sizeof(double);

  double *d_c, *d_a;
  gpuErrchk( cudaMalloc((void**)&d_a, sz) );
  gpuErrchk( cudaMalloc((void**)&d_c, sz) );
  gpuErrchk( cudaMemcpy(d_a, c, sz, cudaMemcpyHostToDevice) );

  dim3 dimBlock(16, 16), dimGrid;
  dimGrid.x = (n + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (n + dimBlock.y - 1) / dimBlock.y;

  printf("size  %4d\n", sz);
  printf("block %4d %4d\n", dimBlock.x, dimBlock.y);
  printf("grid  %4d %4d\n", dimGrid.x, dimGrid.y);

  init_kernel<<<dimGrid, dimBlock>>>(n, d_c);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  tgem_kernel<<<dimGrid, dimBlock>>>(n, d_a, d_c);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  gpuErrchk( cudaMemcpy(c, d_c, sz, cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaFree(d_a) );
  gpuErrchk( cudaFree(d_c) );
}

}
