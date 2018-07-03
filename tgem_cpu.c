#include "apsp.h"
#include <stdlib.h>
#include <string.h>

static inline double min(double a, double b) {
  return a < b ? a : b;
}
static inline uint64_t idx(uint64_t n, uint64_t i, uint64_t j) {
  return j + n*i;
}

void tgem(uint64_t n, double *c) {
  double *a = calloc(n*n, sizeof(double));
#pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < n; ++i)
    for (uint64_t j = 0; j < n; ++j)
        a[idx(n,i,j)] = INFINITY;

#ifndef BLOCKING
#pragma omp parallel for collapse(2)
  for (uint64_t k = 0; k < n; ++k)
    for (uint64_t i = 0; i < n; ++i)
      for (uint64_t j = 0; j < n; ++j)
        a[idx(n,i,j)] = min(c[idx(n,i,j)], c[idx(n,i,k)] + c[idx(n,k,j)]);
#else
  uint64_t t = BLOCKING;
#pragma omp parallel for collapse(3)
  for (uint64_t I = 0; I < n; I += t)
    for (uint64_t J = 0; J < n; J += t)
      for (uint64_t K = 0; K < n; K += t)
        for (uint64_t i = I; i < min(I+t, n); i++)
          for (uint64_t j = J; j < min(J+t, n); j++) {
            double x = INFINITY;
            for (uint64_t k = K; k < min(K+t, n); k++)
              x = min(x, c[idx(n,i,k)] + c[idx(n,k,j)]);
            a[idx(n,i,j)] = min(a[idx(n,i,j)] , x);
          }
#endif

  memcpy(c, a, n*n*sizeof(double));
}
