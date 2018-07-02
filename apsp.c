#include <stdint.h>
#include <math.h>

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })

#ifndef BLOCKING
void tgem(uint64_t n, double c[n][n]) {
  for (uint64_t k = 0; k < n; ++k)
    for (uint64_t i = 0; i < n; ++i)
      for (uint64_t j = 0; j < n; ++j)
        c[i][j] = min(c[i][j], c[i][k] + c[k][i]);
}
#else
void tgem(uint64_t n, double c[n][n]) {
  uint64_t t = BLOCKING;
#pragma omp parallel for collapse(3)
  for (uint64_t I = 0; I < n; I += t)
    for (uint64_t J = 0; J < n; J += t)
      for (uint64_t K = 0; K < n; K += t)
        for (uint64_t i = I; i < min(I+t, n); i++)
          for (uint64_t j = J; j < min(J+t, n); j++) {
            double x = INFINITY;
            for (uint64_t k = K; k < min(K+t, n); k++)
              x = min(x, c[i][k] + c[k][j]);
            c[i][j] = min(c[i][j], x);
          }
}
#endif

void init(uint64_t nodes, uint64_t edges, uint64_t u[edges], uint64_t v[edges], double w[edges], double c[nodes][nodes]) {
  for (uint64_t i = 0; i < nodes; ++i)
    for (uint64_t j = 0; j < nodes; ++j)
      c[i][j] = INFINITY;

  for (uint64_t i = 0; i < edges; ++i)
    c[u[i]][v[i]] = w[i];

  for (uint64_t i = 0; i < nodes; ++i)
    c[i][i] = 0;
}

void apsp(uint64_t nodes, uint64_t edges, uint64_t u[edges], uint64_t v[edges], double w[edges], double c[nodes][nodes]) {
  init(nodes, edges, u, v, w, c);
  tgem(nodes, c);
}

