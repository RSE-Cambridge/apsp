#include <math.h>
#include <stdio.h>

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })

#ifndef BLOCKING
void tgem(int n, double c[n][n]) {
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        c[i][j] = min(c[i][j], c[i][k] + c[k][i]);
}
#else
void tgem(int n, double c[n][n]) {
  int t = BLOCKING;
#pragma omp parallel for collapse(3)
  for (int I = 0; I < n; I += t)
    for (int J = 0; J < n; J += t)
      for (int K = 0; K < n; K += t)
        for (int i = I; i < min(I+t, n); i++)
          for (int j = J; j < min(J+t, n); j++) {
            double x = INFINITY;
            for (int k = K; k < min(K+t, n); k++)
              x = min(x, c[i][k] + c[k][j]);
            c[i][j] = min(c[i][j], x);
          }
}
#endif

void init(int nodes, int edges, int u[edges], int v[edges], double w[edges], double c[nodes][nodes]) {
  for (int i = 0; i < nodes; ++i)
    for (int j = 0; j < nodes; ++j)
      c[i][j] = INFINITY;

  for (int i = 0; i < edges; ++i)
    c[u[i]][v[i]] = w[i];

  for (int i = 0; i < nodes; ++i)
    c[i][i] = 0;
}

void apsp(int nodes, int edges, int u[edges], int v[edges], double w[edges], double c[nodes][nodes]) {
  init(nodes, edges, u, v, w, c);
  tgem(nodes, c);
}

