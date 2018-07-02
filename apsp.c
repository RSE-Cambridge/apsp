#include <math.h>

void tgemm(int n, double a[n][n], double b[n][n], double c[n][n]) {
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        c[i][j] = fmin(c[i][j], a[i][k] + b[k][j]);
}

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
  tgemm(nodes, c, c, c);
}

