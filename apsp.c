#include "apsp.h"

inline uint64_t idx(uint64_t n, uint64_t i, uint64_t j) {
  return j + n*i;
}

void init(uint64_t nodes, uint64_t edges, uint64_t *u, uint64_t *v, double *w, double *c) {
  for (uint64_t i = 0; i < nodes; ++i)
    for (uint64_t j = 0; j < nodes; ++j)
      c[idx(nodes,i,j)] = INFINITY;

  for (uint64_t i = 0; i < edges; ++i)
    c[idx(nodes,u[i],v[i])] = w[i];

  for (uint64_t i = 0; i < nodes; ++i)
    c[idx(nodes,i,i)] = 0;
}

void apsp(uint64_t nodes, uint64_t edges, uint64_t *u, uint64_t *v, double *w, double *c) {
  init(nodes, edges, u, v, w, c);
  tgem(nodes, c);
}
