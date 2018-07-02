#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif
void apsp(uint64_t nodes, uint64_t edges, uint64_t *u, uint64_t *v, double *w, double *c);
void init(uint64_t nodes, uint64_t edges, uint64_t *u, uint64_t *v, double *w, double *c);
void tgem(uint64_t n, double *c);
#ifdef __cplusplus
}
#endif
