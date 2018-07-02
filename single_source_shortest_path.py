from __future__ import print_function

from sys import argv
from ctypes import *
from time import clock

try:
    fname = argv[1]
except:
    fname = 'input/small/V10-E40'

lib = './libapsp.so'
apsp = cdll.LoadLibrary(lib)
print("LOAD LIBRARY %s" % lib)

u, v, w = zip(*[line.split() for line in open(fname)])
u = map(int, u)
v = map(int, v)
w = map(float, w)

nodes = max(max(u), max(v)) - min(min(u), min(v)) + 1
print("READ %d NODES" % (nodes,))

edges = len(u)
print("READ %d EDGES" % (edges,))

node_t = c_int * edges
weight_t = c_double * edges

u = pointer(node_t(*u))
v = pointer(node_t(*v))
w = pointer(weight_t(*w))

c_t = c_double * nodes * nodes
c = pointer(c_t())

c_size = sizeof(c_t)
for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(c_size) < 1024.0:
        print("ALLOCATE result array SIZE %d%sB" % (c_size, unit))
        break
    c_size /= 1024.0

def show_result(c):
    print('%17s '%'--', ' '.join("%17d"%y for y in range(0,nodes)))
    for i, x in zip(range(0,nodes), c):
        print('%17d '%i, ' '.join("%17.2f" % y for y in x))

apsp.init(nodes, edges, u, v, w, c)
#print("INITIAL")
#show_result(c.contents)

start = clock()

apsp.tgemm(nodes, c, c, c)
#apsp.apsp(nodes, edges, u, v, w, c)

end = clock()

#print("RESULT")
#show_result(c.contents)
print("ELAPSED seconds %f" % (end - start))
