from __future__ import print_function

from sys import argv
from ctypes import *
from timeit import default_timer as clock

try:
    fname = argv[1]
except:
    fname = 'input/small/V10-E40'

def write(s, *x):
    print("APSP", s%x)

class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = clock()
        return self
    def __exit__(self, *args):
        self.end = clock()
        write("ELAPSED %s seconds %f", self.name, self.end - self.start)

lib = './libapsp.so'
apsp = cdll.LoadLibrary(lib)
write("LOAD LIBRARY %s", lib)

with Timer("READ"):
    u, v, w = zip(*[line.split() for line in open(fname)])
    u = map(int, u)
    v = map(int, v)
    w = map(float, w)

nodes = max(max(u), max(v)) - min(min(u), min(v)) + 1
write("READ %d NODES", nodes)

edges = len(u)
write("READ %d EDGES", edges)

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
        write("ALLOCATE result array SIZE %d%sB", c_size, unit)
        break
    c_size /= 1024.0

def show_result(c):
    print('%17s '%'--', ' '.join("%17d"%y for y in range(0,nodes)))
    for i, x in zip(range(0,nodes), c):
        print('%17d '%i, ' '.join("%17.2f" % y for y in x))

with Timer("INIT"):
    apsp.init(nodes, edges, u, v, w, c)
#print("INITIAL")
#show_result(c.contents)

with Timer("TGEM"):
    apsp.tgem(nodes, c)

#print("RESULT")
#show_result(c.contents)
