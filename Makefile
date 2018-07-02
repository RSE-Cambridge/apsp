
NVCC := nvcc
CFLAGS := -std=c99 -qopenmp -fPIC -DBLOCKING=64
NVFLAGS := -std=c++11 --compiler-options '-fPIC' -arch sm_60

all : libapsp.so libapsp_gpu.so

%.o : %.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<

libapsp_gpu.so : CC = $(NVCC)
libapsp_gpu.so : LIBS = -lm
libapsp_gpu.so : apsp.o tgem_gpu.o
	$(CC) $(NVFLAGS) -shared -o $@ $(LIBS) $^

libapsp.so : CC = icc
libapsp.so : LIBS = -liomp5 -lpthread
libapsp.so : apsp.o tgem_cpu.o
	$(CC) -shared -o $@ $(LIBS) $^
