
CFLAGS := -std=c99 -fPIC -DBLOCKING=64

libapsp.so : apsp.o
	$(CC) -shared -o $@ $(LIBS) $^
