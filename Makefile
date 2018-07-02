
CFLAGS := -std=c99 -fPIC -O3 -Wall -Wextra

libapsp.so : apsp.o
	$(CC) -shared -o $@ $^
