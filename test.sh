#!/bin/bash
set -euo pipefail

cflags='-std=c99 -fPIC -O3 -xcore-avx512 -qopenmp -static-intel'
#cflags="$cflags -qopt-report 5 -qopt-report-phase all"

export OMP_NUM_THREADS=32
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
make -B CC=icc CFLAGS="$cflags -DBLOCKING=64" LIBS='-liomp5 -lpthread'

for i in input/big/V{10,20,50,75,100,125}00-* input/sf
do
  python apsp.py $i
  echo
done
