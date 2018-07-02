set -euo pipefail

function runall {
  for i in input/big/V{10,20,50,75,100,125}00-* input/sf
  do
    time python single_source_shortest_path.py $i
  done
}

cflags='-std=c99 -fPIC -O3 -parallel -qopt-report 5 -qopt-report-phase all -xcore-avx512 -qopenmp -static-intel'

export OMP_NUM_THREADS=32
make -B CC=icc CFLAGS="$cflags -DBLOCKING=64" LIBS='-liomp5 -lpthread'
runall
