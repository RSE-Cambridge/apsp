#!/bin/bash
set -euo pipefail

f=input/big/V7500-E1125000

make libapsp.so && rm -f libapsp_gpu.so
time python apsp.py $f

make
time python apsp.py $f
