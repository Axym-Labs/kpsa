#!/usr/bin/env bash
# One isolated process per measurement; appends JSON lines to $1. Run alone on the GPU.
set -u
out="$1"; shift
py="$(dirname "$0")/rebuttal_py.sh"
export KPSA_GPU_FRACTION=0.95
B="-m kpsa.rebuttal.vision_cost_bench --output $out"
for rep in 1 2; do
for prec in fp32 bf16; do
  $py $B --method attribution --precision $prec
  $py $B --method attribution_cpu --precision $prec
  $py $B --method encoder --precision $prec
  $py $B --method kpsa_proto --precision $prec
  $py $B --method ridge_z --precision $prec
  $py $B --method class_average --precision $prec
  for n in 800 1600 6400; do $py $B --method knn --k 1 --n $n --precision $prec; done
  $py $B --method knn --k 16 --n 6400 --precision $prec
  $py $B --method kpsa_exact --n 6400 --precision $prec
  $py $B --method profile_loop --precision $prec
done
done
$py $B --method construction_extras --n 6400
echo BENCH_DONE
