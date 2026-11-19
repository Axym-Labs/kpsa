#!/usr/bin/env bash
# Audited exclusive-GPU benchmark (v3: frozen encoder, reserved memory, native-dtype hooks, warm construction) (one resident input batch per method). Run alone.
#   cost_bench_v3.jsonl   single-functional methods at N = 200..6,400
#   cost_multi_v3.jsonl   M = 1, 4, 16 functionals
set -u
py="$(dirname "$0")/rebuttal_py.sh"
export KPSA_GPU_FRACTION=0.95
R=~/main/data/kpsa-rebuttal/results
B="-m kpsa.rebuttal.vision_cost_bench"
for rep in 1 2; do
for prec in fp32 bf16; do
  for m in attribution attribution_cpu encoder kpsa_proto ridge_z class_average; do
    $py $B --output $R/cost_bench_v3.jsonl --method $m --precision $prec
  done
  $py $B --output $R/cost_bench_v3.jsonl --method profile_loop --precision $prec
  for n in 200 400 800 1600 3200 6400; do
    $py $B --output $R/cost_bench_v3.jsonl --method knn --k 1 --n $n --precision $prec
    $py $B --output $R/cost_bench_v3.jsonl --method knn --k 16 --n $n --precision $prec
    $py $B --output $R/cost_bench_v3.jsonl --method kpsa_exact --n $n --precision $prec
  done
  for M in 1 4 16; do
    for m in kpsa_multi attribution_multi attribution_multi_vjp direct_multi; do
      $py $B --output $R/cost_multi_v3.jsonl --method $m --k $M --precision $prec
    done
  done
done
done
$py $B --output $R/cost_bench_v3.jsonl --method construction_extras --n 6400
# Batch-size sweep of the memory and throughput ratio (fp32).
for bs in 1 8 32 128; do
  for m in attribution kpsa_proto; do
    $py $B --output $R/cost_batch_sweep_v3.jsonl --method $m --precision fp32 --batch $bs
  done
done
# Batched construction timing on the exclusive GPU.
$py -m kpsa.rebuttal.batched_profiles --output $R/batched_profiles_v3.json --batches 16 32 64
echo FINAL_BENCH_DONE
