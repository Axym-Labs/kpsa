# Task-space neuronal embeddings

This is the reusable codebase for experiments on compact, queryable
parameter-group sensitivity. The active framework represents each group's
raw or group-normalized squared-gradient sensitivity distribution with a
kernel mean embedding over a shared sample representation. Private task
specifications, checkpoints, metrics, reports, and figures live in the sibling directory
[`../task_embeddings_internal`](../task_embeddings_internal/README.md).

The current arc is
[`../task_embeddings_internal/06_strenghening`](../task_embeddings_internal/06_strenghening/kernelized_parameter_group_sensitivity_v8.md).
Its first implementation milestone provides an exact empirical RBF reference,
a common finite-feature interface, linear and k-means++ Nyström maps, and
discrete raw-versus-normalized sensitivity weights. The retained ImageNet
ViT-B result has been rerun through this interface before new application
search begins.
Only empirical sections and appendices belong in `../task-embeddings-paper/`.

The retained v3 study tested one module atlas across source-object compression,
causal interpretation, structured pruning, and continual-learning protection
in three settings:

- a controlled decoder-only Transformer with known latent task geometry;
- a pretrained DINOv2 ViT on CIFAR-100 (documented fallback because the
  requested DINOv3 checkpoint is access-gated);
- full-parameter Qwen3-1.7B on six heterogeneous instruction tasks.

The v3 blanket **no-go is withdrawn**. Its global-geometry metrics, causal assay,
pruning operating points, and continual-learning protocol contained confounds
that prevented a fair method comparison. The v4 repair adds cross-fitted task
queries, premise-gated necessity/sufficiency curves, a planted positive control,
and genuine seen-only sequential continual learning. The corrected result is
bounded: the Task-Basis Embedding (TBE) helps held-out querying in the positive
control and against matched JL in the naturally trained controlled model, and
residual-normalized TBE enables the controlled applications. It has no
demonstrated downstream advantage over the full residual-normalized OPG atlas
or several alternatives; in the natural model,
task-agnostic and permuted-task-basis rankings also recover almost all of the
held-out query signal. Its controlled value is instead compact utility across
the applications: including the task-feature table, the natural TBE index is
4.90x smaller than the full atlas while retaining about 96--97% of the
full-atlas benefit in the downstream summaries. This does not imply universal
compute savings, and matched-dimensional JL has the same footprint. These
are historical v4 conclusions, not the verdict of the active domain studies. See the
[repair report](../task_embeddings_internal/03_exploratory/artifacts/report.pdf)
and [machine-readable aggregate](../task_embeddings_internal/03_exploratory/artifacts/aggregate_metrics.json).

## Layout

- `src/task_embeddings/experiment_core.py`: canonical estimator,
  representation, application, and run-profile vocabulary;
- `src/task_embeddings/kernel_sensitivity.py`: exact and finite-dimensional
  kernel mean sensitivity indices, including the default k-means++ Nyström map;
- `src/task_embeddings/representation_sensitivity.py`: raw/normalized
  sensitivity weights, streaming sufficient statistics, exact coarsening, and
  retrieval metrics;
- `src/task_embeddings/runner.py`: common `explore`/`paper` experiment entry
  point and output routing;
- `src/task_embeddings/domain_{data,train,optimizer}.py`: reproducible real
  corpora, true scratch training, implicit parameter partitions, and online
  task sketches (no dense parameter-to-group ID arrays);
- `src/task_embeddings/domain_applications.py`: cross-fitted attribution,
  perturbation sensitivity, parameter-budgeted sparsification, and mixed
  precision allocation, with common EF/normalized-OPG/Fisher controls;
- `src/task_embeddings/domain_{campaign,analysis}.py`: resumable finite
  development grids and model-clustered uncertainty; repeated task queries
  are not counted as independent model seeds;
- `src/task_embeddings/domain_continual.py`: sequential domain adaptation,
  parameter-diagonal and exact grouped EWC, acquisition/forgetting curves;
- `src/task_embeddings/domain_scaling.py`: bounded, continuous language-indexed
  SwiGLU scaling on an intact pretrained model. No binary masks or recovery
  metric; full/TBE/JL and task-agnostic controls share the intervention budget.
  This is a validation-only premise test, not established application utility;
- `src/task_embeddings/importance.py`: shared raw/residual-normalized OPG by
  full-atlas/TBE/JL/mean comparison grid;
- `src/task_embeddings/{inference_control,optimizer_experiment,task_optimizer}.py`:
  archival binary feature-masking and checkpoint-recovery optimizer tests;
- `src/task_embeddings/language_inference_control.py`: disjoint-example
  archival Qwen3 feature-masking check using retained profile archives;
- `src/task_embeddings/common.py`: balanced atlases, sketches, fidelity, and
  shared accumulators;
- `src/task_embeddings/applications.py`: common application metrics;
- `src/task_embeddings/*_v3.py`: archival v3 experiment paths;
- `src/task_embeddings/{controlled,controlled_assay,planted,continual,analysis}_v4.py`:
  repaired premise-gated controlled paths;
- `src/task_embeddings/{controlled,vision,language}.py`: retained v1 study
  implementations;
- `tests/`: deterministic unit and tiny end-to-end checks.

## Environment and tests

The recorded environment is Python 3.12 with PyTorch 2.13.0+cu130 on one RTX
5090. Dependencies are listed in `requirements-lock.txt`.

```bash
cd /home/davwis/main/workspace/task_embeddings
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m unittest discover -s tests -v
```

Smoke-test all v3 settings:

```bash
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.controlled_v3 --smoke
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.vision_v3 --smoke
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.language_v3 --smoke
```

Run the active unified paths (`paper` increases the budget, but never certifies
that validity or uncertainty gates passed). Scratch optimization rejects a
checkpoint. `--task-mode single` pools the same domain batches; it does not
change the data to an easier one-domain problem.

```bash
TASK_DATA=../task_embeddings_internal/04_queryable_mechanisms/artifacts/domain_medium/data
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.runner \
  optimizer --profile explore --seed 11 --data "$TASK_DATA" --method adamw
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.runner \
  optimizer --profile paper --seed 21 --data "$TASK_DATA" \
  --method tbe --partition swiglu --score-link log --task-mode multi
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.runner \
  domain_applications --profile explore --seed 11 --data "$TASK_DATA" \
  --checkpoint ../task_embeddings_internal/04_queryable_mechanisms/artifacts/domain_medium/adamw_medium_dev.pt \
  --feature-file token_features.pt
```

Use `--role test` only after freezing choices on validation data. The optional
log-score variant enforces positive reconstructed scores and is reported
separately from linear TBE. Optimizer batch-gradient second moments are **not**
per-example EF. In application profiling, one corpus row is one example.
The older `inference_control` and `language_inference_control` paths below are
archival **task-conditioned pruning** experiments, not continuous scaling.
They do not satisfy the continuous-scaling application requirement.

Masked weights and simulated quantization measure quality at an intervention
budget, not deployed kernel speed. Importance-index savings, total optimizer
state, peak allocation, and wall time are different quantities.

The matched-Adafactor diagnostic uses `mean_adafactor`, `full_adafactor`,
`tbe_adafactor`, and `jl_adafactor`: Adafactor-style parameter scaling, update
clipping and time-dependent second-moment decay, with only the variance
representation changed. It currently requires raw arithmetic second moments.
`optimizer_adafactor_kernel_screen()` is a bounded row-only validation screen;
it does not automatically launch medium confirmation.

The active optimizer also accepts `mean_nomomentum`, `full_nomomentum`,
`tbe_nomomentum`, and `jl_nomomentum`. These set the first-moment coefficient
to zero **without allocating a momentum tensor**; all other grouped-update
mechanics are unchanged. A single task uses one coefficient per group, with
no redundant task-feature slope. Group-size metadata is included in measured
state. The finite momentum-free campaign uses arithmetic (not log) second
moments, validation-only selection, and separate output paths; it does not
replace the earlier Adam-like runs. For example, use `--method mean_nomomentum
--task-mode single --score-link linear` in the optimizer command above.

Use `--skip-pruning` with `domain_applications` to evaluate precision allocation
without rerunning sparsification. Budget selection is ranked first-fit: groups
that do not fit are skipped, and actual parameter fractions are recorded.
`--method adamw8bit` adds the quantized-moment baseline to the same scratch path.
`--method adammini` uses the authors' Adam-mini 1.1.1 implementation, with their
short-run whole-value-tensor setting and weight decay matched on all parameters.
For continuous interventions, `domain_scaling --coordinate gate` profiles the
actual multiplier sensitivities; the default `parameter` coordinate uses mean
parameter OPG. `domain_scaling --translation` measures generated translation
quality with source-conditioned calibration, not just language-model loss.
These native scaling paths remain exploratory; they do not implement pruning.

Retained debugging/archival paths:

```bash
CHECKPOINT=../task_embeddings_internal/03_exploratory/artifacts/controlled/controlled_v4_premise_seed2.pt
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.runner \
  optimizer_recovery --profile explore --seed 2 --checkpoint "$CHECKPOINT"
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.runner \
  inference_control --profile explore --seed 2 --checkpoint "$CHECKPOINT"
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.runner \
  continual --profile explore --seed 2 --order forward
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.runner \
  language_inference_control --profile explore --seed 1 \
  --checkpoint ../task_embeddings_internal/02_exploratory/artifacts/language/language_v3_seed1.pt \
  --profiles ../task_embeddings_internal/02_exploratory/artifacts/language/language_v3_embeddings_seed1.npz
```

The controlled Transformer is a premise/debugging environment, not a
paper-ready model. The shared SwiGLU feature-gating and optimizer group mapping
also support the retained Qwen3 wrapper; promotion of the two new applications
requires the modern pretrained-model profile and independent final data.

Run the repaired controlled experiments and aggregation:

```bash
for seed in 2 3 4; do
  PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.controlled_v4 --seed "$seed"
  PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.controlled_assay_v4 \
    --checkpoint "../task_embeddings_internal/03_exploratory/artifacts/controlled/controlled_v4_premise_seed${seed}.pt" \
    --seed "$seed"
done
for seed in 1 2 3; do
  PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.planted_v4 --seed "$seed"
done
for order in forward reverse; do
  for seed in 1 2 3; do
    PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.continual_v4 --seed "$seed" --full --order "$order"
  done
done
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.analysis_v4
```

Run and regenerate the complete retained study:

```bash
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.controlled_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.vision_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.language_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.analysis_v3
PYTHONPATH=src /home/davwis/main/venvs/vllm-nvfp4/bin/python -m task_embeddings.report_v3
```

The vision path enforces deterministic CUDA execution and freezes only the
DINOv2 positional-embedding tensor because its bicubic interpolation backward
has no deterministic CUDA implementation.

V3 defaults resolve to `../task_embeddings_internal/02_exploratory/artifacts/`;
V4 defaults resolve to `../task_embeddings_internal/03_exploratory/artifacts/`.
Experiment artifacts are deliberately not written into this reusable codebase.
The retained seed-1 natural checkpoints are development attempts and are not
included in confirmatory uncertainty estimates.
