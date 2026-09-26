# KPSA: Kernelized Parameter-Sensitivity Atlas

KPSA stores a compact, queryable summary of how parameter groups respond to
inputs. For each group, it forms a kernel mean embedding of an input-space
distribution weighted by squared parameter-gradient sensitivity. A frozen
atlas can then retrieve input-conditioned parameter groups without running a
new backward pass for every query.

This repository is the lean experiment supplement for the current paper. It
contains the estimator, the retained vision, forecasting, and protein
experiments, causal interventions, analysis code, and the result-bundle
generator. Historical experiments remain available in Git history; private
specifications, datasets, checkpoints, and result artifacts live in the
sibling `kpsa-internal` repository.

## Evidence in the supplement

The retained code supports two applications:

1. **Input-conditioned localization.** KPSA predicts parameter groups with high
   per-query sensitivity and evaluates them with exact activation ablations.
2. **Input-conditioned parameter-influence circuits.** KPSA retrieves groups
   whose direct parameter perturbation changes the requested output, then tests
   the resulting circuits with causal interventions and interpretability
   controls.

The experiments cover pretrained vision transformers, a pretrained time-series
foundation model, and a pretrained protein transformer. The result bundle also
contains space, kernel, compression, normalization, resolution, atlas-size,
and cost-accuracy ablations used to delimit the claims.

## Layout

- `src/kpsa/kernel_sensitivity.py`: exact and finite-dimensional kernel maps.
- `src/kpsa/representation_sensitivity.py`: atlas statistics, cold-query
  retrieval, coarsening, and ranking metrics.
- `src/kpsa/parameter_groups.py`: complete parameter partitions used by the
  experiments.
- `src/kpsa/vision_*_v8.py` and `src/kpsa/imagenet1k_*_v8.py`: retained vision
  confirmations and design ablations.
- `src/kpsa/timeseries_parameter_influence_v8.py`: forecasting confirmation.
- `src/kpsa/protein_neuron_localization_v8.py`: protein localization and
  parameter-influence confirmation.
- `src/kpsa/parameter_influence_*_v8.py`: causal and interpretability analyses.
- `src/kpsa/positive_evidence_bundle.py`: reproducible tables, figures,
  captions, source-result manifest, and ZIP archive.
- `tests/`: focused unit tests for the retained paper code.

## Environment and verification

The recorded environment is Python 3.12 with PyTorch 2.13.0+cu130 on one RTX
5090. Install the locked environment or the package with its optional
forecasting dependency:

```bash
python -m pip install -e '.[timeseries]'
```

Run the focused supplement tests and lint checks:

```bash
PYTHONPATH=src python -m pytest -q
python -m ruff check src tests
```

## Regenerating the result bundle

The bundle generator consumes the recorded metric files from the private
empirical repositories and performs no model inference:

```bash
PYTHONPATH=src python -m kpsa.positive_evidence_bundle \
  --arc ../kpsa-internal/05_refined_scope \
  --strengthening-arc ../kpsa-internal/06_strenghening \
  --output ../kpsa-internal/05_refined_scope/artifacts/positive_evidence_results \
  --zip ../kpsa-internal/05_refined_scope/artifacts/positive_evidence_results_2026-09-26.zip
```

Key experiment modules expose their full configuration through `--help`:

```bash
PYTHONPATH=src python -m kpsa.imagenet1k_parameter_influence_confirmation_v8 --help
PYTHONPATH=src python -m kpsa.timeseries_parameter_influence_v8 --help
PYTHONPATH=src python -m kpsa.protein_neuron_localization_v8 --help
```

Experiment outputs are deliberately written outside this public codebase.
