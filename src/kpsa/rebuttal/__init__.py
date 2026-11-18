"""Rebuttal experiments (AISTATS 2027 reviews).

Importing this package caps the process's CUDA allocation (default 30% of the
device) so that a rebuttal job fails with OOM instead of starving another
long-running job that shares the GPU. Override with KPSA_GPU_FRACTION.
"""

import os

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("KPSA_GPU_FRACTION", "0.3")))
